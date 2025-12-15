#!/usr/bin/env python3
"""
GPU Burn 測試監控工具
自動啟動 gpu-burn 並記錄 GPU 溫度、記憶體、使用率等指標，產生視覺化圖表

使用方式:
    python gpu_monitor.py --duration 300      # 運行 gpu-burn 並監控 300 秒
    python gpu_monitor.py --duration 10m      # 運行 gpu-burn 並監控 10 分鐘
    python gpu_monitor.py --duration 1h       # 運行 gpu-burn 並監控 1 小時
    python gpu_monitor.py --duration 300 --interval 0.5  # 每 0.5 秒取樣一次
    python gpu_monitor.py --duration 5m --no-burn        # 只監控，不啟動 gpu-burn
    python gpu_monitor.py --duration 5m --gpu-burn-path /opt/gpu-burn/gpu_burn  # 指定 gpu-burn 路徑
"""

import subprocess
import time
import argparse
import csv
import sys
import os
import signal
import shutil
from datetime import datetime
from pathlib import Path
import threading

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安裝，將無法產生圖表")
    print("安裝方式: pip install matplotlib")


def parse_duration(duration_str: str) -> int:
    """解析時間字串，支援秒(s)、分鐘(m)、小時(h)格式"""
    duration_str = str(duration_str).strip().lower()
    
    if duration_str.endswith('h'):
        return int(float(duration_str[:-1]) * 3600)
    elif duration_str.endswith('m'):
        return int(float(duration_str[:-1]) * 60)
    elif duration_str.endswith('s'):
        return int(float(duration_str[:-1]))
    else:
        return int(float(duration_str))


def check_nvidia_smi() -> bool:
    """檢查 nvidia-smi 是否可用"""
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def find_gpu_burn() -> str:
    """尋找 gpu-burn 執行檔路徑"""
    # 常見路徑
    common_paths = [
        'gpu_burn',
        'gpu-burn',
        '/usr/local/bin/gpu_burn',
        '/usr/local/bin/gpu-burn',
        '/opt/gpu-burn/gpu_burn',
        '/opt/gpu_burn/gpu_burn',
        './gpu_burn',
        './gpu-burn',
    ]
    
    # 先檢查 PATH 中是否有
    for cmd in ['gpu_burn', 'gpu-burn']:
        path = shutil.which(cmd)
        if path:
            return path
    
    # 檢查常見路徑
    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    return None


def start_gpu_burn(duration: int, gpu_burn_path: str = None, use_sudo: bool = True) -> subprocess.Popen:
    """啟動 gpu-burn 程序"""
    if gpu_burn_path is None:
        gpu_burn_path = find_gpu_burn()
    
    if gpu_burn_path is None:
        print("⚠️  警告: 找不到 gpu-burn，將只進行監控")
        print("   請確認 gpu-burn 已安裝，或使用 --gpu-burn-path 指定路徑")
        print("   安裝方式: git clone https://github.com/wilicc/gpu-burn && cd gpu-burn && make")
        return None
    
    # 構建命令
    cmd = []
    if use_sudo:
        cmd.append('sudo')
    cmd.extend([gpu_burn_path, str(duration)])
    
    print(f"🔥 啟動 gpu-burn: {' '.join(cmd)}")
    
    try:
        # 使用 Popen 在背景運行
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        return process
    except Exception as e:
        print(f"⚠️  無法啟動 gpu-burn: {e}")
        return None


def stop_gpu_burn(process: subprocess.Popen):
    """停止 gpu-burn 程序"""
    if process is None:
        return
    
    try:
        # 嘗試優雅地終止進程組
        if os.name != 'nt':
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        
        # 等待最多 5 秒
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # 強制終止
            if os.name != 'nt':
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        
        print("🛑 gpu-burn 已停止")
    except Exception as e:
        print(f"⚠️  停止 gpu-burn 時發生錯誤: {e}")


class GpuBurnOutputReader(threading.Thread):
    """背景讀取 gpu-burn 輸出的執行緒"""
    def __init__(self, process: subprocess.Popen, output_path: Path):
        super().__init__(daemon=True)
        self.process = process
        self.output_file = output_path / "gpu_burn_output.log"
        self.lines = []
        self.running = True
    
    def run(self):
        try:
            with open(self.output_file, 'w') as f:
                for line in iter(self.process.stdout.readline, ''):
                    if not self.running:
                        break
                    if line:
                        self.lines.append(line.strip())
                        f.write(line)
                        f.flush()
        except Exception:
            pass
    
    def stop(self):
        self.running = False
    
    def get_last_lines(self, n: int = 3) -> list:
        return self.lines[-n:] if self.lines else []


def get_gpu_info() -> list[dict]:
    """使用 nvidia-smi 取得所有 GPU 的詳細資訊"""
    query_fields = [
        'index',
        'name',
        'temperature.gpu',
        'utilization.gpu',
        'utilization.memory',
        'memory.used',
        'memory.total',
        'memory.free',
        'power.draw',
        'power.limit',
        'clocks.current.graphics',
        'clocks.current.memory',
        'clocks.current.sm',
        'fan.speed',
        'pstate',
        'pcie.link.gen.current',
        'pcie.link.width.current',
    ]
    
    cmd = [
        'nvidia-smi',
        f'--query-gpu={",".join(query_fields)}',
        '--format=csv,noheader,nounits'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        gpus = []
        
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
                
            values = [v.strip() for v in line.split(',')]
            
            def safe_float(val, default=0.0):
                try:
                    if val in ['[N/A]', 'N/A', '[Not Supported]', 'Not Supported', '']:
                        return default
                    return float(val)
                except (ValueError, TypeError):
                    return default
            
            def safe_int(val, default=0):
                try:
                    if val in ['[N/A]', 'N/A', '[Not Supported]', 'Not Supported', '']:
                        return default
                    return int(float(val))
                except (ValueError, TypeError):
                    return default
            
            gpu = {
                'index': safe_int(values[0]),
                'name': values[1] if len(values) > 1 else 'Unknown',
                'temperature': safe_float(values[2]) if len(values) > 2 else 0,
                'gpu_utilization': safe_float(values[3]) if len(values) > 3 else 0,
                'memory_utilization': safe_float(values[4]) if len(values) > 4 else 0,
                'memory_used': safe_float(values[5]) if len(values) > 5 else 0,
                'memory_total': safe_float(values[6]) if len(values) > 6 else 0,
                'memory_free': safe_float(values[7]) if len(values) > 7 else 0,
                'power_draw': safe_float(values[8]) if len(values) > 8 else 0,
                'power_limit': safe_float(values[9]) if len(values) > 9 else 0,
                'clock_graphics': safe_int(values[10]) if len(values) > 10 else 0,
                'clock_memory': safe_int(values[11]) if len(values) > 11 else 0,
                'clock_sm': safe_int(values[12]) if len(values) > 12 else 0,
                'fan_speed': safe_float(values[13]) if len(values) > 13 else 0,
                'pstate': values[14] if len(values) > 14 else 'N/A',
                'pcie_gen': safe_int(values[15]) if len(values) > 15 else 0,
                'pcie_width': safe_int(values[16]) if len(values) > 16 else 0,
            }
            gpus.append(gpu)
        
        return gpus
    except subprocess.CalledProcessError as e:
        print(f"nvidia-smi 執行錯誤: {e}")
        return []


def format_time(seconds: int) -> str:
    """將秒數格式化為易讀的時間字串"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


def print_status(gpu: dict, elapsed: int, total: int):
    """印出即時狀態"""
    progress = elapsed / total * 100
    bar_width = 30
    filled = int(bar_width * elapsed / total)
    bar = '█' * filled + '░' * (bar_width - filled)
    
    # 清除上一行並印出新狀態
    status_lines = [
        f"\r{'─' * 70}",
        f"  GPU {gpu['index']}: {gpu['name']}",
        f"  進度: [{bar}] {progress:5.1f}% ({format_time(elapsed)} / {format_time(total)})",
        f"  🌡️  溫度: {gpu['temperature']:5.1f}°C    ⚡ 功耗: {gpu['power_draw']:6.1f}W / {gpu['power_limit']:.0f}W",
        f"  📊 GPU使用率: {gpu['gpu_utilization']:5.1f}%    💾 記憶體: {gpu['memory_used']:.0f} / {gpu['memory_total']:.0f} MB ({gpu['memory_utilization']:.1f}%)",
        f"  🔧 時脈: Graphics {gpu['clock_graphics']} MHz | Memory {gpu['clock_memory']} MHz | SM {gpu['clock_sm']} MHz",
        f"  🌀 風扇: {gpu['fan_speed']:.0f}%    🔌 PCIe: Gen{gpu['pcie_gen']} x{gpu['pcie_width']}    State: {gpu['pstate']}",
    ]
    
    # 移到最上方並印出
    print('\033[7A', end='')  # 向上移動 7 行
    for line in status_lines:
        print(f"\033[K{line}")  # 清除該行並印出


def monitor_gpus(duration: int, interval: float = 1.0, output_dir: str = None,
                 run_gpu_burn: bool = True, gpu_burn_path: str = None, 
                 use_sudo: bool = True) -> dict:
    """監控 GPU 並記錄數據，可選擇同時運行 gpu-burn"""
    if output_dir is None:
        output_dir = f"gpu_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化資料結構
    data = {
        'timestamps': [],
        'elapsed_seconds': [],
        'gpus': {}
    }
    
    # 取得 GPU 數量
    initial_gpus = get_gpu_info()
    if not initial_gpus:
        print("錯誤: 找不到任何 GPU")
        return data
    
    num_gpus = len(initial_gpus)
    print(f"\n🔍 偵測到 {num_gpus} 個 GPU:")
    for gpu in initial_gpus:
        print(f"   GPU {gpu['index']}: {gpu['name']}")
        data['gpus'][gpu['index']] = {
            'name': gpu['name'],
            'temperature': [],
            'gpu_utilization': [],
            'memory_utilization': [],
            'memory_used': [],
            'memory_total': gpu['memory_total'],
            'power_draw': [],
            'power_limit': gpu['power_limit'],
            'clock_graphics': [],
            'clock_memory': [],
            'clock_sm': [],
            'fan_speed': [],
        }
    
    print(f"\n📋 監控設定:")
    print(f"   持續時間: {format_time(duration)}")
    print(f"   取樣間隔: {interval} 秒")
    print(f"   輸出目錄: {output_path.absolute()}")
    print(f"   GPU Burn: {'啟用' if run_gpu_burn else '停用'}")
    
    # 啟動 gpu-burn
    gpu_burn_process = None
    gpu_burn_reader = None
    
    if run_gpu_burn:
        gpu_burn_process = start_gpu_burn(duration, gpu_burn_path, use_sudo)
        if gpu_burn_process:
            gpu_burn_reader = GpuBurnOutputReader(gpu_burn_process, output_path)
            gpu_burn_reader.start()
            time.sleep(1)  # 給 gpu-burn 一點啟動時間
    
    print(f"\n🚀 開始監控... (Ctrl+C 可提前結束)\n")
    
    # 預留空間給狀態顯示
    for _ in range(7):
        print()
    
    start_time = time.time()
    samples = 0
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed >= duration:
                break
            
            # 檢查 gpu-burn 是否還在運行
            if gpu_burn_process and gpu_burn_process.poll() is not None:
                # gpu-burn 已結束
                pass
            
            # 取得 GPU 資訊
            gpus = get_gpu_info()
            timestamp = datetime.now()
            
            data['timestamps'].append(timestamp)
            data['elapsed_seconds'].append(elapsed)
            
            for gpu in gpus:
                idx = gpu['index']
                if idx in data['gpus']:
                    data['gpus'][idx]['temperature'].append(gpu['temperature'])
                    data['gpus'][idx]['gpu_utilization'].append(gpu['gpu_utilization'])
                    data['gpus'][idx]['memory_utilization'].append(gpu['memory_utilization'])
                    data['gpus'][idx]['memory_used'].append(gpu['memory_used'])
                    data['gpus'][idx]['power_draw'].append(gpu['power_draw'])
                    data['gpus'][idx]['clock_graphics'].append(gpu['clock_graphics'])
                    data['gpus'][idx]['clock_memory'].append(gpu['clock_memory'])
                    data['gpus'][idx]['clock_sm'].append(gpu['clock_sm'])
                    data['gpus'][idx]['fan_speed'].append(gpu['fan_speed'])
            
            # 顯示第一個 GPU 的狀態
            if gpus:
                print_status(gpus[0], int(elapsed), duration)
            
            samples += 1
            
            # 等待下一次取樣
            next_sample_time = start_time + samples * interval
            sleep_time = next_sample_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\n\n⚠️  監控被使用者中斷")
    finally:
        # 停止 gpu-burn
        if gpu_burn_reader:
            gpu_burn_reader.stop()
        if gpu_burn_process:
            stop_gpu_burn(gpu_burn_process)
    
    actual_duration = time.time() - start_time
    print(f"\n\n✅ 監控完成!")
    print(f"   總取樣數: {samples}")
    print(f"   實際運行時間: {format_time(int(actual_duration))}")
    
    # 儲存 CSV
    save_csv(data, output_path)
    
    # 產生圖表
    if HAS_MATPLOTLIB:
        generate_charts(data, output_path)
    
    # 顯示 gpu-burn 最後輸出
    if gpu_burn_reader and gpu_burn_reader.lines:
        print(f"\n📝 gpu-burn 輸出已儲存至: {output_path / 'gpu_burn_output.log'}")
    
    return data


def save_csv(data: dict, output_path: Path):
    """儲存數據為 CSV 檔案"""
    for gpu_idx, gpu_data in data['gpus'].items():
        csv_file = output_path / f"gpu_{gpu_idx}_data.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'elapsed_seconds', 'temperature_c', 
                'gpu_utilization_pct', 'memory_utilization_pct', 
                'memory_used_mb', 'power_draw_w',
                'clock_graphics_mhz', 'clock_memory_mhz', 'clock_sm_mhz',
                'fan_speed_pct'
            ])
            
            for i, ts in enumerate(data['timestamps']):
                writer.writerow([
                    ts.isoformat(),
                    f"{data['elapsed_seconds'][i]:.2f}",
                    gpu_data['temperature'][i],
                    gpu_data['gpu_utilization'][i],
                    gpu_data['memory_utilization'][i],
                    gpu_data['memory_used'][i],
                    gpu_data['power_draw'][i],
                    gpu_data['clock_graphics'][i],
                    gpu_data['clock_memory'][i],
                    gpu_data['clock_sm'][i],
                    gpu_data['fan_speed'][i],
                ])
        
        print(f"   📁 已儲存: {csv_file}")


def generate_charts(data: dict, output_path: Path):
    """產生視覺化圖表"""
    if not data['timestamps']:
        print("警告: 沒有資料可以繪製圖表")
        return
    
    # 設定中文字體 (如果可用)
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    elapsed_minutes = [s / 60 for s in data['elapsed_seconds']]
    
    for gpu_idx, gpu_data in data['gpus'].items():
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'GPU {gpu_idx}: {gpu_data["name"]} - Performance Monitor', 
                     fontsize=14, fontweight='bold')
        
        # 1. 溫度圖
        ax = axes[0, 0]
        ax.plot(elapsed_minutes, gpu_data['temperature'], 'r-', linewidth=1.5, label='Temperature')
        ax.fill_between(elapsed_minutes, gpu_data['temperature'], alpha=0.3, color='red')
        ax.set_ylabel('Temperature (°C)')
        ax.set_xlabel('Time (minutes)')
        ax.set_title('GPU Temperature')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # 標註最高溫度
        max_temp = max(gpu_data['temperature'])
        max_temp_idx = gpu_data['temperature'].index(max_temp)
        ax.annotate(f'Max: {max_temp:.1f}°C', 
                    xy=(elapsed_minutes[max_temp_idx], max_temp),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
        
        # 2. GPU 使用率圖
        ax = axes[0, 1]
        ax.plot(elapsed_minutes, gpu_data['gpu_utilization'], 'g-', linewidth=1.5)
        ax.fill_between(elapsed_minutes, gpu_data['gpu_utilization'], alpha=0.3, color='green')
        ax.set_ylabel('Utilization (%)')
        ax.set_xlabel('Time (minutes)')
        ax.set_title('GPU Utilization')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        
        avg_util = sum(gpu_data['gpu_utilization']) / len(gpu_data['gpu_utilization'])
        ax.axhline(y=avg_util, color='darkgreen', linestyle='--', alpha=0.7, label=f'Avg: {avg_util:.1f}%')
        ax.legend(loc='lower right')
        
        # 3. 記憶體使用圖
        ax = axes[1, 0]
        memory_gb = [m / 1024 for m in gpu_data['memory_used']]
        memory_total_gb = gpu_data['memory_total'] / 1024
        ax.plot(elapsed_minutes, memory_gb, 'b-', linewidth=1.5)
        ax.fill_between(elapsed_minutes, memory_gb, alpha=0.3, color='blue')
        ax.axhline(y=memory_total_gb, color='darkblue', linestyle='--', alpha=0.5, 
                   label=f'Total: {memory_total_gb:.1f} GB')
        ax.set_ylabel('Memory Used (GB)')
        ax.set_xlabel('Time (minutes)')
        ax.set_title('GPU Memory Usage')
        ax.set_ylim(0, memory_total_gb * 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        # 4. 功耗圖
        ax = axes[1, 1]
        ax.plot(elapsed_minutes, gpu_data['power_draw'], 'orange', linewidth=1.5)
        ax.fill_between(elapsed_minutes, gpu_data['power_draw'], alpha=0.3, color='orange')
        if gpu_data['power_limit'] > 0:
            ax.axhline(y=gpu_data['power_limit'], color='red', linestyle='--', alpha=0.5,
                       label=f'Limit: {gpu_data["power_limit"]:.0f}W')
        ax.set_ylabel('Power Draw (W)')
        ax.set_xlabel('Time (minutes)')
        ax.set_title('Power Consumption')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        # 5. 時脈圖
        ax = axes[2, 0]
        ax.plot(elapsed_minutes, gpu_data['clock_graphics'], 'purple', linewidth=1.5, label='Graphics')
        ax.plot(elapsed_minutes, gpu_data['clock_sm'], 'magenta', linewidth=1.5, label='SM', alpha=0.7)
        ax.set_ylabel('Clock Speed (MHz)')
        ax.set_xlabel('Time (minutes)')
        ax.set_title('GPU Clock Speeds')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_ylim(bottom=0)
        
        # 6. 風扇速度圖
        ax = axes[2, 1]
        ax.plot(elapsed_minutes, gpu_data['fan_speed'], 'cyan', linewidth=1.5)
        ax.fill_between(elapsed_minutes, gpu_data['fan_speed'], alpha=0.3, color='cyan')
        ax.set_ylabel('Fan Speed (%)')
        ax.set_xlabel('Time (minutes)')
        ax.set_title('Fan Speed')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 儲存圖表
        chart_file = output_path / f"gpu_{gpu_idx}_chart.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   📊 已儲存圖表: {chart_file}")
    
    # 產生綜合報告圖
    if len(data['gpus']) > 1:
        generate_summary_chart(data, output_path)


def generate_summary_chart(data: dict, output_path: Path):
    """產生多 GPU 比較圖"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-GPU Comparison', fontsize=14, fontweight='bold')
    
    elapsed_minutes = [s / 60 for s in data['elapsed_seconds']]
    colors = plt.cm.tab10(range(len(data['gpus'])))
    
    # 溫度比較
    ax = axes[0, 0]
    for (gpu_idx, gpu_data), color in zip(data['gpus'].items(), colors):
        ax.plot(elapsed_minutes, gpu_data['temperature'], color=color, 
                linewidth=1.5, label=f'GPU {gpu_idx}')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Time (minutes)')
    ax.set_title('Temperature Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 使用率比較
    ax = axes[0, 1]
    for (gpu_idx, gpu_data), color in zip(data['gpus'].items(), colors):
        ax.plot(elapsed_minutes, gpu_data['gpu_utilization'], color=color,
                linewidth=1.5, label=f'GPU {gpu_idx}')
    ax.set_ylabel('Utilization (%)')
    ax.set_xlabel('Time (minutes)')
    ax.set_title('GPU Utilization Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # 功耗比較
    ax = axes[1, 0]
    for (gpu_idx, gpu_data), color in zip(data['gpus'].items(), colors):
        ax.plot(elapsed_minutes, gpu_data['power_draw'], color=color,
                linewidth=1.5, label=f'GPU {gpu_idx}')
    ax.set_ylabel('Power (W)')
    ax.set_xlabel('Time (minutes)')
    ax.set_title('Power Consumption Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 記憶體比較
    ax = axes[1, 1]
    for (gpu_idx, gpu_data), color in zip(data['gpus'].items(), colors):
        memory_gb = [m / 1024 for m in gpu_data['memory_used']]
        ax.plot(elapsed_minutes, memory_gb, color=color,
                linewidth=1.5, label=f'GPU {gpu_idx}')
    ax.set_ylabel('Memory (GB)')
    ax.set_xlabel('Time (minutes)')
    ax.set_title('Memory Usage Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    summary_file = output_path / "multi_gpu_summary.png"
    plt.savefig(summary_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   📊 已儲存綜合圖表: {summary_file}")


def print_summary(data: dict):
    """印出統計摘要"""
    print("\n" + "=" * 70)
    print("📈 統計摘要")
    print("=" * 70)
    
    for gpu_idx, gpu_data in data['gpus'].items():
        if not gpu_data['temperature']:
            continue
            
        print(f"\n🖥️  GPU {gpu_idx}: {gpu_data['name']}")
        print("-" * 50)
        
        # 溫度統計
        temps = gpu_data['temperature']
        print(f"  溫度    : 最低 {min(temps):.1f}°C | 平均 {sum(temps)/len(temps):.1f}°C | 最高 {max(temps):.1f}°C")
        
        # 使用率統計
        utils = gpu_data['gpu_utilization']
        print(f"  GPU使用率: 最低 {min(utils):.1f}% | 平均 {sum(utils)/len(utils):.1f}% | 最高 {max(utils):.1f}%")
        
        # 記憶體統計
        mems = gpu_data['memory_used']
        print(f"  記憶體  : 最低 {min(mems):.0f}MB | 平均 {sum(mems)/len(mems):.0f}MB | 最高 {max(mems):.0f}MB")
        
        # 功耗統計
        powers = gpu_data['power_draw']
        print(f"  功耗    : 最低 {min(powers):.1f}W | 平均 {sum(powers)/len(powers):.1f}W | 最高 {max(powers):.1f}W")
        
        # 風扇統計
        fans = gpu_data['fan_speed']
        if any(f > 0 for f in fans):
            print(f"  風扇    : 最低 {min(fans):.0f}% | 平均 {sum(fans)/len(fans):.0f}% | 最高 {max(fans):.0f}%")


def main():
    parser = argparse.ArgumentParser(
        description='GPU Burn 測試監控工具 - 自動啟動 gpu-burn 並監控 GPU 狀態',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  %(prog)s --duration 300           # 運行 gpu-burn 並監控 300 秒
  %(prog)s --duration 10m           # 運行 gpu-burn 並監控 10 分鐘
  %(prog)s --duration 1h            # 運行 gpu-burn 並監控 1 小時
  %(prog)s -d 5m -i 0.5             # 每 0.5 秒取樣，持續 5 分鐘
  %(prog)s -d 30m -o my_test        # 結果存到 my_test 目錄
  %(prog)s -d 10m --no-burn         # 只監控，不啟動 gpu-burn
  %(prog)s -d 5m --no-sudo          # 不使用 sudo 執行 gpu-burn
  %(prog)s -d 5m --gpu-burn-path /opt/gpu-burn/gpu_burn  # 指定 gpu-burn 路徑
        """
    )
    
    parser.add_argument('-d', '--duration', type=str, required=True,
                        help='監控持續時間 (支援格式: 300, 300s, 10m, 1h)')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                        help='取樣間隔，單位秒 (預設: 1.0)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='輸出目錄名稱 (預設: gpu_monitor_YYYYMMDD_HHMMSS)')
    parser.add_argument('--no-burn', action='store_true',
                        help='不啟動 gpu-burn，只進行監控')
    parser.add_argument('--no-sudo', action='store_true',
                        help='不使用 sudo 執行 gpu-burn')
    parser.add_argument('--gpu-burn-path', type=str, default=None,
                        help='指定 gpu-burn 執行檔路徑')
    
    args = parser.parse_args()
    
    # 檢查 nvidia-smi
    if not check_nvidia_smi():
        print("錯誤: nvidia-smi 不可用，請確認已安裝 NVIDIA 驅動程式")
        sys.exit(1)
    
    # 解析時間
    try:
        duration = parse_duration(args.duration)
    except ValueError:
        print(f"錯誤: 無效的時間格式 '{args.duration}'")
        sys.exit(1)
    
    if duration <= 0:
        print("錯誤: 持續時間必須大於 0")
        sys.exit(1)
    
    if args.interval <= 0:
        print("錯誤: 取樣間隔必須大於 0")
        sys.exit(1)
    
    # 開始監控
    print("\n" + "=" * 70)
    print("🔥 GPU Burn 測試監控工具")
    print("=" * 70)
    
    data = monitor_gpus(
        duration=duration,
        interval=args.interval,
        output_dir=args.output,
        run_gpu_burn=not args.no_burn,
        gpu_burn_path=args.gpu_burn_path,
        use_sudo=not args.no_sudo
    )
    
    # 印出統計摘要
    print_summary(data)
    
    print("\n" + "=" * 70)
    print("✅ 監控完成!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
