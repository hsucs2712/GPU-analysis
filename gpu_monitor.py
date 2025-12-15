#!/usr/bin/env python3
"""
GPU Burn 監控工具
啟動 gpu-burn 壓力測試，同時監控並記錄 GPU 狀態，產生報告與圖表

使用方式:
    python gpu_burn_monitor.py 300          # 運行 300 秒
    python gpu_burn_monitor.py 10m          # 運行 10 分鐘
    python gpu_burn_monitor.py 1h           # 運行 1 小時
"""

import subprocess
import time
import sys
import os
import csv
import signal
from datetime import datetime
from pathlib import Path
from threading import Thread
from queue import Queue

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("提示: pip install matplotlib 可產生圖表")


def parse_duration(s: str) -> int:
    """解析時間: 300, 10m, 1h"""
    s = s.strip().lower()
    if s.endswith('h'):
        return int(float(s[:-1]) * 3600)
    elif s.endswith('m'):
        return int(float(s[:-1]) * 60)
    elif s.endswith('s'):
        return int(float(s[:-1]))
    return int(s)


def get_gpu_stats() -> list[dict]:
    """取得 GPU 狀態"""
    cmd = [
        'nvidia-smi',
        '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit,clocks.current.graphics,fan.speed',
        '--format=csv,noheader,nounits'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            v = [x.strip() for x in line.split(',')]
            
            def num(x, default=0):
                try:
                    return float(x) if x not in ['[N/A]', 'N/A', ''] else default
                except:
                    return default
            
            gpus.append({
                'index': int(v[0]),
                'name': v[1],
                'temp': num(v[2]),
                'gpu_util': num(v[3]),
                'mem_used': num(v[4]),
                'mem_total': num(v[5]),
                'power': num(v[6]),
                'power_limit': num(v[7]),
                'clock': num(v[8]),
                'fan': num(v[9]),
            })
        return gpus
    except Exception as e:
        print(f"nvidia-smi 錯誤: {e}")
        return []


def run_gpu_burn(duration: int, output_queue: Queue):
    """在背景執行 gpu-burn"""
    cmd = f"gpu_burn {duration}"
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in iter(process.stdout.readline, ''):
            if line:
                output_queue.put(line.strip())
        process.wait()
        output_queue.put(None)  # 結束信號
    except Exception as e:
        output_queue.put(f"錯誤: {e}")
        output_queue.put(None)


def generate_report(data: dict, output_dir: Path):
    """產生文字報告"""
    report_file = output_dir / "report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("GPU BURN 測試報告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"測試時間: {data['start_time']}\n")
        f.write(f"持續時間: {data['duration']} 秒\n")
        f.write(f"取樣數: {len(data['timestamps'])}\n\n")
        
        for gpu_id, gpu_data in data['gpus'].items():
            f.write("-" * 60 + "\n")
            f.write(f"GPU {gpu_id}: {gpu_data['name']}\n")
            f.write("-" * 60 + "\n\n")
            
            # 溫度
            temps = gpu_data['temp']
            f.write(f"溫度 (°C):\n")
            f.write(f"  最低: {min(temps):.1f}  平均: {sum(temps)/len(temps):.1f}  最高: {max(temps):.1f}\n\n")
            
            # GPU 使用率
            utils = gpu_data['gpu_util']
            f.write(f"GPU 使用率 (%):\n")
            f.write(f"  最低: {min(utils):.1f}  平均: {sum(utils)/len(utils):.1f}  最高: {max(utils):.1f}\n\n")
            
            # 記憶體
            mems = gpu_data['mem_used']
            f.write(f"記憶體使用 (MB):\n")
            f.write(f"  最低: {min(mems):.0f}  平均: {sum(mems)/len(mems):.0f}  最高: {max(mems):.0f}\n")
            f.write(f"  總容量: {gpu_data['mem_total']:.0f} MB\n\n")
            
            # 功耗
            powers = gpu_data['power']
            f.write(f"功耗 (W):\n")
            f.write(f"  最低: {min(powers):.1f}  平均: {sum(powers)/len(powers):.1f}  最高: {max(powers):.1f}\n")
            f.write(f"  功耗上限: {gpu_data['power_limit']:.0f} W\n\n")
            
            # 時脈
            clocks = gpu_data['clock']
            f.write(f"GPU 時脈 (MHz):\n")
            f.write(f"  最低: {min(clocks):.0f}  平均: {sum(clocks)/len(clocks):.0f}  最高: {max(clocks):.0f}\n\n")
            
            # 風扇
            fans = gpu_data['fan']
            if max(fans) > 0:
                f.write(f"風扇轉速 (%):\n")
                f.write(f"  最低: {min(fans):.0f}  平均: {sum(fans)/len(fans):.0f}  最高: {max(fans):.0f}\n\n")
        
        f.write("=" * 60 + "\n")
    
    print(f"📄 報告已儲存: {report_file}")


def generate_charts(data: dict, output_dir: Path):
    """產生圖表"""
    if not HAS_MATPLOTLIB:
        return
    
    elapsed = data['elapsed']
    
    for gpu_id, gpu_data in data['gpus'].items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"GPU {gpu_id}: {gpu_data['name']}", fontsize=14, fontweight='bold')
        
        # 溫度
        ax = axes[0, 0]
        ax.plot(elapsed, gpu_data['temp'], 'r-', linewidth=1.5)
        ax.fill_between(elapsed, gpu_data['temp'], alpha=0.3, color='red')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature')
        ax.grid(True, alpha=0.3)
        max_temp = max(gpu_data['temp'])
        ax.axhline(y=max_temp, color='darkred', linestyle='--', alpha=0.5, label=f'Max: {max_temp:.1f}°C')
        ax.legend()
        
        # GPU 使用率
        ax = axes[0, 1]
        ax.plot(elapsed, gpu_data['gpu_util'], 'g-', linewidth=1.5)
        ax.fill_between(elapsed, gpu_data['gpu_util'], alpha=0.3, color='green')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Utilization (%)')
        ax.set_title('GPU Utilization')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        
        # 功耗
        ax = axes[1, 0]
        ax.plot(elapsed, gpu_data['power'], 'orange', linewidth=1.5)
        ax.fill_between(elapsed, gpu_data['power'], alpha=0.3, color='orange')
        ax.axhline(y=gpu_data['power_limit'], color='red', linestyle='--', alpha=0.5, label=f'Limit: {gpu_data["power_limit"]:.0f}W')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (W)')
        ax.set_title('Power Consumption')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 記憶體
        ax = axes[1, 1]
        mem_gb = [m / 1024 for m in gpu_data['mem_used']]
        total_gb = gpu_data['mem_total'] / 1024
        ax.plot(elapsed, mem_gb, 'b-', linewidth=1.5)
        ax.fill_between(elapsed, mem_gb, alpha=0.3, color='blue')
        ax.axhline(y=total_gb, color='darkblue', linestyle='--', alpha=0.5, label=f'Total: {total_gb:.1f}GB')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('Memory Usage')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        chart_file = output_dir / f"gpu_{gpu_id}_chart.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 圖表已儲存: {chart_file}")


def save_csv(data: dict, output_dir: Path):
    """儲存 CSV"""
    for gpu_id, gpu_data in data['gpus'].items():
        csv_file = output_dir / f"gpu_{gpu_id}_data.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['elapsed_sec', 'temp_c', 'gpu_util_pct', 'mem_used_mb', 'power_w', 'clock_mhz', 'fan_pct'])
            for i in range(len(data['elapsed'])):
                writer.writerow([
                    f"{data['elapsed'][i]:.1f}",
                    gpu_data['temp'][i],
                    gpu_data['gpu_util'][i],
                    gpu_data['mem_used'][i],
                    gpu_data['power'][i],
                    gpu_data['clock'][i],
                    gpu_data['fan'][i],
                ])
        print(f"📁 CSV 已儲存: {csv_file}")


def main():
    if len(sys.argv) < 2:
        print("使用方式: python gpu_burn_monitor.py <時間>")
        print("範例: python gpu_burn_monitor.py 5m")
        sys.exit(1)
    
    duration = parse_duration(sys.argv[1])
    print(f"\n{'='*60}")
    print(f"🔥 GPU Burn 監控工具")
    print(f"{'='*60}")
    print(f"測試時間: {duration} 秒")
    
    # 檢查 GPU
    gpus = get_gpu_stats()
    if not gpus:
        print("錯誤: 找不到 GPU")
        sys.exit(1)
    
    print(f"偵測到 {len(gpus)} 個 GPU:")
    for g in gpus:
        print(f"  GPU {g['index']}: {g['name']}")
    
    # 建立輸出目錄
    output_dir = Path(f"gpu_burn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(exist_ok=True)
    print(f"輸出目錄: {output_dir}")
    
    # 初始化資料
    data = {
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'duration': duration,
        'timestamps': [],
        'elapsed': [],
        'gpus': {}
    }
    for g in gpus:
        data['gpus'][g['index']] = {
            'name': g['name'],
            'temp': [],
            'gpu_util': [],
            'mem_used': [],
            'mem_total': g['mem_total'],
            'power': [],
            'power_limit': g['power_limit'],
            'clock': [],
            'fan': [],
        }
    
    # 啟動 gpu-burn
    print(f"\n🚀 啟動 gpu_burn {duration}...")
    burn_queue = Queue()
    burn_thread = Thread(target=run_gpu_burn, args=(duration, burn_queue), daemon=True)
    burn_thread.start()
    
    # 監控迴圈
    print(f"📊 開始監控... (Ctrl+C 提前結束)\n")
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration + 5:  # 多等 5 秒確保結束
            elapsed = time.time() - start_time
            gpus = get_gpu_stats()
            
            if gpus:
                data['timestamps'].append(datetime.now())
                data['elapsed'].append(elapsed)
                
                for g in gpus:
                    gd = data['gpus'][g['index']]
                    gd['temp'].append(g['temp'])
                    gd['gpu_util'].append(g['gpu_util'])
                    gd['mem_used'].append(g['mem_used'])
                    gd['power'].append(g['power'])
                    gd['clock'].append(g['clock'])
                    gd['fan'].append(g['fan'])
                
                # 顯示狀態
                g = gpus[0]
                progress = min(elapsed / duration * 100, 100)
                bar = '█' * int(progress / 5) + '░' * (20 - int(progress / 5))
                print(f"\r[{bar}] {progress:5.1f}% | "
                      f"Temp: {g['temp']:5.1f}°C | "
                      f"GPU: {g['gpu_util']:5.1f}% | "
                      f"Mem: {g['mem_used']:.0f}MB | "
                      f"Power: {g['power']:.0f}W", end='')
            
            # 檢查 gpu-burn 輸出
            while not burn_queue.empty():
                msg = burn_queue.get_nowait()
                if msg is None:
                    break
            
            if elapsed >= duration:
                break
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 使用者中斷")
    
    print(f"\n\n✅ 測試完成!")
    print(f"取樣數: {len(data['elapsed'])}")
    
    # 產生輸出
    if data['elapsed']:
        save_csv(data, output_dir)
        generate_report(data, output_dir)
        generate_charts(data, output_dir)
    
    print(f"\n📂 所有檔案已儲存至: {output_dir}")


if __name__ == '__main__':
    main()
