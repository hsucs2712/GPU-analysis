## GPU-analysis-tool

##使用方法

- ツールをダウンロード
- 
``` bash
git clone https://github.com/hsucs-2712/GPU-analysis
```

- gpu_monitor.py を gpu_burn のディレクトリに移動

- gpu_monitor.py を実行

  ``` bash
  python3 ./gpu_monitor.py <time>
  ## 30s ->   python3 ./gpu_monitor.py 30 ro python3 ./gpu_monitor.py 30s
  ## 5 min -> python3 ./gpu_monitor.py 5m
  ## 1 hour -> python3 ./gpu_monitor.py 1h
  ```
