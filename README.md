## GPU-analysis-tool

##使用方法

- ツールをダウンロード
- 
``` bash
git clone https://github.com/hsucs-2712/GPU-analysis
```

- gpu-mon.py を gpu_burn のディレクトリに移動

- gpu-mon.py を実行

  ``` bash
  python3 ./gpu-mon.py <time>
  ## 30s ->   python3 ./gpu-mon.py 30 ro python3 ./gpu-mon.py 30s
  ## 5 min -> python3 ./gpu-mon.py 5m
  ## 1 hour -> python3 ./gpu-mon.py 1h
  ```
