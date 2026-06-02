# Feedthrough Predictor

该目录保存 `ftpred` 的源码、Python loader 与 FLUTE 数据文件。新路径下首次
运行且尚未生成 `feedthrough/build` 内的可执行文件时，程序在
`config.py` 中启用 `auto_build_feedthrough=True` 会自动执行：

```powershell
cmake -S feedthrough -B feedthrough/build -G "MinGW Makefiles"
cmake --build feedthrough/build --config Release
```

最终评估只启动一次 `FtpredBinSession`，并在该常驻进程中依次计算所有 net 的
feedthrough。MCTS 的 reward 目前仍仅由 HPWL 决定，feedthrough 权重保持为 `0`。
可执行文件一旦生成，后续运行会直接复用，不会重复调用 CMake。
