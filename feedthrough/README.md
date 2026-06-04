# Feedthrough Predictor

该目录保存 `ftpred` 源码、Python loader 和 FLUTE 数据文件。

Stage1 默认不再自动调用 CMake 编译 `ftpred`。运行前请确保可执行文件已经存在：

```powershell
feedthrough/build/ftpred.exe
```

如需手动编译，可在仓库根目录执行：

```powershell
cmake -S feedthrough -B feedthrough/build -G "MinGW Makefiles"
cmake --build feedthrough/build --config Release
```

程序运行时会直接复用已有可执行文件；如果找不到，会报错提示先完成编译，不会在全流程中自动重新编译。
