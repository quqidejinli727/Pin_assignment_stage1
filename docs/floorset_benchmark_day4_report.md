# FloorSet Benchmark Day 4：缩放器校准与中间规模验证

## 范围与持久化证据

Day 4 使用真实 `outputs/floorset_day2/config_21_torch`，只生成 1K/3K calibration smoke；正式 20K/30K/200K 均未生成。可重复 suite runner 为 [day4_suite.py](/C:/Users/DELL/Documents/MCTS/floorset_benchmark/day4_suite.py)，持久化总结果为 `outputs/day4_suite_summary.json`，每档另有 `calibration_report.json`。

独立校验/统计实现于 [calibration.py](/C:/Users/DELL/Documents/MCTS/floorset_benchmark/calibration.py)：polygon overlap 使用 Shapely 正面积相交，边界接触不算重叠；同时检查 schema、parent/module、net endpoint、full Pin、width、canonical homology/direction 和精确 PinGroup。

## 四档实际结果

| Case | Modules | PinGroups | Reused instances | ratio | Pins | Nets | JSON MB | Generation s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K-R60 | 380 | 1,000 | 228/380 | 0.60 | 1,224 | 537 | 0.614 | 0.326 |
| 1K-R95 | 380 | 1,000 | 361/380 | 0.95 | 2,446 | 1,148 | 0.959 | 0.446 |
| 3K-R60 | 1,200 | 3,000 | 720/1,200 | 0.60 | 3,741 | 1,635 | 1.904 | 1.622 |
| 3K-R95 | 1,200 | 3,000 | 1,140/1,200 | 0.95 | 8,309 | 3,919 | 3.198 | 2.037 |

PinGroup 目标均精确命中，复用率误差为 0 个 instance。辅助 `reuse_module_type_ratio` 仍单列，不用于 case 标签。四档 validation 均 `valid=true`、overlap pair=0。

## PinGroup-Net 与资源冲突图

PinGroup-Net 图的 G/N/E 分别是 group nodes、net nodes、unique group-net edges；component 是二部图节点连通分量。资源冲突边定义为两个 batch 的候选 `module_name:S*` resource ID 集合相交；wave 是按稳定 batch 顺序贪心放入资源不相交集合。

| Case | G/N/E | Components | Largest component nodes | Conflict edges | Waves | Max wave batches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K-R60 | 1000/537/1182 | 355 | 21 | 2,833 | 13 | 64 |
| 1K-R95 | 1000/1148/1793 | 761 | 13 | 5,198 | 22 | 68 |
| 3K-R60 | 3000/1635/3674 | 961 | 65 | 8,525 | 13 | 101 |
| 3K-R95 | 3000/3919/5958 | 2477 | 34 | 44,238 | 46 | 83 |

## 持久化成本测量

| Case | PlaceDB s | Segment s | Batch plan s | Validation s | Peak tracemalloc MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1K-R60 | 0.053 | 0.077 | 0.058 | 1.725 | 5.765 |
| 1K-R95 | 0.113 | 0.079 | 0.166 | 2.298 | 8.456 |
| 3K-R60 | 0.189 | 0.260 | 0.321 | 23.565 | 17.673 |
| 3K-R95 | 0.366 | 0.223 | 1.054 | 20.764 | 27.447 |

时间与 bytes 均来自 `day4_suite_summary.json`，不是临时命令输出。tracemalloc 只测 Python allocations，不含 native RSS、进程启动、feedthrough 或 MCTS。

## 保守预测与 Go/No-Go

按 3K 实测值线性外推并统一乘 2 安全系数；时间、磁盘 bytes、峰值 tracemalloc 均乘 2。3K calibration 使用 1,200 modules；正式目标默认 240 modules，因此这是偏保守的 module-scale 预算。validation 时间包含当前 polygon overlap 检查。

| Target | Generation time R60/R95 | JSON bytes R60/R95 | Peak tracemalloc budget R60/R95 | Decision |
| --- | ---: | ---: | ---: | --- |
| 20K | 21.6/27.2 s | 25.4/42.6 MB | 236/366 MB | 预算阶段，可先生成评审 |
| 30K | 32.4/40.7 s | 38.1/64.0 MB | 353/549 MB | 预算阶段，可先生成评审 |
| 200K | 216.2/271.6 s | 254/426 MB | 2.36/3.66 GB | No-Go，先做分块/内存试验 |

200K 继续 No-Go：线性预测可能低估 Python、Shapely、PlaceDB 和批次规划非线性开销；必须先完成 20K/30K 和分块写出/加载验证。

## 测试

```text
E:\Anaconda\envs\zyzpytorch\python.exe -m unittest tests.test_floorset_single_converter tests.test_floorset_calibration -q
```

结果：18 个测试通过，`py_compile` 通过；覆盖精确 PinGroup、同 seed JSON、损坏 parent/width/full Pin、polygon overlap、方向和 homology 校验。
