# FloorSet Benchmark Day 3 实际执行报告

## 完成内容

新增 `floorset_benchmark/scaler.py`，以已转换的单样本 `block`/`pingroup` 对象为输入，提供：

- `ScaleConfig(target_module_count, target_reuse_rate, base_seed, case_id, source_sample_id)`；
- 确定性 module instance 复制和 `TOP.S######` 实例重命名；
- 按 `reused module instances / all non-root module instances` 精确分配 reused module type；
- 按源样本 tile 的 bounding box 做 X 方向平移，并重新计算 root 边界；
- 复用 module type 只分配给同一 source instance/template 的几何等价副本；校验器检查同 module_name 的归一化 polygon 一致；
- 默认 `direction_strategy=cycle8`，按确定性顺序覆盖 PlaceDB 的 0–7 八种方向；`r0-only` 可关闭方向变换；
- 每个副本实际执行正交旋转/镜像后再平移，面积与边长集合保持不变，`direction` 字段与真实变换一致；
- 源 polygon 若已有非 R0 `source_direction`，先用 `canonical_vertex_mapping` 还原基础局部形状，再施加派生方向，避免重复变换。
- 重写 Pin `parent_inst`、`parent_module` 和 net 引用；尾 tile 保留所有端点存在的 net，并记录过滤数；
- 输出独立 `lineage.json`，记录 module/net 的 source、tile、copy、offset、改名和 reuse 分配。

不完整尾 tile 仅过滤无法闭合的 net，不丢弃整个 tile。

复用分配使用按 source template 副本数的确定性可表示性搜索：每个模板选择 0 或 2..c 个副本进入 reused type，无法达到目标时明确拒绝。这样真实 config21 的 76 个源模板复制到 200 个实例时，R95 可精确得到 190 个 reused instances。

## Smoke case

使用现有单样本 converter 的本地 fixture（不是 FloorSet 官方样本，不能作为官方数据结果）生成：

```text
outputs/floorset_day3_smoke_200_r60/
```

实际统计：

| 指标 | 值 |
| --- | ---: |
| module instances（非 root） | 200 |
| reused module instances | 120 |
| `reuse_instance_ratio` | 0.60 |
| reused module types | 60 |
| module types | 140 |
| `reuse_module_type_ratio` | 0.4285714286 |
| derived seed | 7776764255070902238 |

该 smoke case 只验证缩放器的确定性、计数、命名、Pin 引用和基础 JSON 校验，不代表 FloorSet 样本，也不属于 20K/30K/200K 正式矩阵。

## 官方 Day 3 smoke

使用已生成的官方 Lite 单样本输出 `outputs/floorset_day2/config_21_torch`，生成：

```text
outputs/floorset_day3_official_config21_200_r60/
```

统计：200 个非 root module instances、120 个 reused instances、主复用率 0.60、60 个 reused module types、140 个 module types；保留 269 个 net，过滤 28 个无法在尾 tile 闭合的 net。来源 sample 为 `floorset_config21_1`，FloorSet commit 为 `aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa`。

使用 `E:\Anaconda\envs\zyzpytorch\python.exe` 实际加载成功：PlaceDB 报告 201 modules、269 nets、613 pins；SegmentManager 创建 560 个 abstract segments 和 800 个 instance mappings。

另生成非正式 R95 smoke：

```text
outputs/floorset_day3_official_config21_200_r95_smoke/
```

统计为 200 个 module instances、190 个 reused instances、`reuse_instance_ratio=0.95`、73 个 reused module types、83 个 module types；保留 269 个 net、过滤 28 个尾 tile 非闭合 net。使用指定 Anaconda 环境加载成功：PlaceDB 为 201 modules、269 nets、613 pins；SegmentManager 创建 332 个 abstract segments 和 800 个 instance mappings。该 case 仅为 Day3 smoke，不是正式矩阵。

旋转 smoke（均为非正式矩阵）已生成：

- `outputs/floorset_day3_official_config21_200_r60_rotation_smoke/`：200 instances、120 reused、269 retained nets、28 filtered nets，`reuse_module_type_ratio=0.3846153846`。
- `outputs/floorset_day3_official_config21_200_r95_rotation_smoke/`：200 instances、190 reused、269 retained nets、28 filtered nets，`reuse_module_type_ratio=0.8795180723`。

两个旋转 case 的方向分布均为 `{0:25, 1:25, 2:25, 3:25, 4:25, 5:25, 6:25, 7:25}`，manifest/provenance 均写入 FloorSet commit `aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa`。PlaceDB/SegmentManager 加载结果分别为：R60 `201 modules, 269 nets, 613 pins, 520 abstract segments, 800 mappings`；R95 `201 modules, 269 nets, 613 pins, 332 abstract segments, 800 mappings`。

## 测试结果

命令：

```text
E:\Anaconda\envs\zyzpytorch\python.exe -m unittest tests.test_floorset_single_converter -v
```

结果：13 个测试全部通过，包括非 R0 source polygon 先 canonicalize 的回归、76 源模板复制到 200/R95 的回归、八方向覆盖、面积/边长保持、错误 direction/polygon 拒绝、200-module/60% smoke、重复运行一致性、写出 JSON 字节一致性、Pin 引用重写、非法 parent 拒绝、同 module_name 几何不一致拒绝和 target 小于源规模拒绝。

另已通过 `py_compile`。此前 bundled Python 缺少 `shapely`，未用于 PlaceDB 验证；指定 Anaconda 环境已完成官方 smoke 的 PlaceDB/SegmentManager 加载。

## 后续限制

- fixture smoke 仍保留用于单元测试；官方 Day 3 smoke 已使用真实 Day2 输出。
- 当前复用分配采用适合 200-instance smoke 的确定性动态搜索；进入 20K/30K/200K 生成前，Day 4 必须将其替换为内存受控的构造式算法并完成复杂度实测。
- 缩放器尚未实现跨样本 composer、端口语义证明、资源冲突图和 20K/30K/200K 生成。
- tile 内部几何合法性依赖输入 converter；新增 validator 会拒绝同 module_name 的归一化 polygon 不一致。
- Day 4 需要补充独立的 overlap/connection/homology 统计，以及真实单样本输入上的验证。
