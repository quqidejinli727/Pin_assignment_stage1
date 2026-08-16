# FloorSet Benchmark Day 5：20K 两档生成与验证实验计划

## 1. Day 5 目标

在固定 `12 parent + 228 leaf = 240` 个非 root module instance 的层次结构下，生成并验证两档 20K PinGroup case：

| case | PinGroup 目标 | 复用实例目标 | 复用率目标 | 输出目录 |
|---|---:|---:|---:|---|
| `20k_r60` | 20,000 | 144 | 60% | `outputs/floorset_hierarchical_day5_20k/20k_r60` |
| `20k_r95` | 20,000 | 228 | 95% | `outputs/floorset_hierarchical_day5_20k/20k_r95` |

Day 5 只建设和验证 20K 两档，不开始 30K 或 200K。两档必须使用相同的 FloorSet 源样本、层级布局种子、拓扑主种子和宽度种子；只有复用选择种子允许不同。R60 的复用实例集合应是 R95 复用实例集合的子集，以便做可解释的对照。

## 2. 复用 PinGroup 的连接语义冻结

### 2.1 正确语义

设 `a1/a2` 是复用 module type A 的两个实例上、处于同一模块相对位置的同构 Pin；`b1/b2` 同理属于复用 module type B。`a1` 与 `a2` 使用相同 `pingroup_name`，因此项目中的同构键相同：

```text
homology(a1) = homology(a2) = A.<pingroup_name>
```

同构只要求端口身份、width 和最终模块相对 segment 映射一致，不要求不同实例具有完全相同的连接对象。生成器必须同时覆盖三种模式：

1. `aligned_pair`，多数：

   ```text
   net_1: a1 -> b1
   net_2: a2 -> b2
   ```

2. `dangling_counterpart`，少数：

   ```text
   net_1: a1 -> b1
   net_2: [a2]        # 显式单 Pin Net
   net_3: [b2]        # 显式单 Pin Net
   ```

3. `remapped_counterpart`，少数：

   ```text
   net_1: a1 -> b1
   net_2: a2 -> c1
   ```

   `c1` 可以来自另一个 module type、另一个 parent 或更高层 gateway。若 `b2` 没有其他连接，它应作为显式 singleton Net 保留；若 `b2` 被另一个合法 net 使用，则不得重复创建 singleton。

### 2.2 默认生成比例

先采用保守比例，并把比例写入 manifest，而不是写死在算法中：

| 模式 | 默认占复用连接模板比例 | 20K 验证要求 |
|---|---:|---|
| `aligned_pair` | 85% | 非零且为多数 |
| `dangling_counterpart` | 5% | 非零；两个 singleton 均被记录 |
| `remapped_counterpart` | 10% | 非零；至少覆盖 local、cross-parent、gateway 中两类对端 |

这里的比例以“复用连接模板”为分母，不以 Net、Pin 或 PinGroup 为分母。报告必须同时输出模板数和最终展开后的 Net/Pin 数，避免 singleton 展开导致比例误读。

### 2.3 当前代码是否已经满足

结论：**没有满足，只有底层数据结构具备部分承载能力。**

- `floorset_benchmark/hierarchical_net_synthesizer.py::_build` 当前在 `local_leaf_to_leaf` 中，把复用实例 pair 上的两个同构 Pin 放进同一条 Net 并直接相连。这不是 `a1-b1、a2-b2` 的实例级展开，更没有 `a2-c1` 或 singleton counterpart。
- `PlaceDB.py::load_pingroup_json` 会逐 Net 构建 Pin，并按 `parent_module + "." + pingroup_name` 建立 `pins_by_homology`，因此同一 homology group 的实例 Pin 可以位于不同 Net。
- `homology.py::HomologyManager.get_related_nets` 会汇总一个 homology group 涉及的所有不同 Net，因此 `a1/a2` 分属不同 Net 是受支持的。
- `tests/test_geometry_and_solver.py::test_single_pin_nets_are_skipped_for_metric_feedthrough` 已证明单 Pin Net 可以被 `PlaceDB` 保留，并在最终 HPWL/feedthrough 指标中被显式跳过。
- `floorset_benchmark/single_converter.py::validate_artifacts` 当前把所有 `<2 Pin` 的 Net 判为 `short_net`。Day 5 必须把“有 provenance 的预期 singleton”与意外 short net 区分开，不能直接删除这一保护。

## 3. 20K 之前必须完成的语义前置改造

### 3.1 生成器改造

在 `hierarchical_net_synthesizer.py` 中增加职责清晰的复用连接展开阶段，不用旧的平面补网函数冒充：

```text
select reusable endpoint templates
  -> expand aligned_pair / dangling_counterpart / remapped_counterpart
  -> reserve every full Pin name exactly once
  -> emit explicit successors for nets with degree >= 2
  -> emit [] successors for singleton nets
  -> record pattern provenance
```

每个模板至少记录：

- `reuse_pattern`；
- `homology_group_a`、`homology_group_b`，以及可选的 `homology_group_c`；
- 对应实例 `a1/a2/b1/b2/c1` 的完整 Pin 名；
- 展开后的 net index；
- singleton 原因：`missing_corresponding_connection`；
- topology seed；
- 是否跨 parent、是否经过 gateway；
- width provenance。

### 3.2 singleton Net 合同

只允许以下 singleton：

- lineage 中 `net_kind == "dangling_singleton"`；
- `reason == "missing_corresponding_connection"`；
- 恰好包含一个 Pin；
- `successors == []`；
- 该 Pin 完整名称未出现在其他 Net；
- 该 Pin 是一个至少含两个实例的真实 homology group 成员；
- 该 singleton 有对应的复用模板和已连接 sibling instance，可追溯到 `a1-b1` 一侧。

其他任何 `<2 Pin` Net 继续报错。`validate_artifacts` 应新增可选的 lineage-aware 策略；旧 case、未携带 Day 5 provenance 的 case 仍按原规则拒绝 short net。

单 Pin Net 不参与 HPWL 和 feedthrough 预测，但其 PinGroup 仍必须参与 SegmentManager 容量 packing 和同构 segment 分配。

### 3.3 独立正确性测试

先建立一个最小 fixture，固定包含：

- `a1-b1、a2-b2`；
- `a1-b1、[a2]、[b2]`；
- `a1-b1、a2-c1`；
- `a1-b1、a2-c1、[b2]`；
- `c1` 分别位于同 parent、不同 parent、parent gateway 的变体。

必须断言：

1. `a1/a2` 属于同一 `HomologyManager.pin_groups`；
2. `a1/a2` 可以具有不同 related net 和不同对端；
3. `a1/a2` width 一致；
4. 分配后 `a1/a2` 落在相同 module-relative segment id；
5. 旋转或镜像实例通过方向映射后仍指向对应 segment；
6. singleton 被加载和分配，但 HPWL/feedthrough 统计标记为 skipped；
7. 每个 full Pin name 只出现一次；
8. 非 singleton Net 的 successor 只引用同 Net Pin，且无环；
9. 意外 singleton、无 provenance singleton 和重复 Pin 均被拒绝。

此 fixture 未通过时，不生成 20K。

## 4. Day 5 执行顺序

### 阶段 A：冻结接口和验证合同

主实现和审查使用 `gpt-5.6-sol`、`reasoning_effort=high`：

1. 增加三类复用连接模式配置；
2. 增加 lineage-aware singleton 合同；
3. 增加复用连接语义报告；
4. 增加上述最小 fixture 与负例；
5. 运行现有 FloorSet、层级、旋转、容量和 solver 回归。

只有接口、字段和验收门槛冻结后，批量运行与统计汇总才交给 `gpt-5.6-luna`。

### 阶段 B：1K 双档回归

用新语义重新生成 `1k_r60` 和 `1k_r95`：

- 三种复用模式均非零；
- singleton 合同验证通过；
- PinGroup 精确 1,000；
- hard overflow、unassigned group、missing successor 和 cycle 均为 0；
- 与原有 1K case 的八类层次 Net 覆盖相比不发生退化。

任一失败即停止，不放大到 20K。

### 阶段 C：20K R60

1. 生成到临时目录；
2. 完成结构、层级、复用、Net、successor、同构语义和容量预检；
3. 通过后原子提交为 `20k_r60`；
4. 使用 strict capacity、feedthrough reward 关闭的 correctness smoke 完成全 PinGroup 分配；
5. 独立运行 feedthrough successor 展开校验，但不把 singleton 送入 predictor；
6. 同 seed 完整重跑一次，比较规范化输出哈希。

R60 未通过，不运行 R95。

### 阶段 D：20K R95

按相同流程生成和验证 `20k_r95`。除复用选择外，配置和种子必须与 R60 对齐。最后生成两档汇总对照。

## 5. 每个 case 必须输出的统计信息

### 5.1 身份和可追溯性

- case id、suite version、source sample、source commit；
- base/hierarchy/reuse/topology/width seed；
- `block.json`、`pingroup.json` 和各报告的 SHA-256；
- synthetic/FloorSet-derived Net 数量与比例；
- 配置快照和运行命令。

### 5.2 层级和复用

- root、parent、leaf、nonroot instance 数量；
- module type 总数、复用 module type 数、singleton module type 数；
- 目标/实际复用实例数和复用率；
- 每种 module type 的 instance 数；
- R0/R90/R180/R270/镜像方向直方图；
- parent containment、sibling overlap、root containment 和归一化几何违规数。

### 5.3 Pin、Net 和 PinGroup

- Net、Pin、`HomologyManager.pin_groups` 的精确数量；
- Net degree 的 min/p50/p90/p99/max 和直方图；
- singleton/2-Pin/3-Pin/multicast Net 数；
- 八类层次化 Net 的 count 和 ratio；
- local、child-parent、cross-parent、gateway-chain、multicast 数量；
- gateway Pin、successor edge、连通分量和 fanout 数量；
- missing successor、跨 Net successor、自环、cycle、重复 full Pin 数量。

### 5.4 复用连接语义专项统计

- `reuse_template_count`；
- `aligned_pair_template_count/ratio`；
- `dangling_counterpart_template_count/ratio`；
- `remapped_counterpart_template_count/ratio`；
- `remapped_local/cross_parent/gateway_count`；
- `expected_singleton_net_count` 与 `actual_singleton_net_count`；
- `unprovenanced_singleton_net_count`；
- `singleton_metric_skipped_count`；
- `homology_group_multi_net_count`；
- `same_group_distinct_partner_count` 的分布；
- `homology_width_mismatch_count`；
- `homology_relative_segment_mismatch_count`；
- `full_pin_multi_net_violation_count`；
- `connected_counterpart_missing_lineage_count`；
- `orphan_singleton_count`。

专项明细写入 `reuse_connectivity_report.json`，不能只写汇总布尔值。

### 5.5 width 与容量

- width、pressure 的 count/min/p50/p90/p99/max；
- small/normal/large/stress 数量与比例；
- admission scale；
- 每个 module type 的 demand、capacity、利用率；
- 每个 PinGroup 的初始/最终候选 segment 数；
- 因容量不足削减的候选数；
- packing rejected、unplaceable、hard overflow 数；
- singleton PinGroup 的 demand、候选和分配结果单独统计。

### 5.6 集成与求解

- PlaceDB module/net/pin 数；
- HomologyManager group 数和 multi-net group 数；
- SegmentManager segment、容量和分配数；
- BatchPlanner 分量、批次、wave、资源冲突数；
- feedthrough child net 数、missing successor 数、skipped singleton 数；
- assigned/unassigned Pin、assigned/unassigned PinGroup；
- assignment round、MCTS simulation、assignment issue、capacity violation 数；
- correctness smoke 的 HPWL 只统计 degree >= 2 的 Net，并同时报告 metric net count 与 skipped singleton count。

### 5.7 时间、内存和文件规模

- hierarchy、topology、width calibration、validation、load、batch planning、assignment、summary 各阶段 wall time；
- 总 wall time；
- Python peak allocation；
- 若本机可用则记录进程 peak RSS；
- 每个 JSON 文件大小、总输出大小；
- 重跑哈希比较结果。

## 6. 输出文件

每档 case 至少包含：

```text
block.json
pingroup.json
manifest.json
lineage.json
provenance.json
validation.json
capacity_report.json
hierarchy_report.json
net_distribution_report.json
reuse_connectivity_report.json
case_statistics.json
assignment_smoke.json
```

套件根目录增加：

```text
suite_summary.json
20k_r60_vs_r95.json
```

## 7. 采用门槛

两档必须同时满足：

- 240 个非 root instance；R60 为 144 个复用实例，R95 为 228 个；
- `HomologyManager.pin_groups == 20,000`，不得用 manifest 伪装；
- 三种复用连接模式均非零，且 `aligned_pair` 为多数；
- `dangling_counterpart` 与 `remapped_counterpart` 的实际比例相对配置误差不超过 1 个模板；
- `remapped_counterpart` 至少出现 local 和 cross-parent 两类，gateway 目标若容量允许也必须非零；
- 所有 singleton 均有合法 lineage，`unprovenanced_singleton_net_count == 0`、`orphan_singleton_count == 0`；
- singleton 保留并参与同构/容量分配，但 metric/feedthrough 调用数为 0；
- 同一 homology group width 一致，分配后的 module-relative segment 一致；
- full Pin 跨 Net 重复、missing successor、跨 Net successor、自环和 cycle 均为 0；
- 八类既有层次 Net 均非零；
- small PinGroup 比例不少于 70%；
- packing rejected、unassigned group、capacity violation 和 hard overflow 均为 0；
- PlaceDB、HomologyManager、SegmentManager、BatchPlanner 和 feedthrough successor 展开均通过；
- 同 seed 的规范化构建产物字节一致；
- 20K 的 generation + validation 时间不超过对应 10K case 的 3 倍，峰值 RSS 和产物总大小不超过 2.5 倍；超限记为性能 No-Go，但保留产物和 profile 用于 Day 6 优化。

## 8. 当前 10K 对照基线

Day 5 报告必须使用已经生成的 10K case 作增长对照：

| 指标 | 10K R60 | 10K R95 |
|---|---:|---:|
| nonroot module instances | 240 | 240 |
| reused module instances | 144 | 228 |
| PinGroup | 10,000 | 10,000 |
| Net | 3,811 | 3,811 |
| Pin | 10,478 | 10,478 |
| gateway Net | 2,381 | 2,381 |
| successor edge | 6,667 | 6,667 |
| admission scale | 0.679396 | 0.641527 |
| packing rejected | 0 | 0 |
| solver unassigned group | 0 | 0 |
| hard overflow | 0 | 0 |

这组 10K 数据不包含本计划新增的三类实例级复用连接语义，因此只能作为规模、容量和性能基线，不能作为新语义正确性的基线。

## 9. Day 5 Go/No-Go 决策

- **Go**：语义 fixture、1K 双档回归和 20K R60 全部通过后，再运行 20K R95。
- **修复后重跑**：仅配置比例、精确计数、singleton lineage 或报告字段失败，不改变数据合同即可修复时，保留 seed 并重跑失败阶段。
- **No-Go**：底层 segment 分配无法保证同构实例的 module-relative segment 一致；singleton 被错误送入 feedthrough predictor；20K 容量预检无法在不改模块数量/尺寸的情况下达到 hard overflow 0；或 20K R60 无法精确命中 20,000 PinGroup。No-Go 时不继续 R95，并把失败的最小 counterexample、容量下界和 profile 写入报告。
