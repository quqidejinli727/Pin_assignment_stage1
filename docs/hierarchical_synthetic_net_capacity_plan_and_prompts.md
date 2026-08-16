# 固定模块规模下的层级 PinGroup/Net 扩增、宽度与容量计划

## 1. 结论与边界

后续 benchmark 不再通过持续增加 module instance 来追赶 20K/30K/200K PinGroup，而是把三个控制量解耦：

1. module instance 数固定，v1 默认使用 240 个非 root 实例；
2. PinGroup 数由独立的 hierarchical B2B net synthesizer 精确控制；
3. 复用率继续只由 module type 分区控制，不能用 PinGroup 或 net 数代替。

FloorSet 作为叶子几何、局部 b2b/p2b、MIB/constraint 和连接权重分布的种子；跨层级 module tree、hierarchical port 和跨层 net 必须标记为 synthetic。FloorSet 的 B*Tree 是矩形布局表示，不是设计层级树，不能直接当作 `PlaceDB.children` 层级。

当前 `scaler._append_synthetic_pingroup_nets` 只适合 Day 4 计数校准：它使用固定宽度 `0.001`，主要在同一 module type 的两个实例间补 net，且未写显式 `successors`。它不应继续扩展为正式层级 net 生成器。

## 2. 现有代码能力与必须补齐的语义

- `PlaceDB.parse_module_from_dict` 和 `collect_all_modules` 已递归处理 `children`，Pin 可以引用叶子或非叶 module instance；
- `PlaceDB.load_pingroup_json` 不限制 net 两端的层级，因此同层、跨层、跨父模块 net 都可表达；
- `SegmentManager` 会为所有带 Pin 的 module type 建立抽象 segment，父模块端口可以参与容量分配；
- `HomologyManager` 以 `parent_module.pingroup_name` 建组，容量使用该组的 `max_pin_width`；
- feedthrough loader 只把显式 `successors` 展开为二端 child edges；没有 successors 的 synthetic net 在 feedthrough 路径中会得到 0 条边。

因此必须先冻结以下规则：

- 每条 synthetic net 都写显式、可解析且位于同一 net 内的 `successors`；
- successor 图是有向无环图，可以是一条链或有限 fanout，不使用 Pin 数组顺序推断连接；
- parent pin 表示真实层级边界端口，不是只用于计数的占位 Pin；
- `scope` 可记录层级路径元数据，但当前求解器不依赖 `scope` 判断连接；
- 同一 Pin full name 只能出现在一条 net 中；同一 homology group 内的 Pin width 保持一致。

## 3. 固定 240 实例的层级模板

第一版使用两级层次，便于先验证当前求解器：

```text
TOP
└── 12 个 L1 parent macro
    └── 每个 parent 19 个 L2 leaf block
```

非 root 实例总数为 `12 + 12 × 19 = 240`。FloorSet `TOP.B*` block 用作 leaf 几何模板；转换器产生的 `TOP.IO*` 不再作为普通 leaf module，而是转成 parent/root 边界端口候选。

复用率仍按全部 240 个非 root 实例计算：

- R60：144 reused instances、96 singleton instances；
- R95：228 reused instances、12 singleton instances。

建议 12 个 parent macro 均保持 singleton type。这样 R95 可令 228 个 leaves 全部进入复用 type，R60 可令 144 个 leaves 复用、84 个 leaves singleton，主复用率可以精确命中。任何共享 `module_name` 的实例必须具有相同的逆方向归一化形状。

父模块轮廓以 FloorSet root/较大 block 为几何种子，再按内部 leaf 排布和端口容量定标。leaf 只要求相对 parent 缩小，不预设绝对单位；不能把 leaf 缩放到其最大边小于所分配 PinGroup width。

## 4. 不依赖原始连接的 hierarchical B2B 生成

正式生成器以目标 PinGroup 数为主循环，不以原始 FloorSet edge 数为循环上限。FloorSet 原始局部 b2b 可以保留 10%–20% 作为 local seed，其余连接由稳定 seed 独立生成。

建议按新增 PinGroup 配额而不是 net 数分配 archetype：

| Archetype | 建议占比 | 例子 |
| --- | ---: | --- |
| 同 parent 的 leaf↔leaf | 30% | `P0.C1 → P0.C8` |
| leaf↔所属 parent port | 15% | `P0.C1 → P0.UP` |
| 同一 parent 内经 parent port 的 leaf↔leaf | 15% | `C1 → P0.GW0 → C8` |
| 跨 parent 的 leaf↔leaf 层级链 | 25% | `P0.C1 → P0.UP → P7.DOWN → P7.C3` |
| parent↔parent | 5% | `P2.EAST → P9.WEST` |
| 跨层 multicast/fanout | 10% | 一个 leaf 经多个 parent gateway 到 2–4 个 leaves |

比例是 profile 默认值，不是硬编码真值。生成器应支持 `locality_profile`、`fanout_profile`、`cross_parent_ratio` 和 `floor_set_seed_edge_ratio`。

每条 net 的步骤：

1. 先选择 archetype 和不同的合法 endpoint instance；
2. 为每个层级穿越点创建唯一 parent gateway PinGroup；
3. 生成显式 successor DAG；
4. 统计这条 net 新增的 homology key；
5. 若会超过目标 G，则换用较少 endpoint 的 residual template；
6. 最后用 2-Pin 或 3-Pin residual nets 精确补齐 G，禁止修改 manifest 数字冒充命中。

候选 endpoint 采样混合三种先验：FloorSet 的 inverse-distance/locality、均匀跨 parent、低比例 preferential attachment。必须限制单 module/type degree，避免全部连接集中到少量宏模块。

## 5. FloorSet-first PinGroup width 与容量压力

`50.0` 只是用于说明宽度可能悬殊的示例，不是 benchmark 的固定值或验收目标。真正的目标是：存在少量相对较大的 PinGroup，使 `SegmentManager` 的候选筛选、packing 或提交顺序出现可观测的容量取舍，同时不产生 hard overflow。

先使用 FloorSet 已有信息，避免过度合成。对本地 `config21` 样本的审计结果为：

- b2b edge weight 共 44 个，约从 `5.74e-4` 到 `1.15e-2`，最大值约为最小值的 20 倍；
- p2b edge weight 共 85 个，当前样本中全部约为 `5.74e-4`，本身不能提供大宽度尾部；
- 当前转换器在 `width_scale=0.01`、`width_min=0.001` 下把 228 个转换后 Pin width 全部压成 `0.001`，FloorSet 的相对差异没有保留下来。

FloorSet 的 b2b/p2b 第三列语义是连接权重，不是物理 Pin 宽度。因此只能把其排序和分位数作为 synthetic width 的先验，不能宣称为原始物理宽度。v1 应先修正映射，使较大的 FloorSet 权重单调映射为较大的容量压力；只有实际实验表明该分布无法改变候选或 packing 时，才增加少量并明确标记为 `synthetic_capacity_stress` 的重尾样本。

容量压力使用无量纲比值定义。对于一个 PinGroup `g`，先取得它所有 endpoint module type 可用候选 segment 的最大容量，并取这些上限的最小值为 `C_edge(g)`：

```text
pressure(g) = width(g) / C_edge(g)
C_edge(g) = min(max_candidate_segment_capacity(endpoint_type))
```

建议 v1 目标分布如下；这是初始校准范围，不是必须命中特定绝对宽度：

| 档位 | PinGroup 目标占比 | `pressure` 初始范围 | 放置限制 |
| --- | ---: | --- | --- |
| small | >=70%，默认 75% | `[0.002, 0.02]` | 任意有容量 module |
| normal | 20% 左右 | `(0.02, 0.10]` | leaf 或 parent |
| large | 1%–5% | `(0.10, 0.35]` | 通过真实 segment 预检查后放置 |
| stress | 可选且 <=0.5% | `(0.35, 0.60]` | 仅在 FloorSet-first 映射效果不足时启用，优先 parent gateway |

同一 homology group 内所有 Pin 使用相同 width；同一 successor edge 两端默认同宽。实际 width 由 `pressure * C_edge` 得到，因此随几何和 segment 容量变化，不要求出现 `0.02`、`50.0` 或任何固定极值。最终 manifest 同时输出绝对 width 与 `pressure` 的直方图、p50/p90/p99/max、FloorSet source-weight 分位数、映射方法和是否启用 synthetic tail。

## 6. 容量分配与降级策略

当前容量按 abstract module type 共享，而不是简单按 module instance 分别累加。对每个 module type `t`：

```text
D_t = Σ max_pin_width(group), group.parent_module = t
C_t = Σ capacity(segment), segment.module_name = t
```

`D_t <= C_t` 只是必要条件。正式 admission 使用实际 segment bins 做 first-fit-decreasing 或 best-fit-decreasing：

- 每个 group 至少有一个候选 segment 满足 `segment.capacity >= group.width`；
- soft utilization 目标不超过 75%；
- fallback 后不得超过 90%；
- hard overflow 必须为 0；
- reused type 的所有实例必须保持相同 segment 拓扑和归一化几何。

失败时按以下顺序处理，不直接增加 module：

1. 换到同层级、剩余容量更大的 module type；
2. 将 large/stress group 上移到 parent gateway；
3. 在同一 width 档内重采样较小值；
4. 在预先允许的范围内放大 parent macro，并重新检查 children containment；
5. 若仍不可行，输出 manifest-only No-Go，不写正式 case。

当前 FloorSet `config21` 的真实 block 最大边约为 29，synthetic IO 最大边约为 0.0248，说明绝对 width 与几何单位强相关，不能以固定数值作为跨 case 门槛。parent macro 仍须完成容量定标；synthetic IO 不作为承载 large/stress group 的普通 module。

## 7. 必须输出的校验

### 结构与层级

- 非 root module instance 数精确为 240；
- parent/child 路径唯一，child polygon 被 parent 包含；
- sibling 正面积不重叠；
- parent/root bounds 能包含全部后代；
- 同 module type 的 direction-aware canonical geometry 一致。

### Net 与 PinGroup

- G 精确为 20K/30K/200K，且由 `HomologyManager.pin_groups` 实测；
- 每条 net 至少 2 Pin、至少 1 successor edge；
- successor 只引用同一 net 内存在的 full Pin；
- successor DAG 无环、无自环、无 missing reference；
- 分别报告 local、child-parent、cross-parent、multicast 数量；
- synthetic 与 FloorSet-seeded net 分别统计和追溯。

### Width 与容量

- small group 至少 70%，默认目标 75%，按 `pressure <= 0.02` 判定；
- FloorSet-seeded width 映射不能退化为单值，source weight 与映射 width 保持单调；
- 至少存在一批 `pressure > 0.10` 的大组，并在“全 small 基线”对照下产生非零候选 segment 削减、packing 改变或容量拒绝事件；不要求固定绝对最大 width；
- 每个 module type 输出 demand、capacity、utilization、最大单组 width 和 packing 结果；
- 同时输出绝对 width 与相对 `pressure` 分位数，并区分 FloorSet-first 与 `synthetic_capacity_stress`；
- soft-over-75%、fallback-over-90%、hard overflow 分开报告；
- hard overflow 为 0 才可冻结 case。

### 当前项目集成

- `PlaceDB`、`HomologyManager`、`SegmentManager`、`BatchPlanner` 完整加载；
- feedthrough `build_nets_text_all` 的 `missing_successors` 为空，`child_net_count > 0`；
- HPWL、PinGroup-Net component、batch-resource conflict 和 wave 统计可生成；
- 同 seed 的 block/pingroup/manifest/provenance/lineage 字节一致。

## 8. 分阶段工作与采用门槛

1. **契约与审计**：冻结层级路径、gateway、successor、width、容量和 provenance schema；只读核对 FloorSet 可复用字段。
2. **240-module hierarchy composer**：先生成无新增 net 的两级树，验证 containment、方向、复用率和 PlaceDB/SegmentManager。
3. **独立 net synthesizer**：在固定 240 modules 下做 1K、3K、10K G，验证六类 archetype、successor 和精确计数。
4. **width/capacity allocator**：先修正 FloorSet weight 到 width 的映射并加入相对 pressure profile、parent gateway 和 segment packing；用相同拓扑比较“全 small 基线”和“FloorSet-first 分布”。只有后者未产生可观测容量效应时，才启用少量 synthetic stress tail。
5. **联合校准**：先 20K-R60/R95，再 30K-R60/R95；只有四档 hard overflow=0、missing successor=0、确定性和加载全通过后，才对 200K 做 manifest-only 容量评估。
6. **200K 门槛**：预计 RSS、文件、验证和批次规划均在预算内，分块写出/加载已验证，且容量 packing 预检查通过，才允许真实生成。

## 9. 模型方案

- 架构、层级语义、容量数学和代码接口审查：`gpt-5.6-sol`，`reasoning_effort=high`；遇到跨 feedthrough/MCTS 的高风险设计判断再用 `xhigh`。
- 实现 hierarchy composer、net synthesizer 和 capacity allocator：`gpt-5.6-sol`，`reasoning_effort=high`。
- 大批量确定性生成、重复校验、统计汇总：`gpt-5.6-luna`，`reasoning_effort=medium` 或 `high`；只执行已经冻结的接口和验收，不自行改变语义。
- 独立代码复核可用 `gpt-5.6-terra` high；关键容量/正确性结论仍由 sol 做最终审查。

## 10. 可复用提示词

### 提示词 A：技术设计与 FloorSet 适配审计

```text
基于当前仓库，为固定 240 个非 root module instances 下生成 20K/30K/200K PinGroups 设计层级 synthetic B2B benchmark。先只读检查 PlaceDB.py、homology.py、segment.py、plan_mcts_batches.py、feedthrough/ftpred_loader.py、floorset_benchmark/* 和现有测试；同时核对本地/官方 FloorSet README、loader 和一个真实 Lite 样本。

必须回答并形成设计文档：
1. FloorSet 哪些字段可用于 leaf 几何、局部 b2b/p2b、MIB、权重和 locality 先验；哪些层级连接必须标记 synthetic。不要把 B*Tree 当设计层级。
2. 固定 240 实例的两级 module tree、R60/R95 精确分配、实例命名、parent/child containment 和 direction 规则。
3. 不依赖 FloorSet 原始 edge 数的 net archetype：同 parent leaf-leaf、leaf-parent、parent gateway chain、跨 parent leaf-leaf、parent-parent、multicast；每类给出 successor DAG 示例和配额。
4. G 必须由 HomologyManager 实测，net/pin/G 分开；说明如何用 residual nets 精确命中任意目标。
5. width profile 至少 70% 为相对容量压力不超过 2% 的小组；先保留 FloorSet weight 的排序和分位数，比较全 small 基线，证明较大组是否实际削减候选 segment 或改变 packing。不要要求固定绝对 width；只有 FloorSet-first 映射效果不足时才设计带 provenance 的少量 synthetic stress tail。
6. 按 module type 的 segment packing、75% soft/90% fallback/0 overflow 门槛，以及不可行时的降级顺序。
7. 给出需要新增/修改的具体文件、类、函数、JSON schema、测试和阶段 Go/No-Go。

本阶段不实现代码、不生成正式 20K/30K/200K case。所有判断引用具体代码函数和官方 FloorSet 字段。
```

### 提示词 B：hierarchy composer 与独立 net synthesizer 实现

```text
按已冻结的层级 benchmark 设计实际实现代码。使用 gpt-5.6-sol high。不要继续膨胀 Day4 的 _append_synthetic_pingroup_nets；新增职责独立的 hierarchy_composer、net_synthesizer、width_allocator 和 validator。

实现边界：
- 从真实 FloorSet block 模板生成 TOP/12 parents/每 parent 19 leaves，共 240 个非 root instances；TOP.IO* 只作为边界端口种子，不作为普通 leaf。
- 保持 R60=144/240、R95=228/240，复用 type 只能共享 direction-aware canonical geometry。
- net 生成不受原始 FloorSet edge 数限制，支持六类层级 archetype；每条 synthetic net 写显式 successors，引用只能位于同一 net。
- target_pingroup_count 以 HomologyManager 实测精确命中；Pin、net、PinGroup 分开统计。
- 输出逐 module/net/pin 的 lineage，标明 FloorSet-seeded 或 synthetic、层级类别、source sample 和 seed。
- 先只生成 1K/3K smoke，不生成 20K/30K/200K。

验收：PlaceDB、HomologyManager、SegmentManager、BatchPlanner 和 feedthrough successor 展开均通过；missing successor=0；child_net_count>0；故意损坏 parent、successor、cycle、层级 containment 和重复 full Pin 的测试必须失败；同 seed 全部 JSON 字节一致。完成后报告代码、测试和实际 smoke 数据，不只写计划。
```

### 提示词 C：width/capacity 校准与规模门槛

```text
在固定 240-module hierarchy 和已通过的 synthetic net generator 上实现 width/capacity 校准。使用 gpt-5.6-sol high 完成算法与正确性审查；冻结接口后可让 gpt-5.6-luna 执行重复 case。

要求：
- 先审计 FloorSet b2b/p2b weight 和当前转换 width，修复 `width_min` 导致全部 Pin width 被压成单值的问题。FloorSet weight 只作为 synthetic width 的排序/分位先验，不能误称为原始物理 Pin width。
- 使用 `pressure=width/C_edge` 校准：small >=70% 且默认位于 `[0.002,0.02]`，normal 约 20% 位于 `(0.02,0.10]`，large 1%–5% 位于 `(0.10,0.35]`；仅当 FloorSet-first 分布不能产生容量效应时，允许加入 <=0.5%、`pressure` 位于 `(0.35,0.60]` 的 `synthetic_capacity_stress`。不要求 0.02、50.0 或任何固定绝对 width。
- 同一 homology group width 一致；successor edge 两端默认同宽。
- 用 SegmentManager 的真实 abstract segments 做 per-module-type best-fit/first-fit decreasing packing，报告 demand、capacity、利用率和无法放置组。
- 用同一拓扑和 seed 生成“全 small 基线”与“FloorSet-first 分布”对照；统计候选 segment 数、容量拒绝、packing、利用率和分配结果的变化。若变化已足以形成容量压力，禁止额外增强大宽度尾部。
- soft utilization<=75%，fallback<=90%，hard overflow=0；large/stress 优先 parent gateway。失败时依次换 module、上移 parent、同档降宽、受控放大 parent，仍失败则 No-Go，禁止静默增加 modules。
- 依次运行 1K、3K、10K；通过后才生成 20K-R60/R95 和 30K-R60/R95。200K 先做 manifest-only 容量/RSS/磁盘/规划预算，不满足门槛不得生成。
- 记录每档 G、pins、nets、六类 net 比例、绝对 width 与 pressure 分布、FloorSet/source 映射、容量 packing、候选削减、容量拒绝、HPWL、component、resource conflict、waves、生成/加载/验证/规划时间和 RSS。

采用门槛：G 精确；复用实例误差<=1；small ratio>=70%；width 映射不退化为单值；相对全 small 基线存在非零候选削减、packing 改变或容量拒绝事件；missing successors=0；hard overflow=0；PlaceDB/SegmentManager/BatchPlanner/feedthrough 全通过；同 seed 字节一致。输出机器可读 suite summary 和执行报告。
```
