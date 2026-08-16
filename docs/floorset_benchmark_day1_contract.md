# FloorSet Benchmark 建设 Day 1：数据契约与目标矩阵

## 1. Day 1 结论

本日冻结的是 benchmark 的输入契约和生成规则，不是转换器实现，也不是性能结果。针对当前任务，目标契约为模块实例数不少于 200，PinGroup 数为 20,000、30,000、200,000，模块复用率为 60% 或 95%，共 6 档；该矩阵已同步替代主规划文档中旧的 S/M/L 规模目标。所有生成结果必须标记为 **FloorSet-derived synthetic PinAssign benchmark**，不得称为 Intel 原始 PinAssign 数据。

本日未下载 FloorSet 数据，也不生成或验证 20K/30K/200K 目标 case。FloorSet 原始 Lite/Prime 样本是约 21–120 blocks 的 floorplanning case，不能直接满足 200+ module instances、20K/30K/200K PinGroups 或指定复用率；规模放大、模块别名复用和 PinGroup 合成属于 Day 2 之后的适配/优化工作。仓库当前没有本地 FloorSet 样本，因此尚未完成真实 tensor 的数值/填充检查；这列为 Day 2 的阻塞验证项。

## 2. FloorSet 官方字段契约（已由官方代码/README 核对）

官方 Lite loader 返回每个样本的八个张量，且 batch 内会按最大维度补齐；非 bool 张量补值为 `-1`。Lite 的 `fp_sol` 形状为 `n_blocks x 4`，每行是 `[w, h, x, y]`；Prime 的 polygon 形状不同，Day 1 不假设 Prime 可直接转换。

| 官方字段 | 已确认语义 | Day 1 处理 |
| --- | --- | --- |
| `area_target` | 每个 block 的目标面积 | 与 `fp_sol[:,0] * fp_sol[:,1]` 校验；记录误差 |
| `b2b_connectivity` | block-to-block 边及权重/约束编码 | 只把有效、非 padding 边转为候选 net；权重缩放必须进 manifest |
| `p2b_connectivity` | pin/terminal-to-block 边及权重 | 需结合真实样本和 loader 定义核对边的索引编码 |
| `pins_pos` | 外部 pin 坐标，形状为 `n_pins x 2` | 建立稳定 terminal 名；不得把 `-1` padding 当成 pin |
| `placement_constraints` | `n_blocks x 5`：`[fixed, preplaced, multi-instantiation, cluster, boundary]` | 保留原值到 manifest；仅支持已验证的约束进入硬校验 |
| `tree_sol` | Lite 的 B*Tree 标签 | 仅 provenance/诊断，不直接进入当前输入 |
| `fp_sol` | Lite 的 `[w,h,x,y]` 目标布局 | 生成 block polygon `[(x,y),(x+w,y),(x+w,y+h),(x,y+h)]` |
| `metrics_sol` | 面积、pin/net 数、b2b/p2b net 数、约束数及线长指标 | 只作来源统计和转换校验，不伪造 Pin width |

官方 loader 还明确说明不同样本的 blocks/pins 数量不同，padding 应被过滤。当前仓库没有本地样本可用于确认 `p2b_connectivity` 的端点编码、权重列数及 padding 组合，因此这些字段仍需 Day 2 以实际 tensor 验证。

来源： [IntelLabs/FloorSet README](https://github.com/IntelLabs/FloorSet)、[官方 Lite loader](https://github.com/IntelLabs/FloorSet/blob/main/liteLoader.py)、[官方 Lite dataset loader](https://github.com/IntelLabs/FloorSet/blob/main/lite_dataset.py)、[FloorSet 论文](https://arxiv.org/abs/2405.05480)。

## 3. 到当前项目输入的映射

当前 `PlaceDB` 的 `block.json` 是一个递归 Module 对象：每个对象需要 `name`、`module_name`、`vertex`，可选 `direction`、`color`、`children`。`name` 是实例唯一键；Pin 的 `parent_inst` 必须能在该键中查到。当前 `pingroup.json` 顶层是 net 数组；每个 net 是 Pin 对象数组，每个 Pin 至少使用 `parent_inst`、`parent_module`、`pingroup_name`、`scope`、`successors`、`width`。

转换约束如下：

1. block index `i` 映射为稳定实例名 `TOP.B####`，`module_name` 由模块别名表决定；所有 name 和 children 按数值/字典序输出。
2. FloorSet Lite 的 `fp_sol` 生成四边形 `vertex`。坐标单位保持原值；如使用 `width_scale`，必须显式记录公式和输入/输出单位。
3. 同一 module type 的多个实例共享 `module_name`，从而在 `segment.py` 中共享 `module_name:S<index>` 抽象 segment。无复用实例使用唯一 module type。
4. 每条有效 b2b edge 生成一个 net；p2b edge 生成 terminal module 或 root-level terminal（具体表示须先通过 PlaceDB 校验）。不能把没有两个有效端点的 edge 静默写入 net。
5. `parent_module` 必须等于其 `parent_inst` 对应 Module 的 `module_name`。`pingroup_name` 是同构键的一部分：`homology.py` 按 `parent_module.pingroup_name` 聚合 PinGroup。
6. synthetic homology 采用 canonical 端口规则：对每个 module type，按端口类别和确定性 incident-net signature 排序，命名 `PG####`；若两个实例无法证明端口语义一致，必须标记 `synthetic_homology=true`，不能声称是 FloorSet 原始标签。
7. `width` 不从 edge weight 默默复制。Day 2 需选择并固定 `width = max(width_min, weight * width_scale)` 或等价公式，写入 manifest；容量压力由合成层控制。

代码依据： [PlaceDB.py](/C:/Users/DELL/Documents/MCTS/PlaceDB.py)、[homology.py](/C:/Users/DELL/Documents/MCTS/homology.py)、[segment.py](/C:/Users/DELL/Documents/MCTS/segment.py)、[plan_mcts_batches.py](/C:/Users/DELL/Documents/MCTS/plan_mcts_batches.py)。

## 4. 6 档目标 benchmark 矩阵

“模块规模 200+”解释为每个 case 的 module instance 数 `M_inst >= 200`，而不是 module type 数。后续适配器可暂以 `M_inst=240` 作为候选默认值，如源样本组合需要可上调，但不得低于 200；Day 1 不生成该规模。

| Case | PinGroup 数 `G` | 复用率 `ρ` | 默认实例数 | 目的 |
| --- | ---: | ---: | ---: | --- |
| PG20K-R60 | 20,000 | 60% | 240 | 中规模、部分 segment 共享 |
| PG20K-R95 | 20,000 | 95% | 240 | 中规模、高冲突 |
| PG30K-R60 | 30,000 | 60% | 240 | 中高规模、部分共享 |
| PG30K-R95 | 30,000 | 95% | 240 | 中高规模、高冲突 |
| PG200K-R60 | 200,000 | 60% | 240 | 大规模并行压力 |
| PG200K-R95 | 200,000 | 95% | 240 | 大规模极高复用/长尾压力 |

每个 case 还必须有 `case_id`、source sample IDs、生成 seed、converter version、module alias profile、实际 module/net/Pin/PinGroup 计数和资源冲突统计。20K/30K/200K 是 PinGroup 计数，不是 Pin 计数；每个 PinGroup 可包含多个同构 Pin。

## 5. 复用率精确定义与生成规则

定义 module instance 集合 `I`，`M=|I|`；按 `module_name` 分组。复用实例集合为 `I_reused = {i ∈ I : size(group(module_name(i))) >= 2}`，复用率定义为：

```text
ρ = |I_reused| / M
```

因此 60%/95% 是“属于具有至少两个实例的 module type 的实例占全部 module instances 的比例”，不是 module type 数比例，也不是 PinGroup 比例。生成器必须先选 `M_inst >= 200`，再选择 module type 分区，使 `|I_reused|` 精确达到目标整数（允许 manifest 中记录因整数约束产生的实际值，验收误差不超过 `1/M`）。每个复用 type 至少两个实例；未复用实例使用 singleton type。模块复用率与 PinGroup 数相互独立，后者通过端口/edge 生成和确定性扩增控制。

为避免语义混淆，同时记录两个辅助量：`reused_module_type_ratio` 与 `reused_pingroup_pin_ratio`，但它们不替代 `ρ`。

## 6. PinGroup 计数、放大与组合（后续适配，不属于 Day 1 生成）

`G` 定义为 `len(HomologyManager.pin_groups)`，即去重后的 `(parent_module, pingroup_name)` 键数量；不是 `PlaceDB.total_pin_count`，也不是 net 数。生成器完成转换后必须以当前 `HomologyManager` 实际加载值作为最终计数，并在 manifest 中同时写 `pin_count`、`net_count`、`pingroup_count`。

后续放大策略按顺序执行：先选固定的 Lite validation/test source IDs；对每个源样本建立 namespace 和空间 offset；再按 module alias profile 进行跨样本 module type 合并；最后对端口模板和有效 connectivity 做确定性复制/组合，直到目标 `G`。任何复制都必须改变 instance/net/Pin 唯一名，并保留 `source_sample_id`。不得通过修改 JSON 中计数值冒充放大。大 case 先生成 manifest-only 估算，验证内存和磁盘后再落盘。Day 1 只冻结接口与验收规则，**不生成、不验证 20K/30K/200K 文件**。

建议的冲突 profile：R60 使用 60% reused instances，R95 使用 95% reused instances；资源冲突边和波次数量以实际 `module_name:S*` 集合计算，不能预先声称单调结果。每档至少记录 `resource_id_count`、冲突边数、wave_count、最大 wave 并行度。

## 7. 许可、provenance 与确定性

FloorSet 仓库代码为 Apache-2.0；数据集为 CC BY 4.0。生成产物必须附 attribution，记录官方仓库 URL、commit/tag、论文 DOI/arXiv、原始 sample ID、下载时间、数据许可证、转换器版本和本项目配置。不要把未确认的第三方 tensor 数据提交进仓库。

所有随机选择使用显式整数 seed（建议 `sha256(base_seed|case_id|source_id|replicate)` 截断），禁止 Python 内置 `hash()`。JSON 键、数组、module alias、sample ID 和输出文件名稳定排序；同一配置应得到字节一致的 `block.json`、`pingroup.json`、`manifest.json`。原始 FloorSet 文件保留其 provenance，不在本项目中重新发布完整数据。

## 8. Day 1 待验证项与阻塞项

- 尚无真实 Lite validation/test tensor：需确认实际 `n_blocks`、`n_pins`、b2b/p2b edge 维度、padding 和权重列。
- 需确认 p2b terminal 在当前 `PlaceDB` 中的合法表示；否则先做 HPWL-only synthetic benchmark，feedthrough 标记阻塞。
- 需确认 `direction`、polygon 顶点方向和退化 polygon 是否满足当前几何工具。
- 需用真实样本验证 MIB 到 `module_name` 的映射；当前 60%/95% 是生成层定义，不是 FloorSet 标签直接统计。
- 20K/30K/200K 的磁盘、解析内存和批次规划时间尚无实测，不得预写性能结论。

## 9. 规模 gap 与 Day 2+ 适配接口

FloorSet 的原始 block 数上限、原始 pin/net 语义和 MIB/constraint 表达，均不足以直接推导目标矩阵。因此必须分两层：Day 2 先做“单个官方样本 → 当前 JSON”的契约适配和实际 tensor 检查；Day 3+ 再做确定性样本组合、module alias、synthetic homology、PinGroup 扩增和成本估算；目标 case 只有在转换器/校验器通过后才允许生成。任何由复制或合成得到的数量都必须在 manifest 中单独标注，不能作为 FloorSet 原始统计。

## 10. Day 2 接口草案

输入：`floorset_root`、固定 source ID 清单、`case_id`、`target_pingroup_count`、`target_reuse_rate`、`module_instance_count (>=200)`、`base_seed`、`width_scale`、输出目录和 `dry_run`。

输出：`block.json`、`pingroup.json`、`manifest.json`、`provenance.json`、`validation.json`；dry-run 额外输出预计 module/Pin/net/PinGroup 数、冲突边/波次和存储估算。转换器与校验器分离，校验器必须能拒绝缺失 parent instance、padding 泄漏、非法 polygon、重复 full Pin 名和不足两端的 net。

## 11. Day 1 验收清单

- [x] 已阅读当前路线/提示词文档及四个数据契约文件。
- [x] 已核对官方 README、Lite loader、Lite dataset loader、论文和许可证页面。
- [x] 已冻结 20K/30K/200K × 60%/95% 六档目标契约（仅目标，不代表已生成）。
- [x] 已定义复用率与 PinGroup 计数，避免与 Pin 数/net 数混淆。
- [x] 已记录 provenance、seed、输出排序和 synthetic homology 限制。
- [x] 已明确未下载完整数据，未生成/验证目标大规模 case，真实 tensor 检查移交 Day 2。
- [ ] Day 2 获取至少一个官方小样本并保存字段检查报告。
- [ ] Day 2 实现并通过单样本 schema/PlaceDB 校验。
