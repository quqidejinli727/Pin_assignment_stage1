# 并行 MCTS 技术路线、阶段提示词与 FloorSet Benchmark 计划

## 1. 文档目的

本文档保存当前项目并行 MCTS 调研的收敛结论，供后续 Codex 任务直接调用。内容包括：

1. 修正后的两条并行化技术路线；
2. 第三阶段“一周工作与实验规划”提示词；
3. 第四阶段“结果评审与实施决策”提示词；
4. 基于 Intel FloorSet 构建大规模 PinAssign benchmark 的适配思路；
5. FloorSet benchmark 构建提示词与下一周工作目标。

本文档不代表已经完成并行化实现或 FloorSet 数据转换。所有实验结果必须由实际运行产生，不得从方案描述中推断或虚构。

## 2. 已确认的项目事实

- 局部 MCTS 当前串行执行 select、expand、simulate、backpropagate，主要循环位于 `mcts.py::_search_basic()` 和 `mcts.py::_search_hybrid_beam()`。
- `MCTSNode`、`SegmentUsage`、assignments、随机数、统计计数和候选缓存均为可变状态。
- `SegmentUsage.clone()` 可以为树节点建立轻量容量快照，但真实提交仍会修改全局 `AbstractSegment.used_width`、Pin 坐标和同构组状态。
- 外层 `AssignmentSolver` 当前按照 seed group 串行建树，搜索完成后立即提交，再构造下一棵树。
- 默认 reward 同时包含 wirelength 和 feedthrough；`FeedthroughContext` 持有长生命周期外部进程 session，并在单次评估时临时修改 Pin/Net 状态，因此不能跨线程或跨 worker 共享。
- 项目已有 PinGroup-Net 二部图及连通分量规划能力，但连通分量只表达 net 关系，不表达 segment 容量冲突。

## 3. 收敛后的技术路线

### 3.1 主路线 A：资源冲突感知的局部 MCTS 批次/分量多进程并行

#### 核心定义

每个批次的资源集合定义为：

```text
R(batch) = 批次内所有 group 可选 abstract segment_id 的并集
```

两个批次满足以下条件时建立资源冲突边：

```text
R(batch_i) ∩ R(batch_j) != ∅
```

同一执行波次必须满足：

```text
对任意 i != j：R(batch_i) ∩ R(batch_j) = ∅
```

当前代码中 group 只能选择自身 `module_name` 下的 abstract segments，因此首版可以用“两个批次是否包含相同 module_name”作为保守冲突条件。该规则可能减少并行度，但不会漏掉现有候选 segment 冲突。

需要明确区分：

- 批次/分量：冲突图中的一个节点；
- 执行波次：一组资源集合两两不相交、可以并行运行的批次；
- 批次内部：允许多个 group 竞争同一 segment，由单棵 MCTS 的 `SegmentUsage` 管理，不属于跨进程竞争。

#### 正确执行流程

1. 根据 PinGroup-Net 图形成候选批次；
2. 计算每个批次的候选 segment 资源集合；
3. 构建批次资源冲突图；
4. 将图划分为确定性的无冲突执行波次；
5. 波次开始时由主进程生成统一容量快照；
6. worker 使用独立 MCTS、PlaceDB/Pin 状态、reward evaluator 和 feedthrough session；
7. worker 只返回 proposed assignment、reward、计数与计时，不修改主进程状态；
8. 主进程按稳定 `batch_id` 排序结果，重新校验后顺序提交；
9. 波次全部提交后生成下一波次快照。

建议加入运行时硬断言：

```python
for batch_i, batch_j in combinations(current_wave, 2):
    assert resource_ids[batch_i].isdisjoint(resource_ids[batch_j])
```

#### 随机种子

不能继续依赖 `base_seed + assignment_rounds`，因为并发完成顺序会改变 round。使用跨进程稳定派生：

```text
seed = sha256(base_seed, component_id, batch_id, repetition_id) 的固定整数截断
```

禁止使用 Python 内置 `hash()` 作为持久实验种子。

#### 主要风险

- 冲突图构建遗漏候选 segment；
- 调度器把存在冲突边的批次放入同一波次；
- worker 没有使用波次开始时的同一容量快照；
- 结果按完成时间而非稳定 batch 顺序提交；
- 后续新增跨 module 候选或 fallback 策略，但冲突规则未同步更新；
- 进程间重复初始化 PlaceDB、几何数据或 feedthrough session 的开销过大；
- 大批次形成长尾，限制波次实际加速。

#### 采用门槛

必须同时满足：

- worker=4 相对新串行批次版的全流程中位加速不低于 1.5 倍；
- 相同 seed 下并行版与新串行批次版结果一致；
- 相对旧串行求解器，综合质量下降不超过 1%；
- capacity violation、未分配组和强制 overflow 数量不增加；
- feedthrough 并发运行无 session 异常、死锁或结果漂移。

若复用持久 worker 后，worker=4 仍低于 1.3 倍，或质量下降超过 2%，则淘汰路线 A 并检查路线 B 的切换条件。

### 3.2 备选路线 B：单棵局部树同步批量 rollout 多进程并行

第一版只适配 hybrid beam 的同层 children，不修改基础 MCTS 的完整 UCB 循环。

主进程负责：

- 生成 children；
- 确定 rollout 任务及稳定顺序；
- 汇总 reward；
- 更新 visits 和 total_reward；
- 执行 beam 选择和最终提交。

worker 只执行纯 rollout 任务。建议的任务数据为：

```text
RolloutTask:
  batch_id
  tree_depth
  child_id
  rollout_id
  group_index
  assignments
  used_width_snapshot
  task_seed
```

每个 worker 必须拥有独立 evaluator 和 feedthrough session。同步屏障后由主进程统一回传统计，因此不需要共享树或 virtual loss。

路线 B 只有在以下条件同时成立时才启用：

- 路线 A 的 worker=4 全流程加速低于 1.3 倍；
- 单棵大树中 `_simulate + reward` 占运行时间超过 60%；
- 最大批次具有足够多的同层 children 和 rollout 预算。

采用门槛：最大三个批次上 worker=4 rollout 吞吐至少达到 1.8 倍，完整 Stage1 至少达到 1.3 倍，质量下降不超过 1%，进程通信与同步等待低于总时间的 25%。

### 3.3 暂不采用共享树 + virtual loss

当前以下状态都不具备并发安全保证：

- `MCTSNode.children/visits/total_reward/untried_actions`；
- solver 内随机数、计时和计数字典；
- candidate score/cache；
- `FeedthroughContext.session` 和 feedthrough cache；
- feedthrough 评估时临时修改的 Pin 坐标、scope、Net pins 和 feedthrough 字段。

在将 reward 路径改造成不可变数据和并发安全接口之前，不研究共享树并行。

## 4. 第三阶段提示词：一周工作与实验规划

建议模型：`GPT-5.6 Terra`，reasoning `medium`；若需要重新裁决路线，改用 `GPT-5.6 Sol`，reasoning `high`。

```text
请基于当前仓库和已完成的并行 MCTS 技术路线分析，为第一周制定可执行的工作与实验计划。先只读检查代码、现有测试、日志和可用数据，不要立即修改实现。

已确定的路线约束：
1. 第一周主路线是“资源冲突感知的局部 MCTS 批次/分量多进程并行”。
2. 一个批次的资源集合 R(batch) 是其中所有 group 可选 segment_id 的并集。
3. 同一执行波次中的任意两个批次必须满足资源集合不相交。
4. 批次内部的共享容量由单棵 MCTS 的 SegmentUsage 管理，不属于跨进程冲突。
5. worker 不能共享 PlaceDB/Pin 可变状态、RewardEvaluator 或 FeedthroughContext/session。
6. 主进程必须按稳定 batch_id 重新校验并提交结果。
7. 只有路线 A 的 worker=4 全流程加速低于 1.3 倍，且单棵大树的 simulate+reward 占比超过 60%，才允许切换到同步批量 rollout 路线 B。
8. 不考虑共享树和 virtual loss。

请完成：
A. 检查当前批次规划器是否保存了 group_names、net_ids 和候选资源集合；指出缺失接口及具体文件/函数。
B. 设计 serial batched baseline，使它与后续并行版使用完全相同的批次边界、种子和提交规则。
C. 设计资源冲突图、确定性波次划分、运行时不相交断言和提交前容量复检。
D. 设计 Windows 多进程数据边界、持久 worker 初始化方式和每 worker 独立 feedthrough session。
E. 制定 worker=1/2/4/8、至少 3 个固定 seed 的实验矩阵。
F. 明确速度、质量、合法性、可复现性和资源占用指标。
G. 给出每天的目标、涉及文件、产出物、测试命令、验收门槛和失败后的回退路径。

实验必须区分：
- 当前旧串行求解器；
- 新 serial batched baseline；
- 并行 worker=2/4/8；
- 纯 HPWL reward；
- 完整 HPWL + feedthrough reward；
- 固定 simulations 预算，不以时间预算代替主要速度比较。

最低验收标准：
- 相同 seed 重复 3 次，结果 hash 一致；
- 同一波次的批次资源集合两两不相交；
- worker=1 与并行版在相同 seed 下结果一致；
- capacity violation 为 0，未分配组和 forced overflow 不增加；
- worker=4 中位全流程加速至少 1.5 倍；
- 相对旧串行算法综合质量下降不超过 1%。

请把结果同时写入：
- docs/parallel_mcts_week1_plan.md
- docs/parallel_mcts_experiment_matrix.md

如果缺少足够规模的数据，请明确标记为阻塞项，并引用 docs/parallel_mcts_routes_prompts_and_floorset_benchmark.md 中的 FloorSet benchmark 工作，不得虚构性能数据。
```

## 5. 第四阶段提示词：实验结果评审与实施决策

建议模型：`GPT-5.6 Sol`，reasoning `high`。

```text
请作为并行 MCTS 技术负责人，对当前仓库中的实际实验结果进行最终评审。必须读取真实日志、运行报告、配置和输出文件；缺失的数据要明确列出，不得用估计值冒充实验结果。

评审对象限定为：
A. 资源冲突感知的局部 MCTS 批次/分量多进程并行；
B. 单棵 hybrid beam 的同步批量 rollout 多进程并行，仅在 A 满足切换条件时评估。

不要提出第三条路线，不要建议共享树或 virtual loss。

首先验证实验是否公平：
1. 串行和并行是否使用相同输入、批次边界、simulations 预算和 seed 派生规则；
2. worker=1 是否为与并行版同实现的 serial batched baseline；
3. 是否区分纯 HPWL 与完整 feedthrough；
4. 是否包含预热，是否排除一次性数据加载和 worker 启动偏差；
5. 是否至少运行 3 个固定 seed，并报告中位数和离散程度；
6. 是否记录每个波次的关键路径、worker 空闲率和长尾批次；
7. 是否验证结果 hash、容量合法性、未分配组、forced overflow、HPWL 和 feedthrough。

然后按以下硬门槛决策：
- A 采用：worker=4 全流程中位加速 >=1.5x，质量下降 <=1%，合法性零回归，可复现性通过；
- A 淘汰：持久 worker 优化后仍 <1.3x，或质量下降 >2%，或无法稳定运行 feedthrough；
- 切换 B：A <1.3x 且单棵大树 simulate+reward 占比 >60%；
- B 采用：最大三个批次 worker=4 rollout 吞吐 >=1.8x，完整 Stage1 >=1.3x，质量下降 <=1%，IPC+等待 <25%。

输出：
1. 实验完整性结论；
2. 每项硬门槛的实际值、证据文件和通过/失败状态；
3. 性能瓶颈归因；
4. 最终只选择 A、B 或“证据不足，暂不实施”之一；
5. 需要保留的配置、接口和回退开关；
6. 下一阶段实现任务，按 P0/P1/P2 排序；
7. 风险登记表；
8. 一份可评审的 ADR。

请把评审结果写入：
- docs/parallel_mcts_final_review.md
- docs/adr_parallel_mcts.md

如果最终选择实施路线，请先给出文件级修改清单和测试计划；本阶段不要直接大规模重写代码。
```

## 6. FloorSet 用于本项目 Benchmark 的可行性

### 6.1 官方数据事实

Intel Labs 的 FloorSet 官方仓库：<https://github.com/IntelLabs/FloorSet>

FloorSet 论文：<https://arxiv.org/abs/2405.05480>

FloorSet 数据集：<https://huggingface.co/datasets/IntelLabs/FloorSet>

官方信息表明：

- FloorSet-Prime 和 FloorSet-Lite 各包含 100 万训练样本；
- 官方另提供 Prime/Lite 静态测试样本；
- FloorSet floorplan 通常包含 21 到 120 个 blocks；
- Prime 支持 rectilinear polygon，Lite 主要为矩形；
- 输入包括 area target、block-to-block connectivity、pin-to-block connectivity、外部 pin 坐标和 placement constraints；
- 约束包含 fixed、preplaced、multi-instantiation、cluster、boundary；
- 官方仓库代码采用 Apache-2.0，数据集采用 CC BY 4.0；
- 完整数据需要较大存储，官方 README 给出的完整展开需求约 35GB，因此原型阶段不应下载全量数据。

上述 21 到 120 blocks 是 FloorSet 原始单个 floorplan 的常见规模，不满足本项目“200+ 模块实例、20,000/30,000/200,000 PinGroups”的实际测试规模。FloorSet 在本计划中仅提供基础拓扑、几何和约束种子；目标规模必须由后续可追溯的合成缩放层构建。任何报告都不得声称某个 FloorSet 原始样本直接达到本项目目标规模。

### 6.2 与当前 PinAssign 输入的差距

FloorSet 是 floorplanning benchmark，不是直接的 pin-to-segment assignment benchmark。它没有直接提供当前项目需要的：

- 层次化 `block.json`；
- `parent_inst/parent_module/pingroup_name/width` 形式的 Pin；
- 当前项目语义下的同构 PinGroup；
- segment 容量压力分布；
- feedthrough predictor 所需的完整工业语义。

因此必须构建可追溯转换器，不能把 FloorSet tensor 直接交给当前求解器。

### 6.3 建议映射

| FloorSet 字段 | 当前项目字段/用途 | 转换规则 |
| --- | --- | --- |
| `sol` 或 Lite `[w,h,x,y]` | module polygon | 生成平坦 root 下的 block instances；Prime 需去除 polygon padding 并校验顶点顺序 |
| block index | `parent_inst` | 使用确定性实例名，如 `TOP.B0001` |
| MIB group | `parent_module` 复用类型 | 同一 MIB group 映射为同一 module_name；无 MIB block 默认使用唯一 module_name |
| b2b connectivity | `pingroup.json` 中的 net | 每条有效 block-to-block edge 生成一个 net 及其两端 Pin |
| p2b connectivity + pins_pos | 外部 terminal net | 建立显式 IO/terminal module 或合法 root-level terminal instance |
| edge weight | Pin width | 通过记录在 manifest 中的 `width_scale` 转换，禁止隐式缩放 |
| block constraints | benchmark metadata | 保存在 manifest；只有当前求解器真正支持的约束才进入硬验证 |
| 原始 sample id | provenance | 写入每个生成 case 的 manifest 和 attribution 文件 |

同构 PinGroup 的构造必须显式说明规则。建议对同一 MIB module type 的实例，按照端口类别和确定性 incident-edge 排序生成 canonical port 名称；如果无法证明端口语义一致，则将其标记为“synthetic homology”，不能声称为原始 FloorSet 标签。

### 6.4 Benchmark 层次

#### Level 0：转换冒烟集

- 使用 FloorSet-Lite validation/test 的少量样本；
- 覆盖 21、约 60、约 120 blocks；
- 只验证 schema、几何、Pin/Net 完整性和求解器能否启动；
- 不用于性能结论，也不视为目标测试规模的缩小版代表。

#### Level 1：标准单样本集

- 使用官方可公开验证的 100 个 Lite 样本；
- 按 block_count、net_count、MIB 比例和约束数量分桶；
- 每桶固定样本 ID，不随机挑选；
- 用于功能正确性、质量趋势和小规模回归。

#### Level 2：大规模组合压力集

将多个 FloorSet 样本作为基础拓扑和几何种子，在同一个合成 root 下进行确定性组合、复制、扰动与连接扩展；使用空间 offset 保持几何合法，并保留每个生成对象到源样本的 provenance。此层产物统一称为 **FloorSet-derived synthetic PinAssign benchmark**，不能称为 FloorSet 原始 benchmark，也不能暗示原始样本已具备目标规模。

目标矩阵以生成后的实际 PinGroup 数和复用模块实例比例定义，共六档：

| Case ID | 非 root 模块实例数 | PinGroup 数 | 目标复用模块实例比例 |
| --- | ---: | ---: | ---: |
| `PG20K-R60` | >= 200 | 20,000 | 60% |
| `PG20K-R95` | >= 200 | 20,000 | 95% |
| `PG30K-R60` | >= 200 | 30,000 | 60% |
| `PG30K-R95` | >= 200 | 30,000 | 95% |
| `PG200K-R60` | >= 200 | 200,000 | 60% |
| `PG200K-R95` | >= 200 | 200,000 | 95% |

这里的“模块规模 200+”固定解释为生成 case 中 **非 root 模块实例总数不少于 200**，不得用 module type 数替代。

复用率主指标严格定义如下：令 `I` 为全部非 root 模块实例集合，`module_name(i)` 为实例 `i` 的模块类型名；若某个 `module_name` 在 `I` 中出现至少 2 次，则属于该类型的所有实例均为 reused module instances。

```text
reuse_instance_ratio =
  |{i in I : count(module_name(i), I) >= 2}| / |I|
```

六档 case 名中的 60%/95% 只指该主指标。因实例数取整，生成值允许相对目标最多偏差 1 个实例，并必须同时报告目标值、实际分子、实际分母和实际比例。

辅助指标 `reuse_module_type_ratio` 定义为“出现至少 2 次的不同 `module_name` 数 / 全部不同 `module_name` 数”。该指标只用于描述类型结构，不得代替主指标，不得与 60%/95% 标签混用。

复用率会影响 segment 资源共享，但不等同于批次资源冲突率。每个 case 仍须独立报告资源冲突边数、波次数、最大可并行批次数和长尾批次分布，不得用复用率直接推断冲突强度。

### 6.5 每个生成 case 的必需产物

```text
case_<id>/
  block.json
  pingroup.json
  manifest.json
  provenance.json
  validation.json
```

`manifest.json` 至少包含：

- converter version 和代码提交号；
- FloorSet 来源、版本/commit、sample IDs；
- 数据许可和 attribution；
- 生成 seed；
- width scale；
- 目标和实际 `reuse_instance_ratio`（含分子、分母）以及辅助 `reuse_module_type_ratio`；
- module alias/scaling 配置及资源冲突统计；
- block/net/pin/pingroup/module 数量；
- PinGroup-Net 分量数；
- 资源冲突边数、波次数、最大可并行批次数；
- 几何和 schema 校验结果。

### 6.6 Benchmark 验收标准

- 同一生成配置和 seed 得到字节一致的输入文件；
- 所有 Pin 的 parent instance 存在；
- 所有 polygon 合法、非退化，方向和坐标可被当前 `PlaceDB` 解析；
- 所有 net 至少包含两个有效端点，或明确标记 terminal 规则；
- 每个 homology group 的 module/port 规则一致；
- 不存在未记录的数据修复或隐式缩放；
- 每个生成 case 的非 root 模块实例数不少于 200，且实际 PinGroup 数等于目标矩阵值；
- `reuse_instance_ratio` 按统一公式计算并满足 60%/95% 目标；辅助 module type 比例单独报告且未混用；
- 六档目标矩阵都能被批次规划器完整读取后，才能宣称目标 benchmark 建设完成；
- 生成集不包含训练/调参和最终报告之间的 sample 泄漏；
- 报告中明确声明该数据为“FloorSet-derived synthetic PinAssign benchmark”，不冒充 Intel 原始 PinAssign 数据。

## 7. FloorSet Benchmark 构建提示词

建议模型组合：设计与评审使用 `GPT-5.6 Sol/high`；机械实现、转换测试与批量统计使用 `GPT-5.6 Terra/medium`。

```text
请为当前 MCTS PinAssign 项目分阶段建设可复现的 Intel FloorSet-derived synthetic PinAssign benchmark。当前调用只执行 Day 1；完成 Day 1 产物并通过评审后停止，不得提前实现转换器、下载数据集或生成压力 case。

背景与硬约束：
1. 阅读当前仓库的 PlaceDB.py、homology.py、segment.py、plan_mcts_batches.py、assignment_solver.py 和现有测试；
2. 查阅 IntelLabs/FloorSet 官方仓库、官方 loader、数据格式、许可证和论文，只依据官方资料建立数据契约；
3. FloorSet 原始单样本通常只有 21–120 blocks，不适配实际测试规模。FloorSet 仅作为基础拓扑、几何和约束种子；后续产物必须称为 FloorSet-derived synthetic PinAssign benchmark；
4. 实际目标是非 root 模块实例数 >=200，PinGroup 数分别为 20,000、30,000、200,000，每档分别包含 60% 和 95% 两种复用模块实例比例，共六个 case；
5. 60%/95% 的主口径为 reused module instances / all non-root module instances：某实例的 module_name 在全部非 root 实例中出现至少 2 次时，该实例计为 reused；
6. 辅助口径 reuse_module_type_ratio 必须单列，不得代替主口径或与 case 标签混用；
7. Day 1 不下载 FloorSet 大数据，不生成小样本或目标规模 case，不编写批量生成实现，不运行性能 benchmark，也不虚构实际 tensor 检查结果。

Day 1 只完成以下内容：
A. 官方 FloorSet 数据契约：按来源列出 Lite/Prime 的已确认字段、shape/编码、padding/有效值规则、MIB/约束语义和待验证项；事实、推断、假设必须分栏；
B. 当前项目输入契约：从代码和现有测试提取 block.json、pingroup.json、PlaceDB、homology、segment 与批次规划器的必填字段、约束和失败条件；
C. 差距分析：逐项比较 FloorSet 与当前输入，标出“可直接映射”“需确定性派生”“需 synthetic scaling”“无法保留原语义”，特别说明 21–120 blocks 到 200+ 模块实例以及 20K/30K/200K PinGroups 的规模鸿沟；
D. 目标矩阵：固定 PG20K-R60、PG20K-R95、PG30K-R60、PG30K-R95、PG200K-R60、PG200K-R95，写明精确计数口径、允许的整数取整误差和每个 manifest 必须记录的实际值；
E. 许可与 provenance：记录代码和数据许可证、attribution 要求、FloorSet commit/version、源 sample ID、转换器版本、seed，以及 synthetic 派生链；禁止将外部大文件提交进 Git；
F. 接口设计：只设计、不实现 source adapter、single-sample converter、scaler/composer、reuse allocator、validator、manifest/provenance writer 的输入输出边界；要求转换器与校验器分离、稳定排序、稳定 seed，外部路径不得硬编码；
G. 后续验证清单：列出 Day 2 必须用小型公开样本确认的 tensor 细节，以及 Day 3/4 缩放器必须证明的规模、几何、连接、homology、确定性和资源冲突统计性质；
H. 成本模型框架：定义用于预测 20K/30K/200K 生成时间、磁盘、峰值内存和加载/规划成本的测量方法，但 Day 1 不填造实际测量值。

Day 1 输出：
- docs/floorset_benchmark_day1_contract.md；
- 文档内包含官方来源链接、字段映射/差距表、六档目标矩阵、复用率公式、许可与 provenance 表、建议接口、风险与待验证问题；
- 给出 Day 1 验收清单和 Day 2 的 go/no-go 条件。

Day 1 验收条件：
1. 所有 FloorSet 事实都有官方来源，无法确认的内容显式标为待验证；
2. 六档矩阵与复用率口径无歧义，module instance 与 module type 未混用；
3. 明确声明 FloorSet 原样本不满足目标规模，后续必须通过 synthetic scaling 适配；
4. 未下载多 GB 数据，未生成任何 20K/30K/200K case，未声称已有性能或容量结果；
5. 接口设计足以让 Day 2 在不改契约的前提下开始小样本转换。

后续阶段边界（本次不执行）：Day 2 开始小样本转换；Day 3/4 设计并验证缩放器；Day 5 生成并验证 20K 两档；Day 6 生成并验证 30K 两档，同时先做 200K 容量/成本预测，仅在前两档稳定且资源允许后试生成 200K；Day 7 冻结 benchmark v0 并给出 go/no-go。
```

## 8. 下一周工作目标：FloorSet Benchmark 建设

### 周目标

建立可复现、可验证、规模与复用率口径明确的 FloorSet-derived synthetic PinAssign benchmark v0。FloorSet 只作为基础拓扑和几何种子；本周重点是建立契约、验证缩放方法并优先落地 20K/30K 四档。200K 两档先完成容量与成本预测，只有 20K/30K 稳定且资源允许时才试生成，不把未验证的 200K 列为本周必达产物。

### 每日安排

#### Day 1：数据格式与适配契约

- 基于官方仓库、loader、论文和许可文本建立 FloorSet 数据契约，不下载大数据；
- 从当前代码和测试提取 `block.json`、`pingroup.json`、PlaceDB、homology、segment 与批次规划器输入契约；
- 完成字段级 gap 分析，明确原始 21–120 blocks 不满足 200+ 模块实例及 20K/30K/200K PinGroups；
- 固定六档目标矩阵及 `reuse_instance_ratio` 主口径，单列 `reuse_module_type_ratio` 辅助口径；
- 定义许可证、attribution、版本、sample ID、seed 和 synthetic 派生链的 provenance 要求；
- 设计 adapter、converter、scaler、reuse allocator、validator 和 manifest writer 的接口，不实现生成逻辑。

验收：形成 `docs/floorset_benchmark_day1_contract.md`，包含官方契约、当前输入契约、差距表、六档矩阵、复用率公式、provenance 方案、接口设计和待验证清单；未下载大数据、未转换样本、未生成目标 case。

#### Day 2：单样本转换器 v0

- 获取经许可的一个或少量小型公开 validation/test 样本，只下载完成验证所需的最小子集；若下载、依赖或字段编码受阻，记录证据并停止，不用合成数据替代；
- 先完成真实 tensor 检查和 single-sample converter/validator 接口冻结；不做 20K/30K/200K 缩放，不修改 MCTS；
- 若字段契约完整，再生成单样本 `block.json`、`pingroup.json`、manifest、provenance 和 validation 报告；
- 优先支持 Lite 矩形，稳定命名、排序和 seed 必须写入接口契约；
- 用当前 `PlaceDB` 进行单样本 schema/加载验证。

验收：至少一个真实官方 Lite 样本有可复核 tensor report；单样本输出（若可转换）可被当前项目读取，且无目标规模生成或性能结论。

#### Day 3：缩放器设计与校验基础

- 设计从 FloorSet 种子进行确定性组合、复制、扰动和连接扩展的缩放规则；
- 实现空间 offset、namespace、module alias/reuse allocation 与 synthetic homology 规则；
- 增加 schema、polygon、parent instance、net endpoint、homology 和 width 校验；
- 增加同 seed 字节一致性测试；
- 记录所有过滤、修复和缩放行为。

验收：小规模缩放原型可复现；冒烟集全通过，故意损坏的数据能被校验器拒绝；每个派生对象可追溯到源 sample。

#### Day 4：缩放器验证与目标口径校准

- 在不直接生成目标大规模 case 的前提下，用中间规模验证 PinGroup 数控制和 60%/95% 复用实例分配；
- 同时计算并单列 `reuse_instance_ratio` 与 `reuse_module_type_ratio`，验证二者不会混用；
- 生成 PinGroup-Net 图和 segment 资源冲突统计；
- 测量单位 PinGroup/模块实例的生成时间、磁盘和峰值内存，为 20K/30K/200K 建立成本预测；
- 验证当前 `PlaceDB` 和批次规划器可完整读取缩放原型。

验收：缩放器在固定 seed 下字节一致，目标计数误差规则通过，复用率主指标偏差不超过 1 个实例，并形成三档规模成本预测。实际 Day 4 结果见 `docs/floorset_benchmark_day4_report.md`；已完成 1K/3K × R60/R95 calibration smoke，正式 20K/30K/200K 仍未生成。

Day 4 复核发现后续正式 case 不能继续依赖增加 module 数补齐 PinGroup；固定 240-module 的层级 composer、独立 B2B net synthesizer、宽度分布和容量 packing 修订方案见 `docs/hierarchical_synthetic_net_capacity_plan_and_prompts.md`。该修订先替代正式矩阵生成方式，不改变 Day 4 已完成 smoke 的证据性质。

#### Day 5：20K 两档生成与验证

- 生成 `PG20K-R60` 和 `PG20K-R95`；
- 确保每个 case 非 root 模块实例数 >=200、PinGroup 数恰为 20,000；
- 输出主/辅助复用率、源 sample IDs、seed、磁盘、生成时间、加载时间、规划时间、峰值内存和资源冲突统计；
- 使用当前 `PlaceDB`、校验器和批次规划器完整读取。

验收：20K 两档均满足 schema、几何、连接、homology、确定性、provenance 和精确计数要求，且复用率符合统一口径。

#### Day 6：30K 两档与 200K 条件评估

- 生成并验证 `PG30K-R60` 和 `PG30K-R95`，指标与 Day 5 相同；
- 用 Day 4–6 实测数据更新 `PG200K-R60/R95` 的磁盘、生成时间、峰值内存、加载和批次规划成本预测；
- 仅当 20K/30K 四档全部稳定、预测资源在预算内且不会影响 v0 冻结时，才试生成 200K；
- 若不满足条件，只保留 200K 的生成配置、容量预测、风险和后续执行条件，不把未生成视为本周失败；
- 对已经稳定的最小 case 做一次低预算串行可运行性检查，性能结论留待 benchmark 冻结后。

验收：30K 两档完整通过，200K 两档有基于实测单位成本的预测和明确 go/no-go；若试生成，则必须接受同一套校验，不得以抽样校验替代。

#### Day 7：冻结 benchmark v0 与评审

- 固定 converter version、样本列表和 manifest schema；
- 汇总已知偏差和 FloorSet 原始语义与 synthetic PinAssign 语义的差异；
- 输出 benchmark 使用说明和可复现命令；
- 冻结已通过的 20K/30K case、200K 成本预测与条件化生成配置；
- 分别判断是否可进入路线 A 的 20K/30K 实验，以及是否批准生成/使用 200K 两档。

验收：形成 benchmark v0、测试报告、数据清单和两项 go/no-go 结论；未通过的 case 不得纳入冻结集。

## 9. 下一周退出标准

下一周结束时必须满足：

- Day 1 数据契约、项目输入契约、gap 分析、六档矩阵、复用率定义、provenance 和接口设计完成；
- 至少三个 FloorSet-Lite 冒烟 case 转换成功；
- 转换器输出确定且有 provenance；
- 缩放器通过确定性、精确 PinGroup 计数、复用率、schema、几何、连接和 homology 校验；
- `PG20K-R60/R95` 与 `PG30K-R60/R95` 实际生成并验证；若任何一档失败，v0 必须降级标注且不能进入正式并行性能结论；
- `PG200K-R60/R95` 至少完成基于实测数据的容量/成本预测和明确生成条件；只有 20K/30K 稳定且资源允许时才要求试生成；
- 当前 `PlaceDB` 和批次规划器可以读取所有纳入 v0 的生成数据；
- 主复用率和辅助 module type 复用比例分别报告，所有 60%/95% 标签只使用主指标；
- 至少对一个稳定 case 完成低预算串行可运行性检查，但本周不要求并行速度结论；
- 不把 FloorSet-derived benchmark 的质量结果解释成真实工业数据结论。

如果 FloorSet 到当前 PinAssign 语义的映射无法通过校验，则停止扩大数据规模，保留 FloorSet 作为基础拓扑与几何种子，并补充独立、可追溯的 PinGroup/homology/scaling 合成层。任何由该层生成的数据都继续使用 “FloorSet-derived synthetic PinAssign benchmark” 名称。
