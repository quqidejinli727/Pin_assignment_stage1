# FloorSet Benchmark 建设 Day 2：单样本真实验证与转换计划

## 1. 范围和当前证据

Day 2 只处理一个或少量 FloorSet-Lite 官方公开 validation/test 样本：确认真实 tensor、完成单样本转换接口设计评审，并在依赖和数据可用后生成单样本的五类产物。Day 2 不做 20K/30K/200K 缩放，不实现并行 MCTS，不修改 MCTS 或 segment 算法。

Day 1 已按计划不下载 FloorSet 数据、不生成目标 case。对当前工作区的只读检索结果如下：

| 检查项 | 结果 | 可引用证据 |
| --- | --- | --- |
| Day 1 契约 | 存在 | `docs/floorset_benchmark_day1_contract.md` |
| 主路线/官方资料汇总 | 存在 | `docs/parallel_mcts_routes_prompts_and_floorset_benchmark.md` |
| Day 1 日志/样本检查报告 | 未发现 | 无结果可引用 |
| FloorSet tensor 或数据目录 | 未发现 | 无结果可引用 |
| `block.json`/`pingroup.json` 生成输出 | 未发现 | 无结果可引用 |
| `manifest.json`/`provenance.json`/`validation.json` | 未发现 | 无结果可引用 |

因此 Day 2 的真实 tensor 数值、样本 ID、文件 hash 和转换通过率目前均为“未测量”，不能从 Day 1 文档中的字段描述推断结果。

## 2. 官方获取边界

只允许从 [IntelLabs/FloorSet 官方仓库](https://github.com/IntelLabs/FloorSet) 及其官方 loader/数据发布获取一个最小 Lite validation/test 子集。优先使用官方提供的静态 Lite test/validation 样本；若官方分发机制只能按归档下载，则必须先检查 Content-Length 和预计展开大小，选择可接受的最小公开子集，并在获取前记录 URL、版本、大小和许可。

禁止下载完整训练集、完整 Lite/Prime 归档或数 GB 数据；禁止安装新依赖；禁止将外部数据复制进 Git。官方 loader 的自动下载行为必须关闭或改为显式人工确认，避免误触发大文件下载。官方代码仓库为 Apache-2.0，数据集为 CC BY 4.0；须保留 attribution、来源 URL、commit/tag、sample ID 和论文 [arXiv:2405.05480](https://arxiv.org/abs/2405.05480)。

若网络、权限、数据链接或现有 Python/Torch 环境阻塞获取，Day 2 只记录错误证据和阻塞项，保留接口设计，不用合成数据冒充官方样本检查。

## 3. Day 2 目标与非目标

目标：

- 获取至少一个真实 Lite 样本并保存原始字段检查报告；
- 确认有效 block/pin/edge 数、tensor shape、dtype、padding 和索引编码；
- 固定一个单样本 converter 的输入输出边界；
- 生成或设计可验证的 `block.json`、`pingroup.json`、`manifest.json`、`provenance.json`、`validation.json`；
- 用当前 `PlaceDB` 和基础 schema 校验单样本输出；
- 记录所有仍需 Day 3+ synthetic scaling 的差距。

非目标：

- 不生成 20K/30K/200K PinGroup case；
- 不实现跨样本 composer、reuse allocator 或大规模 scaling；
- 不运行 MCTS、并行实验或性能 benchmark；
- 不把一个 FloorSet 样本解释为满足 200+ module instance 或 60%/95% reuse 目标；
- 不虚构 tensor、转换结果、磁盘或时间数据。

## 4. 前置 Go/No-Go

只有满足以下条件才进入真实样本读取：

1. 能确认数据来源为官方仓库/官方数据发布，并记录许可证；
2. 预计下载量和展开量低于本机临时预算，且不是完整训练集；
3. 已确认不会触发 `lite_dataset.py` 的自动大文件下载；
4. 当前环境已经具备读取所需依赖，Day 2 不安装依赖；
5. 输出目录位于工作区外部数据目录或被 Git 忽略的位置。

任一条件不满足则 No-Go：停止获取，输出“无真实样本结果”，只完成字段检查表、接口计划和阻塞记录。

## 5. 真实 tensor 检查表

对每个实际 sample ID 记录原始 shape、dtype、有效元素数、padding 数、最小/最大值和 hash；不能只记录 loader 注释。检查至少包括：

| 字段 | 必查内容 | 通过条件 |
| --- | --- | --- |
| `area_target` | shape、dtype、有效 block 数、`-1` padding | 有效值与 block 数一致且非负 |
| `b2b_connectivity` | edge 维度、列含义、索引范围、padding | 有效端点可映射到 block，padding 不泄漏 |
| `p2b_connectivity` | pin/block 索引编码、权重列、padding | 能与 `pins_pos` 对齐；不能凭形状猜语义 |
| `pins_pos` | `n_pins x 2`、坐标 dtype、padding | 有效 pin 坐标可区分于 padding |
| `placement_constraints` | `n_blocks x 5` 五列编码 | 列含义和有效范围与官方资料一致 |
| `fp_sol` | Lite 的 `[w,h,x,y]`、负值/填充 | 有效 block 数一致，w/h 为正 |
| `tree_sol` | 维度和 padding | 仅记录，不作为当前 JSON 输入 |
| `metrics_sol` | 八项统计向量及与实际计数的关系 | 只做交叉检查，差异须记录 |

报告必须将“官方已确认”“由实际样本确认”“尚未确认”分栏；任何 p2b 编码、weight 列或约束细节若无法确认，转换必须暂停在该字段，不得猜测。

## 6. Single-sample converter 计划（只设计，不实现）

建议后续文件边界：

- `floorset_adapter.py`：读取一个官方样本，输出规范化内部记录和 tensor inspection report；不负责当前 JSON。
- `floorset_single_converter.py`：将规范化 Lite 记录转换为当前项目的 block/pingroup 结构；不做跨样本复制。
- `floorset_validator.py`：独立验证 schema、几何、连接、同构键和 determinism；不修改输入。
- `floorset_manifest.py`：写 manifest/provenance/validation 元数据，所有数组和键稳定排序。

建议接口：

```text
inspect_sample(source_root, sample_id) -> RawLiteRecord, TensorReport
convert_single(raw_record, config) -> BlockJson, PinGroupJson, ConversionReport
validate_case(block_json, pingroup_json, manifest) -> ValidationReport
write_case_artifacts(output_dir, artifacts) -> ArtifactIndex
```

`config` 至少包含 source commit、sample ID、base seed、width policy、terminal policy、synthetic_homology 标记和输出版本。外部路径必须命令行传入，禁止硬编码个人路径。Day 2 不实现这些接口，只冻结函数签名、错误分类和报告字段。

## 7. 单样本产物契约

```text
case_<source_id>/
  block.json
  pingroup.json
  manifest.json
  provenance.json
  validation.json
  tensor_report.json
```

最低字段要求：

- `block.json`：递归 root/module，实例 `name` 唯一，`module_name`、非退化 `vertex`、必要的 `children`；
- `pingroup.json`：顶层 net 数组，每个 Pin 的 `parent_inst` 可解析，`parent_module` 与 module type 一致，`pingroup_name`/`scope`/`successors`/`width` 有明确来源或合成声明；
- `manifest.json`：converter version、source sample ID、FloorSet commit、seed、width/terminal policy、实际 module/net/pin/PinGroup 计数、过滤/修复统计和目标矩阵声明为“未适用”；
- `provenance.json`：官方 URL、许可证、论文、下载时间、源文件 hash、派生关系和 attribution；
- `validation.json`：schema、几何、parent instance、net endpoint、homology、determinism 的逐项状态；
- `tensor_report.json`：第 5 节所有真实 shape/dtype/padding/索引观察。

单样本输出不得写成 20K/30K/200K case，也不得填写虚假的 60%/95% reuse 指标；相关字段应为 `not_applicable` 或明确缺失状态。

## 8. 测试矩阵

| 测试 | 输入 | Day 2 期望 |
| --- | --- | --- |
| tensor inspection | 一个官方 Lite sample | 报告真实 shape、padding、dtype、索引范围 |
| block schema | 单样本转换结果 | root、实例名、vertex 合法 |
| pin reference | `pingroup.json` + block | 所有 `parent_inst` 可解析 |
| net validity | 转换后的 nets | 无 padding 端点；不足两端的 edge 被拒绝或记录 terminal policy |
| geometry | block polygons | w/h 正、非退化、坐标可解析 |
| homology | `parent_module.pingroup_name` | 键稳定；synthetic homology 明确标记 |
| determinism | 同一 sample/config/seed 两次 | JSON/report 字节一致；若未实现则记录为 pending |
| negative validation | 删除 module、注入 padding/重复 pin | validator 拒绝并给出稳定错误码 |
| PlaceDB load | 输出两个 JSON | 仅在 schema 和 terminal 表示确认后执行 |

## 9. 确定性与许可要求

采用 `sha256(base_seed|sample_id|converter_version)` 派生整数 seed；禁止 Python `hash()`。数组、Pin、net、module 和 JSON key 均稳定排序。原始样本不复制到仓库；只保存必要的 hash、字段摘要、来源和许可信息。任何派生的 `module_name`、`pingroup_name`、width scaling 或 terminal 表示必须在 manifest 中显式记录，并声明为 synthetic/derived。

## 10. 分步执行顺序

1. 只读确认工作区和输出目录，创建本地忽略的数据暂存位置；不下载。
2. 记录官方 URL、commit、许可证、目标 sample ID 和预计大小。
3. 做大小/路径/依赖 Go 检查；No-Go 时记录错误并停止。
4. 获取一个最小 Lite sample；禁止自动扩展到训练集。
5. 运行 tensor inspection，仅保存 `tensor_report.json` 和摘要日志。
6. 对照第 5 节逐项确认字段；未确认项建立阻塞记录。
7. 评审单样本 converter 接口和 terminal/width/homology policy；不实现 scaling。
8. 若字段契约完整，生成单样本五类 JSON/报告；否则只保留 inspection 和设计产物。
9. 用 validator/PlaceDB 做小范围读取检查；不运行 MCTS。
10. 汇总磁盘、读取、转换、校验时间和峰值内存，形成 Day 2 报告。

## 11. 验收与停止条件

Day 2 通过条件：至少一个真实官方 Lite sample 有可复核 tensor report；字段 padding/索引语义已确认或逐项标为阻塞；单样本输出（若转换可行）具备五类产物、provenance、许可证和稳定排序；PlaceDB/schema 校验结果有证据；没有大数据下载或目标规模生成。

立即停止条件：下载大小超预算、触发完整数据自动下载、来源/许可证无法确认、p2b/terminal 编码无法确认、输出无法通过 parent instance 或几何校验、或需要安装新依赖。停止时不得用人工合成数据填补官方样本结果。

## 12. 成本记录模板

Day 2 只记录实际测量，不预填数值：

```text
source_sample_id: <id or unavailable>
source_url_and_commit: <url/commit>
archive_bytes: <measured or not_downloaded>
expanded_bytes: <measured or not_downloaded>
tensor_file_bytes: <measured>
download_seconds: <measured or blocked>
inspection_seconds: <measured>
conversion_seconds: <measured or not_run>
validation_seconds: <measured or not_run>
peak_rss_mb: <measured or not_run>
dependency_status: existing | blocked | not_checked
go_no_go: go | no-go
evidence: <log/report path or blocker>
```

The 20K/30K/200K storage, memory and planning costs remain future scaling measurements. Day 2 may define unit-cost fields, but must not extrapolate without measured single-sample data.

## 13. 与主规划的一致性检查

主规划当前已将 FloorSet 原始 21–120 blocks 与目标规模 gap、单样本 Day 2、后续 scaling 的边界写入；本计划进一步收紧 Day 2 为“真实小样本检查 + 单样本转换设计/验证”，不改变主规划的目标矩阵或并行 MCTS 内容。

