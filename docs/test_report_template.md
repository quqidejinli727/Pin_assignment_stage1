# PinAssignFlow 测试报告（模板初稿）

> 说明：本文档为验收测试报告初稿。`[待填写]` 内容需在华为指定测试环境中根据实际运行命令、日志、评估结果补齐。当前模板不展开过细的中间测试数据，仅保留验收所需的环境、基础参数、case 级参数配置和达标结论。

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 项目名称 | 多层次及复用模块的 Pin 自动寻优项目 |
| 测试版本 | `[待填写：代码提交号/交付包版本]` |
| 测试日期 | `[待填写]` |
| 测试人员 | `[待填写]` |
| 测试环境 | `[待填写：华为指定测试环境名称/机器编号]` |
| 测试数据集 | 华为指定数据集 `case1`、`case2`、`case3` |

## 2. 验收标准

本次测试依据以下验收要求进行：

1. 源代码需能够在华为指定数据集上运行，且包含详细的模型参数设置、性能评估等功能。
2. 在 `case1`、`case2`、`case3` 上，面向 200+ 模块规模（harden/partition）、复用模块数量 60%+、总 pin group 数量 5W+、无初始 pin 排布信息的场景，算法支持在线长与 feedthrough 之间灵活权衡。
3. 在 4 小时运行时间内，结果 100% 满足合法性约束和同构一致性。
4. 通过不同配置达成以下优化效果：
   - 线长优化目标：在保持 feedthrough 指标不劣化的条件下，整体曼哈顿线长较华为自研算法降低至少 4%。
   - feedthrough 优化目标：在保持线长指标不劣化的条件下，feedthrough 较华为自研算法减少至少 2%。

## 3. 被测代码与流程

全流程入口：

```text
workspace_import/mcts-with-PPO-PinAssign/run.py
```

全流程阶段如下：

| 阶段 | 模块 | 主要输出 | 说明 |
| --- | --- | --- | --- |
| Stage 1 | `stage1_mcts` | `segment_assignments.json` | MCTS Segment 分配，处理同构约束、容量约束、线长/feedthrough reward。 |
| Stage 2 | `stage2_nlplace` | `result.json` | 基于 Stage1 segment 结果进行连续坐标优化。 |
| Stage 3 | `stage3_legalization` | `result_legalized.json` | QP/规范图合法化，消除重叠并保持同构一致性。 |
| Evaluate | `[待填写：华为指定 evaluate 工具]` | HPWL、feedthrough | 对最终结果进行性能评估。 |

## 4. 测试环境

| 项目 | 内容 |
| --- | --- |
| 操作系统 | `[待填写]` |
| Python 版本 | `[待填写]` |
| CPU / 内存 | `[待填写]` |
| GPU / CUDA | `[待填写；如未使用填写 N/A]` |
| CMake / C++ 编译器 | `[待填写：用于 feedthrough predictor 编译或说明已预编译]` |
| 项目部署目录 | `[待填写]` |
| 数据集目录 | `[待填写]` |
| 输出目录 | `[待填写]` |
| 关键环境变量 | `[待填写：如 CVXPY_SOLVER、OMP_NUM_THREADS、CUDA_VISIBLE_DEVICES 等]` |

环境准备命令：

```bash
cd [待填写：项目根目录]
[待填写：激活 Python/conda/venv 环境命令]
[待填写：依赖安装命令]
[待填写：feedthrough predictor 编译或确认命令]
```

## 5. 基础参数配置

以下为本轮测试的基础参数，除 case 级达标配置中特别说明外，均采用此处设置。

| 参数类别 | 参数 | 取值 |
| --- | --- | --- |
| Stage1 | `mcts_search_mode` | `[待填写，例如 hybrid]` |
| Stage1 | `simulations` / `--num-simulations` | `[待填写]` |
| Stage1 | `time_limit` / `--time-limit` | `[待填写]` |
| Stage1 | `enable_segment_subdivision` | `[待填写]` |
| Stage1 | `segment_length_percentile` | `[待填写]` |
| Stage1 | `mcts_enable_candidate_pruning` | `[待填写]` |
| Stage2 | `--nlplace-iterations` | `[待填写]` |
| Stage2 | `--nlplace-density-weight` | `[待填写]` |
| Stage2 | `--nlplace-params` | `[待填写：params.json 路径或默认]` |
| Stage3 | `--keepout` | `[待填写]` |
| Stage3 | `--hpwl-thresh` | `[待填写]` |
| Stage3 | `--max-outer-iter` | `[待填写]` |
| Evaluate | 评估程序路径 | `[待填写]` |

## 6. 运行命令模板

全流程运行命令：

```bash
python run.py \
  --case [待填写：case目录] \
  --output [待填写：输出目录] \
  --evaluate [待填写：evaluate.py路径，如无则删除该参数] \
  --num-simulations [待填写] \
  --time-limit [待填写] \
  --nlplace-iterations [待填写] \
  --nlplace-density-weight [待填写] \
  --keepout [待填写] \
  --hpwl-thresh [待填写] \
  --max-outer-iter [待填写] \
  [待填写：case 专用 stage1-* 参数]
```

跳过中间阶段复测命令（如需）：

```bash
# 使用已有 Stage1 结果继续跑 Stage2/Stage3
python run.py \
  --case [待填写] \
  --output [待填写] \
  --skip-mcts \
  --segment-assignments [待填写：segment_assignments.json]

# 使用已有 Stage2 结果只跑 Stage3
python run.py \
  --case [待填写] \
  --output [待填写] \
  --skip-mcts \
  --skip-nlplace \
  --result [待填写：result.json] \
  --segment-assignments [待填写：segment_assignments.json]
```

## 7. 各 case 达标配置

说明：三个 case 的达标参数设置允许不同，应分别记录。若某个验收方向当前未达标或本轮不测试，应在“本轮状态”中明确标注。

### 7.1 case1 参数配置

| 验收方向 | 关键参数设置 | 运行命令/输出目录 | 本轮状态 |
| --- | --- | --- | --- |
| 线长优化 | `[待填写：wirelength_weight、feedthrough_weight、Stage2/Stage3 关键差异参数]` | `[待填写]` | `[待填写：已测试/待测试/不适用]` |
| feedthrough 优化 | `[待填写：wirelength_weight、feedthrough_weight、Stage2/Stage3 关键差异参数]` | `[待填写]` | `[待填写：已测试/待测试/不适用]` |

### 7.2 case2 参数配置

| 验收方向 | 关键参数设置 | 运行命令/输出目录 | 本轮状态 |
| --- | --- | --- | --- |
| 线长优化 | `[待填写：wirelength_weight、feedthrough_weight、Stage2/Stage3 关键差异参数]` | `[待填写]` | `[待填写：已测试/待测试/不适用]` |
| feedthrough 优化 | `[本轮跳过：当前 case2 的 feedthrough 优化验收指标未合格，暂不纳入本轮验收结论]` | N/A | 跳过 |

### 7.3 case3 参数配置

| 验收方向 | 关键参数设置 | 运行命令/输出目录 | 本轮状态 |
| --- | --- | --- | --- |
| 线长优化 | `[待填写：wirelength_weight、feedthrough_weight、Stage2/Stage3 关键差异参数]` | `[待填写]` | `[待填写：已测试/待测试/不适用]` |
| feedthrough 优化 | `[待填写：wirelength_weight、feedthrough_weight、Stage2/Stage3 关键差异参数]` | `[待填写]` | `[待填写：已测试/待测试/不适用]` |

## 8. 验收结果汇总

| Case | 验收方向 | 运行时间是否 ≤ 4h | 合法性是否 100% | 同构一致性是否 100% | 优化指标 | 是否达标 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| case1 | 线长优化 | `[待填写]` | `[待填写]` | `[待填写]` | 线长下降 `[待填写]%`，FT 是否不劣化 `[待填写]` | `[待填写]` | `[待填写]` |
| case1 | feedthrough 优化 | `[待填写]` | `[待填写]` | `[待填写]` | FT 下降 `[待填写]%`，线长是否不劣化 `[待填写]` | `[待填写]` | `[待填写]` |
| case2 | 线长优化 | `[待填写]` | `[待填写]` | `[待填写]` | 线长下降 `[待填写]%`，FT 是否不劣化 `[待填写]` | `[待填写]` | `[待填写]` |
| case2 | feedthrough 优化 | N/A | N/A | N/A | N/A | 跳过 | 当前 case2 的 FT 优化验收指标未合格，本轮直接跳过。 |
| case3 | 线长优化 | `[待填写]` | `[待填写]` | `[待填写]` | 线长下降 `[待填写]%`，FT 是否不劣化 `[待填写]` | `[待填写]` | `[待填写]` |
| case3 | feedthrough 优化 | `[待填写]` | `[待填写]` | `[待填写]` | FT 下降 `[待填写]%`，线长是否不劣化 `[待填写]` | `[待填写]` | `[待填写]` |

## 9. 评估方法说明

性能指标由项目集成的 evaluate 工具输出，至少记录：

```text
Total HPWL / 曼哈顿线长：[待填写]
Total Feedthrough：[待填写]
```

下降率计算方式：

```text
线长下降率 = (华为自研算法线长 - 本算法线长) / 华为自研算法线长 * 100%
feedthrough下降率 = (华为自研算法feedthrough - 本算法feedthrough) / 华为自研算法feedthrough * 100%
```

合法性与同构一致性检查需记录检查脚本、命令和结论：

```text
合法性检查命令：[待填写]
合法性检查结论：[待填写：100%通过/未通过]

同构一致性检查命令：[待填写]
同构一致性检查结论：[待填写：100%通过/未通过]
```

## 10. 输出文件与日志

每次验收运行至少归档以下文件：

| Case | 验收方向 | 输出目录 | run log | 最终结果 | 评估日志 |
| --- | --- | --- | --- | --- | --- |
| case1 | 线长优化 | `[待填写]` | `[待填写]` | `[待填写：result_legalized.json]` | `[待填写]` |
| case1 | feedthrough 优化 | `[待填写]` | `[待填写]` | `[待填写：result_legalized.json]` | `[待填写]` |
| case2 | 线长优化 | `[待填写]` | `[待填写]` | `[待填写：result_legalized.json]` | `[待填写]` |
| case2 | feedthrough 优化 | N/A | N/A | N/A | N/A |
| case3 | 线长优化 | `[待填写]` | `[待填写]` | `[待填写：result_legalized.json]` | `[待填写]` |
| case3 | feedthrough 优化 | `[待填写]` | `[待填写]` | `[待填写：result_legalized.json]` | `[待填写]` |

## 11. 结论

本轮测试结论如下：

1. 源代码在华为指定数据集上的可运行性：`[待填写：通过/不通过]`
2. 参数配置与性能评估功能：`[待填写：通过/不通过]`
3. 4 小时运行时间约束：`[待填写：通过/不通过]`
4. 合法性约束和同构一致性：`[待填写：通过/不通过]`
5. 线长优化验收：`[待填写：通过/不通过，说明 case 范围]`
6. feedthrough 优化验收：`[待填写：通过/不通过，说明 case 范围；case2 本轮跳过]`

最终结论：

```text
[待填写：例如“本轮除 case2 feedthrough 优化项跳过外，其余已测试项满足/不满足验收要求”。]
```
