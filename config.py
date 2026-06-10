"""程序运行参数的集中配置。

无命令行参数时，直接修改 ``DEFAULT_CONFIG`` 的字段即可运行程序。
命令行参数仅用于临时覆盖这些默认配置。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class RunConfig:
    """保存单次 Stage1 分配与最终指标评估需要的全部可调参数。"""

    # ===== 输入输出路径 =====
    # block JSON 输入文件路径。
    block_json_path: Path = Path(
        r"C:\Users\DELL\Desktop\test_case_from_huawei\block_case2.json"
    )
    # pingroup JSON 输入文件路径。
    pingroup_json_path: Path = Path(
        r"C:\Users\DELL\Desktop\test_case_from_huawei\pingroup_case2.json"
    )
    # Stage1 完整分配结果 JSON 输出路径。
    assignment_output_path: Path = PROJECT_DIR / "outputs" / "case2_assignment.json"
    # 单次运行参数、summary、net 指标报告根目录。
    results_root: Path = PROJECT_DIR / "run_results"
    # 适配接口格式的 final_result 输出目录。
    interface_result_dir: Path = PROJECT_DIR / "mcts_result" / "final_result"

    # ===== Feedthrough 设置 =====
    # 旧版 feedthrough 程序路径；当 predict/evaluate 路径未单独设置时作为兼容 fallback。
    feedthrough_source_dir: Path = PROJECT_DIR / "feedthrough"
    # MCTS 搜索 reward 使用的 feedthrough 预测器路径，可为可执行文件或包含 build/ftpred 的目录。
    feedthrough_predict_source_dir: Path = PROJECT_DIR / "feedthrough_predict"
    # Stage1 最终指标统计使用的 feedthrough 评估器路径，可为可执行文件或包含 build/ftpred 的目录。
    feedthrough_evaluate_source_dir: Path = PROJECT_DIR / "feedthrough_evaluate"
    # MCTS reward 阶段使用的 feedthrough 来源，可选 "predict" 或 "evaluate"。
    feedthrough_reward_source: str = "evaluate"
    # 最终指标和非零 feedthrough reward 是否启用 feedthrough 预测/评估。
    enable_feedthrough: bool = True
    # 找不到 ftpred 可执行文件时是否自动调用 CMake 编译；默认关闭，要求预先编译好。
    auto_build_feedthrough: bool = False
    # Windows 中 CMake 使用的 generator；Linux/外部入口可传 None。
    cmake_generator: str = "MinGW Makefiles"

    # ===== 通用 MCTS 设置 =====
    # MCTS 输入基准模拟次数；不同模式会基于它计算实际预算。
    simulations: int = 4096
    # 随机种子，用于保证 MCTS 随机补全可复现。
    random_seed: int = 7
    # 容量不足时是否允许最终 fallback 强制 overflow 分配。
    allow_overflow_fallback: bool = True
    # MCTS 搜索模式，可选 "basic"、"layered" 或 "hybrid"。
    mcts_search_mode: str = "hybrid"
    # 是否保留每棵 MCTS 树的搜索画像和路由信息，便于后续扩展到报告输出。
    mcts_enable_search_diagnostics: bool = False

    # ===== Basic 模式设置 =====
    # Basic 模式是否根据总搜索空间动态调整模拟次数。
    mcts_basic_dynamic_simulations: bool = True
    # Basic 模式总搜索空间放大/缩小因子的归一化除数。
    mcts_basic_space_scale_divisor: float = 100_000.0
    # Basic 模式动态模拟次数放大因子的上限。
    mcts_basic_max_space_factor: float = 8.0
    # Basic 模式动态模拟次数下限。
    mcts_basic_min_simulations: int = 1024
    # Basic/basic-like 模式中深度为 1 的局部树使用的固定模拟次数；该类树默认不做候选剪枝。
    mcts_basic_depth1_simulations: int = 32
    # Basic/basic-like 模式中深度为 2 的局部树使用的固定模拟次数。
    mcts_basic_depth2_simulations: int = 512
    # Basic/basic-like 模式中深度小于等于该值时关闭候选剪枝，避免单层选择误剪最优候选。
    mcts_basic_disable_pruning_depth_limit: int = 2

    # ===== Layered 模式设置 =====
    # Layered 模式逐层预算衰减系数。
    mcts_budget_decay: float = 0.6
    # Layered 模式进入长尾深度后的预算衰减系数。
    mcts_tail_decay: float = 0.9
    # Layered 模式估算搜索空间时使用的典型深度。
    mcts_typical_depth: int = 6
    # Layered 模式搜索空间放大因子的归一化除数。
    mcts_space_scale_divisor: float = 1_000_000.0
    # Layered 模式搜索空间放大因子的上限。
    mcts_max_space_factor: float = 10.0
    # Layered 模式每层最小模拟次数。
    mcts_min_layer_simulations: int = 256
    # Layered 模式切换到长尾衰减和早停判断的深度阈值。
    mcts_tail_depth: int = 8
    # Layered 模式早停判断中领先幅度相对标准差的倍数。
    mcts_early_stop_std_multiplier: float = 2.0
    # 是否启用 Layered 模式长尾早停。
    mcts_enable_tail_early_stop: bool = True

    # ===== Hybrid 模式设置 =====
    # Hybrid 中走 basic-like 快速路径的最大树深度。
    mcts_hybrid_basic_depth_limit: int = 4
    # Hybrid 中走 basic-like 快速路径的最大 log 搜索空间，默认 log(1e6)。
    mcts_hybrid_basic_log_space_limit: float = math.log(50_000.0)
    # Hybrid 普通深树每层保留的 beam 路径数。
    mcts_hybrid_beam_width: int = 3
    # Hybrid 长尾树每层保留的 beam 路径数，越大越保守但耗时越高。
    mcts_hybrid_tail_beam_width: int = 4
    # Hybrid 进入长尾 profile 的深度阈值。
    mcts_hybrid_tail_depth: int = 24
    # Hybrid 普通深度下每层预算衰减系数。
    mcts_hybrid_budget_decay: float = 0.65
    # Hybrid 长尾深度下每层预算衰减系数，越大越能穿透长尾。
    mcts_hybrid_tail_budget_decay: float = 0.92
    # Hybrid 每层最小模拟次数。
    mcts_hybrid_min_layer_simulations: int = 96
    # Hybrid 每层最大模拟次数上限，控制单层耗时。
    mcts_hybrid_max_layer_simulations: int = 1024
    # Hybrid 每棵局部树最大总模拟次数上限，控制长尾总耗时。
    mcts_hybrid_max_tree_simulations: int = 30_000
    # Hybrid 是否启用每层早停。
    mcts_hybrid_enable_layer_early_stop: bool = True
    # Hybrid 早停判断中领先幅度相对标准差的倍数。
    mcts_hybrid_early_stop_std_multiplier: float = 2.0
    # Hybrid 单棵树 wall-clock 时间上限；0 表示关闭，仅作为后续接口预留。
    mcts_hybrid_time_limit_seconds: float = 0.0
    # ===== Hybrid 超深树设置 =====
    # 是否启用 Hybrid 超深树 profile；默认关闭以保持 ft_reward_source_select 版本默认行为。
    mcts_hybrid_enable_ultradeep_profile: bool = False
    # Hybrid 超深树阈值；超过该深度后以“前缀搜索 + 快速补全”为主，避免 100+ 到 2000 深度逐层耗时失控。
    mcts_hybrid_ultradeep_depth: int = 100
    # Hybrid 超深树最多展开搜索的前缀深度，剩余同构组走快速补全。
    mcts_hybrid_max_expanded_depth: int = 64
    # Hybrid 超深树每层保留的 beam 路径数，通常取 1-2 控制运行时间。
    mcts_hybrid_ultradeep_beam_width: int = 1
    # Hybrid 超深树每层最小模拟次数，降低长尾树的固定层成本。
    mcts_hybrid_ultradeep_min_layer_simulations: int = 32
    # Hybrid 超深树每层最大模拟次数上限。
    mcts_hybrid_ultradeep_max_layer_simulations: int = 128
    # Hybrid 超深树搜索前缀结束后是否使用 HPWL/容量启发式快速补全，避免后段大量 FT reward 评估。
    mcts_hybrid_use_fast_completion_for_ultradeep: bool = True

    # ===== 同构组提交设置 =====
    # 是否在同构组排序时把多扇出 Pin 按 successors 连接数视为更高复用次数；关闭时只按唯一物理 Pin 数排序。
    homology_use_fanout_reuse_for_sorting: bool = True
    # 当前 pins_in 覆盖同构组 Pin 比例达到该阈值时，允许直接提交整组到同一 segment；1 表示仅完整覆盖才提交。
    homology_group_commit_coverage_threshold: float = 1.0
    # 可提交同构组数占当前 MCTS 搜索同构组数比例不超过该值时，跳过当前 MCTS 树。
    mcts_tree_min_committable_group_ratio: float = 0.3

    # ===== 候选剪枝设置 =====
    # 是否启用候选 segment 预剪枝，减少大分支树搜索空间。
    mcts_enable_candidate_pruning: bool = True
    # 普通树候选剪枝保留的 top-K 数量。
    mcts_candidate_top_k: int = 12
    # 长尾/超大树候选剪枝保留的 top-K 数量。
    mcts_candidate_tail_top_k: int = 8
    # 候选数量小于该阈值时不剪枝。
    mcts_candidate_min_count: int = 16
    # 与最佳候选分数差距在该比例内的候选额外保留，避免过度剪枝。
    mcts_candidate_score_tolerance: float = 0.03

    # ===== Reward 设置 =====
    # reward 中归一化 HPWL improvement 的权重。
    wirelength_reward_weight: float = 0.9
    # reward 中归一化 feedthrough improvement 的权重；0 表示 MCTS 不计算 FT reward。
    feedthrough_weight: float = 0.1
    # reward 归一化分母下限，避免参考值为 0 或过小。
    reward_normalization_floor: float = 1.0
    # 最终 reward 整体放大系数，用于调整 UCB exploitation 量级。
    reward_scale: float = 100.0

    # ===== Segment 和导出设置 =====
    # 是否按边长百分位对 block 边进行 segment 细分。
    enable_segment_subdivision: bool = True
    # segment 细分使用的边长百分位阈值，例如 50 表示 L50。
    segment_length_percentile: int = 50
    # 是否导出接口需要的 final_result 格式。
    export_interface_result: bool = True

    def to_record(self) -> Dict[str, Any]:
        """转换为可保存到 JSON 中的参数记录。"""
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value)
        return values


DEFAULT_CONFIG = RunConfig()
