"""程序运行参数的集中配置。

无命令行参数时，直接修改 ``DEFAULT_CONFIG`` 的字段即可运行程序。
命令行参数仅用于临时覆盖这些默认配置。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class RunConfig:
    """保存单次 MCTS 分配与最终指标评估需要的全部可调参数。"""

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
    # 每次运行的参数、summary、net 指标报告根目录。
    results_root: Path = PROJECT_DIR / "run_results"
    # 适配接口格式的 final_result 输出目录。
    interface_result_dir: Path = PROJECT_DIR / "mcts_result" / "final_result"
    # feedthrough/FLUTE 预测器源码与可执行文件所在目录。
    feedthrough_source_dir: Path = PROJECT_DIR / "feedthrough"
    # MCTS 输入基准模拟次数；Basic 动态模式下作为 N_base。
    simulations: int = 4096
    # 随机种子，用于保证 MCTS 随机补全可复现。
    random_seed: int = 7
    # 容量不足时是否允许最终 fallback 强制 overflow 分配。
    allow_overflow_fallback: bool = True
    # 最终指标和非零 feedthrough reward 是否启用 feedthrough 预测器。
    enable_feedthrough: bool = True
    # 找不到 ftpred 可执行文件时是否自动调用 CMake 编译；默认关闭，要求预先编译好。
    auto_build_feedthrough: bool = False
    # Windows 下 CMake 使用的 generator；Linux/外部入口可传 None。
    cmake_generator: str = "MinGW Makefiles"
    # reward 中归一化 HPWL improvement 的权重。
    wirelength_reward_weight: float = 1.0
    # reward 中归一化 feedthrough improvement 的权重；0 表示 MCTS 不计算 FT reward。
    feedthrough_weight: float = 0.0
    # reward 归一化分母下限，避免参考值为 0 或过小。
    reward_normalization_floor: float = 1.0
    # 最终 reward 整体放大系数，用于调整 UCB exploitation 量级。
    reward_scale: float = 100.0
    # 是否按边长百分位对 block 边进行 segment 细分。
    enable_segment_subdivision: bool = True
    # segment 细分使用的边长百分位阈值，例如 50 表示 L50。
    segment_length_percentile: int = 50
    # 是否导出接口需要的 final_result 格式。
    export_interface_result: bool = True
    # MCTS 搜索模式，可选 "layered" 或 "basic"。
    mcts_search_mode: str = "layered"
    # layered 模式逐层预算衰减系数。
    mcts_budget_decay: float = 0.6
    # layered 模式进入长尾深度后的预算衰减系数。
    mcts_tail_decay: float = 0.9
    # layered 模式估算搜索空间时使用的典型深度。
    mcts_typical_depth: int = 6
    # layered 模式搜索空间放大因子的归一化除数。
    mcts_space_scale_divisor: float = 1_000_000.0
    # layered 模式搜索空间放大因子的上限。
    mcts_max_space_factor: float = 10.0
    # layered 模式每层最小模拟次数。
    mcts_min_layer_simulations: int = 256
    # layered 模式切换到长尾衰减和早停判断的深度阈值。
    mcts_tail_depth: int = 8
    # layered 模式早停判断中领先幅度相对标准差的倍数。
    mcts_early_stop_std_multiplier: float = 2.0
    # 是否启用 layered 模式长尾早停。
    mcts_enable_tail_early_stop: bool = True
    # Basic 模式是否根据总搜索空间动态调整模拟次数。
    mcts_basic_dynamic_simulations: bool = True
    # Basic 模式总搜索空间放大/缩小因子的归一化除数。
    mcts_basic_space_scale_divisor: float = 1_000_000.0
    # Basic 模式动态模拟次数放大因子的上限。
    mcts_basic_max_space_factor: float = 10.0
    # Basic 模式动态模拟次数下限。
    mcts_basic_min_simulations: int = 256

    def to_record(self) -> Dict[str, Any]:
        """转换为可保存到 JSON 中的参数记录。"""
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value)
        return values


DEFAULT_CONFIG = RunConfig()
