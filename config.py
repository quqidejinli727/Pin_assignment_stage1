"""程序运行参数的集中配置。

无需命令行参数时，直接修改 ``DEFAULT_CONFIG`` 的字段即可运行程序。
命令行参数仅用于临时覆盖这些默认设置。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class RunConfig:
    """保存单次 MCTS 分配与最终指标评估需要的全部可调参数。"""

    block_json_path: Path = Path(
        r"C:\Users\DELL\Desktop\test_case_from_huawei\block_case2.json"
    )
    pingroup_json_path: Path = Path(
        r"C:\Users\DELL\Desktop\test_case_from_huawei\pingroup_case2.json"
    )
    assignment_output_path: Path = PROJECT_DIR / "outputs" / "case2_assignment.json"
    results_root: Path = PROJECT_DIR / "run_results"
    interface_result_dir: Path = PROJECT_DIR / "mcts_result" / "final_result"
    feedthrough_source_dir: Path = PROJECT_DIR / "feedthrough"
    simulations: int = 4096
    random_seed: int = 7
    allow_overflow_fallback: bool = True
    enable_feedthrough: bool = True
    auto_build_feedthrough: bool = True
    cmake_generator: str = "MinGW Makefiles"
    wirelength_reward_weight: float = 1.0
    feedthrough_weight: float = 0.0
    reward_normalization_floor: float = 1.0
    reward_scale: float = 100.0
    enable_segment_subdivision: bool = True
    segment_length_percentile: int = 50
    export_interface_result: bool = True
    mcts_search_mode: str = "layered"
    mcts_budget_decay: float = 0.6
    mcts_tail_decay: float = 0.9
    mcts_typical_depth: int = 6
    mcts_space_scale_divisor: float = 1_000_000.0
    mcts_max_space_factor: float = 10.0
    mcts_min_layer_simulations: int = 256
    mcts_tail_depth: int = 8
    mcts_early_stop_std_multiplier: float = 2.0
    mcts_enable_tail_early_stop: bool = True

    def to_record(self) -> Dict[str, Any]:
        """转换为可保存到 JSON 中的参数记录。"""
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value)
        return values


DEFAULT_CONFIG = RunConfig()
