"""Default Stage1 configuration.

All values here are defaults. CLI/run-time overrides should build a copied
RunConfig and must not mutate DEFAULT_CONFIG.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class RunConfig:
    """Configuration values used by Stage1 pin assignment."""

    # Input/output paths
    block_json_path: Path = Path(
        r"C:\Users\DELL\Desktop\test_case_from_huawei\block_case2.json"
    )
    pingroup_json_path: Path = Path(
        r"C:\Users\DELL\Desktop\test_case_from_huawei\pingroup_case2.json"
    )
    assignment_output_path: Path = PROJECT_DIR / "outputs" / "case2_assignment.json"
    results_root: Path = PROJECT_DIR / "run_results"
    interface_result_dir: Path = PROJECT_DIR / "mcts_result" / "final_result"

    # Feedthrough settings
    feedthrough_source_dir: Path = PROJECT_DIR / "feedthrough"
    feedthrough_predict_source_dir: Path = PROJECT_DIR / "feedthrough_predict"
    feedthrough_evaluate_source_dir: Path = PROJECT_DIR / "feedthrough_evaluate"
    feedthrough_reward_source: str = "evaluate"
    enable_feedthrough: bool = True
    auto_build_feedthrough: bool = False
    cmake_generator: str = "MinGW Makefiles"

    # Common MCTS settings
    simulations: int = 5120
    random_seed: int = 7
    allow_overflow_fallback: bool = True
    mcts_search_mode: str = "hybrid"
    mcts_enable_search_diagnostics: bool = False
    mcts_search_each_pingroup_once: bool = False
    assignment_rescan_until_stable: bool = False

    # Basic mode settings
    mcts_basic_dynamic_simulations: bool = True
    mcts_basic_space_scale_divisor: float = 100_000.0
    mcts_basic_max_space_factor: float = 8.0
    mcts_basic_min_simulations: int = 1280
    mcts_enable_depth1_greedy: bool = True
    mcts_basic_depth1_simulations: int = 40
    mcts_basic_depth2_simulations: int = 640
    mcts_basic_disable_pruning_depth_limit: int = 2

    # Hybrid mode settings
    mcts_hybrid_basic_depth_limit: int = 4
    mcts_hybrid_basic_log_space_limit: float = math.log(50_000.0)
    mcts_hybrid_beam_width: int = 3
    mcts_hybrid_tail_beam_width: int = 4
    mcts_hybrid_tail_depth: int = 24
    mcts_hybrid_budget_decay: float = 0.65
    mcts_hybrid_tail_budget_decay: float = 0.92
    mcts_hybrid_min_layer_simulations: int = 128
    mcts_hybrid_max_layer_simulations: int = 1280
    mcts_hybrid_max_tree_simulations: int = 36_000
    mcts_hybrid_enable_layer_early_stop: bool = True
    mcts_hybrid_early_stop_std_multiplier: float = 2.0
    mcts_hybrid_time_limit_seconds: float = 0.0

    # Hybrid ultradeep profile settings
    mcts_hybrid_enable_ultradeep_profile: bool = False
    mcts_hybrid_ultradeep_depth: int = 100
    mcts_hybrid_max_expanded_depth: int = 64
    mcts_hybrid_ultradeep_beam_width: int = 1
    mcts_hybrid_ultradeep_min_layer_simulations: int = 40
    mcts_hybrid_ultradeep_max_layer_simulations: int = 160
    mcts_hybrid_use_fast_completion_for_ultradeep: bool = True

    # Homology settings
    homology_use_fanout_reuse_for_sorting: bool = False
    homology_skip_uncovered_groups: bool = False
    homology_skip_coverage_threshold: float = 1.0
    homology_group_commit_coverage_threshold: float = 1.0

    # Candidate pruning settings
    mcts_enable_candidate_pruning: bool = True
    mcts_candidate_top_k: int = 12
    mcts_candidate_tail_top_k: int = 8
    mcts_candidate_min_count: int = 16
    mcts_candidate_score_tolerance: float = 0.03

    # Reward settings
    wirelength_reward_weight: float = 0.9
    feedthrough_weight: float = 0.1
    reward_normalization_floor: float = 1.0
    reward_scale: float = 100.0

    # Segment/export settings
    enable_segment_subdivision: bool = True
    segment_length_percentile: int = 50
    export_interface_result: bool = True

    def to_record(self) -> Dict[str, Any]:
        """Return a JSON-serializable config record."""
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value)
        return values


DEFAULT_CONFIG = RunConfig()
