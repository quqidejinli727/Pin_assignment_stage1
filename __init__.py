"""Stage 1: MCTS segment assignment.

Provides the ``run_mcts`` entry point used by the top-level PinAssignFlow
``run.py`` script.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if _STAGE_DIR in sys.path:
    sys.path.remove(_STAGE_DIR)
sys.path.insert(0, _STAGE_DIR)

# Stage packages may contain modules with overlapping names. Clear Stage 1
# modules so this package always reloads them from this directory.
_STAGE1_MODULES = [
    "PlaceDB",
    "assignment_solver",
    "config",
    "export_final_result",
    "geometry_utils",
    "homology",
    "mcts",
    "scoring",
    "segment",
    "segment_subdivision",
]
for _mod in _STAGE1_MODULES:
    sys.modules.pop(_mod, None)


def run_mcts(
    block_json: str,
    pingroup_json: str,
    output_dir: str,
    num_simulations: int = 1000,
    time_limit: float | None = None,
    config_overrides: dict | None = None,
) -> str:
    """Run MCTS segment assignment.

    Args:
        block_json: Path to block.json from the benchmark case.
        pingroup_json: Path to pingroup.json from the benchmark case.
        output_dir: Directory for Stage 1 outputs.
        num_simulations: MCTS base simulation count.
        time_limit: Accepted for run.py compatibility; the current solver uses
            simulation count rather than a wall-clock limit.
        config_overrides: Optional Stage 1 RunConfig field overrides supplied
            by the top-level run.py command line.

    Returns:
        Path to the generated ``segment_assignments_*.json`` file.
    """
    from assignment_solver import AssignmentSolver
    from config import DEFAULT_CONFIG
    from export_final_result import write_config_record, write_interface_result
    from scoring import summarize_metrics

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if time_limit is not None:
        logger.info(
            "Stage 1 time_limit=%.1fs accepted for compatibility; "
            "current MCTS uses num_simulations.",
            time_limit,
        )

    overrides = dict(config_overrides or {})
    overrides.setdefault("simulations", num_simulations)

    config = replace(
        DEFAULT_CONFIG,
        **overrides,
    )
    config = replace(
        config,
        block_json_path=Path(block_json).resolve(),
        pingroup_json_path=Path(pingroup_json).resolve(),
        assignment_output_path=output_path / "stage1_assignment.json",
        results_root=output_path / "stage1_run_results",
        interface_result_dir=output_path,
    )
    solver = AssignmentSolver(
        block_json_path=str(config.block_json_path),
        pingroup_json_path=str(config.pingroup_json_path),
        simulations=config.simulations,
        random_seed=config.random_seed,
        allow_overflow_fallback=config.allow_overflow_fallback,
        enable_segment_subdivision=config.enable_segment_subdivision,
        segment_length_percentile=config.segment_length_percentile,
        mcts_search_mode=config.mcts_search_mode,
        mcts_enable_search_diagnostics=config.mcts_enable_search_diagnostics,
        mcts_budget_decay=config.mcts_budget_decay,
        mcts_tail_decay=config.mcts_tail_decay,
        mcts_typical_depth=config.mcts_typical_depth,
        mcts_space_scale_divisor=config.mcts_space_scale_divisor,
        mcts_max_space_factor=config.mcts_max_space_factor,
        mcts_min_layer_simulations=config.mcts_min_layer_simulations,
        mcts_tail_depth=config.mcts_tail_depth,
        mcts_early_stop_std_multiplier=config.mcts_early_stop_std_multiplier,
        mcts_enable_tail_early_stop=config.mcts_enable_tail_early_stop,
        mcts_basic_dynamic_simulations=config.mcts_basic_dynamic_simulations,
        mcts_basic_space_scale_divisor=config.mcts_basic_space_scale_divisor,
        mcts_basic_max_space_factor=config.mcts_basic_max_space_factor,
        mcts_basic_min_simulations=config.mcts_basic_min_simulations,
        mcts_basic_depth1_simulations=config.mcts_basic_depth1_simulations,
        mcts_basic_depth2_simulations=config.mcts_basic_depth2_simulations,
        mcts_basic_disable_pruning_depth_limit=config.mcts_basic_disable_pruning_depth_limit,
        mcts_hybrid_basic_depth_limit=config.mcts_hybrid_basic_depth_limit,
        mcts_hybrid_basic_log_space_limit=config.mcts_hybrid_basic_log_space_limit,
        mcts_hybrid_beam_width=config.mcts_hybrid_beam_width,
        mcts_hybrid_tail_beam_width=config.mcts_hybrid_tail_beam_width,
        mcts_hybrid_tail_depth=config.mcts_hybrid_tail_depth,
        mcts_hybrid_budget_decay=config.mcts_hybrid_budget_decay,
        mcts_hybrid_tail_budget_decay=config.mcts_hybrid_tail_budget_decay,
        mcts_hybrid_min_layer_simulations=config.mcts_hybrid_min_layer_simulations,
        mcts_hybrid_max_layer_simulations=config.mcts_hybrid_max_layer_simulations,
        mcts_hybrid_max_tree_simulations=config.mcts_hybrid_max_tree_simulations,
        mcts_hybrid_enable_layer_early_stop=config.mcts_hybrid_enable_layer_early_stop,
        mcts_hybrid_early_stop_std_multiplier=config.mcts_hybrid_early_stop_std_multiplier,
        mcts_hybrid_time_limit_seconds=config.mcts_hybrid_time_limit_seconds,
        mcts_hybrid_enable_ultradeep_profile=config.mcts_hybrid_enable_ultradeep_profile,
        mcts_hybrid_ultradeep_depth=config.mcts_hybrid_ultradeep_depth,
        mcts_hybrid_max_expanded_depth=config.mcts_hybrid_max_expanded_depth,
        mcts_hybrid_ultradeep_beam_width=config.mcts_hybrid_ultradeep_beam_width,
        mcts_hybrid_ultradeep_min_layer_simulations=config.mcts_hybrid_ultradeep_min_layer_simulations,
        mcts_hybrid_ultradeep_max_layer_simulations=config.mcts_hybrid_ultradeep_max_layer_simulations,
        mcts_hybrid_use_fast_completion_for_ultradeep=config.mcts_hybrid_use_fast_completion_for_ultradeep,
        homology_use_fanout_reuse_for_sorting=config.homology_use_fanout_reuse_for_sorting,
        homology_group_commit_coverage_threshold=config.homology_group_commit_coverage_threshold,
        mcts_tree_min_committable_group_ratio=config.mcts_tree_min_committable_group_ratio,
        mcts_enable_candidate_pruning=config.mcts_enable_candidate_pruning,
        mcts_candidate_top_k=config.mcts_candidate_top_k,
        mcts_candidate_tail_top_k=config.mcts_candidate_tail_top_k,
        mcts_candidate_min_count=config.mcts_candidate_min_count,
        mcts_candidate_score_tolerance=config.mcts_candidate_score_tolerance,
        wirelength_reward_weight=config.wirelength_reward_weight,
        feedthrough_weight=config.feedthrough_weight,
        reward_normalization_floor=config.reward_normalization_floor,
        reward_scale=config.reward_scale,
        feedthrough_source_dir=config.feedthrough_source_dir,
        feedthrough_predict_source_dir=config.feedthrough_predict_source_dir,
        feedthrough_evaluate_source_dir=config.feedthrough_evaluate_source_dir,
        feedthrough_reward_source=config.feedthrough_reward_source,
        enable_feedthrough=config.enable_feedthrough,
        auto_build_feedthrough=config.auto_build_feedthrough,
        cmake_generator=config.cmake_generator if os.name == "nt" else None,
    )
    try:
        assignment_result = solver.solve()
        config.assignment_output_path.write_text(
            json.dumps(assignment_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metrics = solver.final_net_metrics(
            feedthrough_source_dir=config.feedthrough_evaluate_source_dir,
            enable_feedthrough=config.enable_feedthrough,
            auto_build_feedthrough=config.auto_build_feedthrough,
            cmake_generator=config.cmake_generator if os.name == "nt" else None,
        )
        summary = summarize_metrics(metrics)
        print(
            "Stage 1 metrics: "
            f"total_hpwl={summary['total_hpwl']:.6f}, "
            f"total_feedthrough={summary['total_feedthrough']:.6f}"
        )
        logger.info(
            "Stage 1 metrics: total_hpwl=%.6f, total_feedthrough=%.6f",
            summary["total_hpwl"],
            summary["total_feedthrough"],
        )
        interface_timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        interface_result_path = write_interface_result(
            output_path,
            solver.placedb,
            solver.homology,
            solver.segment_manager,
            timestamp=interface_timestamp,
        )
        write_config_record(
            output_path,
            config.to_record(),
            interface_timestamp,
            result_path=interface_result_path,
        )
        return str(interface_result_path)
    finally:
        solver.close_feedthrough_context()
