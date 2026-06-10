"""Command-line entry point for MCTS pin assignment."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from assignment_solver import AssignmentSolver
from config import DEFAULT_CONFIG, RunConfig
from export_final_result import write_config_record, write_interface_result
from scoring import metrics_to_records, summarize_metrics


def parse_args() -> argparse.Namespace:
    """解析可选命令行参数；未提供时使用 config.py 中的默认值。"""
    parser = argparse.ArgumentParser(description="Run MCTS pin-to-segment assignment.")
    parser.add_argument("--block", default=None, help="Path to block JSON file.")
    parser.add_argument("--pingroup", default=None, help="Path to pingroup JSON file.")
    parser.add_argument("--output", default=None, help="Path to full assignment JSON output.")
    parser.add_argument("--results-root", default=None, help="Directory for independent run reports.")
    parser.add_argument("--interface-result-dir", default=None, help="Directory for interface-format output.")
    parser.add_argument("--simulations", type=int, default=None, help="MCTS simulations per local tree.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--strict-capacity",
        action="store_true",
        help="Disable final overflow fallback for groups wider than every candidate segment.",
    )
    parser.add_argument(
        "--no-feedthrough",
        action="store_true",
        help="Skip predictor execution and record feedthrough as 0.",
    )
    parser.add_argument(
        "--no-auto-build",
        action="store_true",
        help="Require a compiled predictor; kept for compatibility because this is now the default.",
    )
    parser.add_argument(
        "--disable-subdivision",
        action="store_true",
        help="Keep original polygon edges without percentile segment subdivision.",
    )
    parser.add_argument(
        "--segment-percentile",
        type=int,
        default=None,
        help="Percentile used as maximum segment length threshold, default L50.",
    )
    parser.add_argument(
        "--no-interface-result",
        action="store_true",
        help="Skip interface-format final_result export.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """使用命令行临时覆盖集中配置中的对应参数。"""
    return replace(
        DEFAULT_CONFIG,
        block_json_path=Path(args.block) if args.block else DEFAULT_CONFIG.block_json_path,
        pingroup_json_path=(
            Path(args.pingroup) if args.pingroup else DEFAULT_CONFIG.pingroup_json_path
        ),
        assignment_output_path=(
            Path(args.output) if args.output else DEFAULT_CONFIG.assignment_output_path
        ),
        results_root=Path(args.results_root) if args.results_root else DEFAULT_CONFIG.results_root,
        interface_result_dir=(
            Path(args.interface_result_dir)
            if args.interface_result_dir
            else DEFAULT_CONFIG.interface_result_dir
        ),
        simulations=args.simulations if args.simulations is not None else DEFAULT_CONFIG.simulations,
        random_seed=args.seed if args.seed is not None else DEFAULT_CONFIG.random_seed,
        allow_overflow_fallback=(
            False if args.strict_capacity else DEFAULT_CONFIG.allow_overflow_fallback
        ),
        enable_feedthrough=False if args.no_feedthrough else DEFAULT_CONFIG.enable_feedthrough,
        auto_build_feedthrough=(
            False if args.no_auto_build else DEFAULT_CONFIG.auto_build_feedthrough
        ),
        enable_segment_subdivision=(
            False if args.disable_subdivision else DEFAULT_CONFIG.enable_segment_subdivision
        ),
        segment_length_percentile=(
            args.segment_percentile
            if args.segment_percentile is not None
            else DEFAULT_CONFIG.segment_length_percentile
        ),
        export_interface_result=(
            False if args.no_interface_result else DEFAULT_CONFIG.export_interface_result
        ),
    )


def write_run_report(
    config: RunConfig,
    assignment_summary: dict,
    metrics: list,
) -> Path:
    """把单次运行参数和最终 Net 指标保存到独立结果目录。"""
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = config.results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "metrics.json"
    report = {
        "parameters": config.to_record(),
        "assignment_summary": assignment_summary,
        "metrics_summary": summarize_metrics(metrics),
        "net_metrics": metrics_to_records(metrics),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def run_pipline(config: RunConfig | None = None) -> Path:
    """执行 Stage 1 分配流程，并返回可供后续阶段使用的结果路径。"""
    if config is None:
        config = config_from_args(parse_args())
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
        cmake_generator=config.cmake_generator,
    )
    try:
        assignment_result = solver.solve()
        config.assignment_output_path.parent.mkdir(parents=True, exist_ok=True)
        config.assignment_output_path.write_text(
            json.dumps(assignment_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metrics = solver.final_net_metrics(
            feedthrough_source_dir=config.feedthrough_evaluate_source_dir,
            enable_feedthrough=config.enable_feedthrough,
            auto_build_feedthrough=config.auto_build_feedthrough,
            cmake_generator=config.cmake_generator,
        )
        report_path = write_run_report(config, assignment_result["summary"], metrics)
        interface_result_path = None
        config_record_path = None
        if config.export_interface_result:
            interface_timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            interface_result_path = write_interface_result(
                config.interface_result_dir,
                solver.placedb,
                solver.homology,
                solver.segment_manager,
                timestamp=interface_timestamp,
            )
            config_record_path = write_config_record(
                config.interface_result_dir,
                config.to_record(),
                interface_timestamp,
                result_path=interface_result_path,
            )
        print(json.dumps(assignment_result["summary"], ensure_ascii=False, indent=2))
        print(json.dumps(summarize_metrics(metrics), ensure_ascii=False, indent=2))
        print(f"Assignment output: {config.assignment_output_path}")
        print(f"Result report: {report_path}")
        if interface_result_path is not None:
            print(f"Interface result: {interface_result_path}")
            print(f"Config record: {config_record_path}")
            return interface_result_path
        return config.assignment_output_path
    finally:
        solver.close_feedthrough_context()


if __name__ == "__main__":
    run_pipline()
