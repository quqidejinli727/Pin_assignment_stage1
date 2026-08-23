"""Benchmark serial Stage1 MCTS against the asynchronous dynamic tree queue.

The serial baseline and every worker-count run receive the same solver
configuration.  In hybrid mode, ``--simulations`` sets the public solver
simulation value while all hybrid layer/tree scheduling parameters remain at
their repository defaults.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

from assignment_solver import AssignmentSolver
from config import DEFAULT_CONFIG
from stage1_dynamic_queue_experiment import run_dynamic_queue
from stage1_parallel_waves import _final_feedthrough_summary, _hashes, _hpwl, _solver_kwargs


HYBRID_DEFAULT_FIELDS = (
    "mcts_hybrid_basic_depth_limit",
    "mcts_hybrid_basic_log_space_limit",
    "mcts_hybrid_beam_width",
    "mcts_hybrid_tail_beam_width",
    "mcts_hybrid_tail_depth",
    "mcts_hybrid_budget_decay",
    "mcts_hybrid_tail_budget_decay",
    "mcts_hybrid_min_layer_simulations",
    "mcts_hybrid_max_layer_simulations",
    "mcts_hybrid_max_tree_simulations",
    "mcts_hybrid_enable_layer_early_stop",
    "mcts_hybrid_early_stop_std_multiplier",
    "mcts_hybrid_time_limit_seconds",
)


def solver_kwargs(
    block: Path,
    pingroup: Path,
    *,
    simulations: int,
    seed: int,
    search_mode: str,
    feedthrough_weight: float,
    feedthrough_source: Path,
) -> dict:
    """Build one shared serial/dynamic configuration without retuning hybrid."""

    values = _solver_kwargs(str(block), str(pingroup), simulations, seed)
    values.update(
        simulations=int(simulations),
        random_seed=int(seed),
        mcts_search_mode=str(search_mode),
        enable_feedthrough=True,
        feedthrough_weight=float(feedthrough_weight),
        feedthrough_reward_source="evaluate",
        feedthrough_source_dir=str(feedthrough_source),
        feedthrough_predict_source_dir=str(feedthrough_source),
        feedthrough_evaluate_source_dir=str(feedthrough_source),
        auto_build_feedthrough=False,
    )
    if search_mode == "hybrid":
        values.update(
            {
                name: getattr(DEFAULT_CONFIG, name)
                for name in HYBRID_DEFAULT_FIELDS
            }
        )
    return values


def run_serial(kwargs: Mapping[str, object]) -> dict:
    """Run the baseline while separating one-time FT lifecycle phases."""

    init_started = time.perf_counter()
    solver = AssignmentSolver(**dict(kwargs))
    init_seconds = time.perf_counter() - init_started

    cold_started = time.perf_counter()
    solver._open_feedthrough_context_if_needed()
    cold_seconds = time.perf_counter() - cold_started
    core_started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        output = solver._solve_with_open_context()
    core_seconds = time.perf_counter() - core_started

    hpwl = _hpwl(solver)
    hashes = _hashes(solver)
    final_feedthrough = _final_feedthrough_summary(solver)
    shutdown_started = time.perf_counter()
    solver.close_feedthrough_context()
    shutdown_seconds = time.perf_counter() - shutdown_started
    wall_seconds = (
        cold_seconds
        + core_seconds
        + float(final_feedthrough["seconds"])
        + shutdown_seconds
    )
    return {
        "performance": {
            "input_init_seconds": init_seconds,
            "worker_ft_cold_start_seconds": cold_seconds,
            "core_warm_seconds": core_seconds,
            "final_feedthrough_seconds": final_feedthrough["seconds"],
            "shutdown_seconds": shutdown_seconds,
            "measured_wall_seconds": wall_seconds,
            "reward_feedthrough_seconds": solver.mcts_reward_feedthrough_seconds,
            "reward_feedthrough_eval_seconds": solver.mcts_reward_feedthrough_eval_seconds,
            "reward_feedthrough_location_seconds": solver.mcts_reward_feedthrough_location_seconds,
        },
        "result": {
            "summary": output["summary"],
            "assignment_issues": output["assignment_issues"],
            "capacity_violations": output["capacity_violations"],
            "assignment_hashes": hashes,
            "hpwl": hpwl,
            "feedthrough": final_feedthrough,
            "total_mcts_simulations": solver.total_mcts_simulations,
        },
    }


def compact_dynamic(report: Mapping[str, object]) -> dict:
    """Drop per-tree traces while retaining scheduler and correctness totals."""

    scheduler = dict(report["scheduler"])
    scheduler.pop("tree_records", None)
    scheduler.pop("completion_tree_order", None)
    feedthrough = dict(report["feedthrough_workers"])
    trace_reports = []
    for trace in feedthrough.get("trace_reports", ()):
        trace = dict(trace)
        trace.pop("intervals", None)
        trace_reports.append(trace)
    feedthrough["trace_reports"] = trace_reports
    concurrency = feedthrough.get("trace_concurrency")
    if isinstance(concurrency, dict):
        concurrency = dict(concurrency)
        concurrency.pop("worker_intervals", None)
        feedthrough["trace_concurrency"] = concurrency
    return {
        "tree_analysis": report["tree_analysis"],
        "performance": report["performance"],
        "scheduler": scheduler,
        "commit": report["commit"],
        "feedthrough": feedthrough,
        "result": report["result"],
    }


def comparison(serial: Mapping[str, object], dynamic: Mapping[str, object]) -> dict:
    serial_result = serial["result"]
    dynamic_result = dynamic["result"]
    serial_core = float(serial["performance"]["core_warm_seconds"])
    dynamic_core = float(dynamic["performance"]["core_adjusted_seconds"])
    serial_wall = float(serial["performance"]["measured_wall_seconds"])
    dynamic_wall = float(dynamic["performance"]["wall_seconds"])
    serial_hpwl = float(serial_result["hpwl"])
    dynamic_hpwl = float(dynamic_result["hpwl"])
    serial_ft = float(serial_result["feedthrough"]["total"])
    dynamic_ft = float(dynamic_result["feedthrough"]["total"])
    return {
        "core_speedup": serial_core / dynamic_core if dynamic_core else None,
        "wall_speedup": serial_wall / dynamic_wall if dynamic_wall else None,
        "simulation_count_equal": int(serial_result["total_mcts_simulations"])
        == int(dynamic_result["summary"]["total_mcts_simulations"]),
        "hpwl_delta_percent": 100.0 * (dynamic_hpwl - serial_hpwl) / serial_hpwl,
        "feedthrough_delta_percent": 100.0 * (dynamic_ft - serial_ft) / serial_ft
        if serial_ft
        else None,
        "serial_complete": int(serial_result["summary"]["unassigned_group_count"]) == 0,
        "dynamic_complete": int(dynamic_result["summary"]["unassigned_group_count"]) == 0,
        "serial_capacity_safe": not serial_result["capacity_violations"],
        "dynamic_capacity_safe": not dynamic_result["capacity_violations"],
    }


def write_checkpoint(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=Path, required=True)
    parser.add_argument("--pingroup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=1024)
    parser.add_argument("--workers", type=int, nargs="+", default=(1, 2, 4, 6))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--search-mode", choices=("basic", "hybrid"), default="hybrid")
    parser.add_argument("--feedthrough-weight", type=float, default=0.1)
    parser.add_argument("--feedthrough-source", type=Path, default=Path(__file__).parent / "feedthrough")
    parser.add_argument("--trace-feedthrough", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workers: Sequence[int] = tuple(dict.fromkeys(max(1, item) for item in args.workers))
    kwargs = solver_kwargs(
        args.block,
        args.pingroup,
        simulations=args.simulations,
        seed=args.seed,
        search_mode=args.search_mode,
        feedthrough_weight=args.feedthrough_weight,
        feedthrough_source=args.feedthrough_source.resolve(),
    )
    report = {
        "input": {
            "block": str(args.block.resolve()),
            "pingroup": str(args.pingroup.resolve()),
        },
        "parameters": {
            "search_mode": args.search_mode,
            "simulations": args.simulations,
            "seed": args.seed,
            "workers": list(workers),
            "feedthrough_enabled": True,
            "feedthrough_weight": args.feedthrough_weight,
            "feedthrough_reward_source": "evaluate",
            "hybrid_budget_source": "repository_defaults",
            "effective_hybrid_budget": {
                name: kwargs[name] for name in HYBRID_DEFAULT_FIELDS
            },
        },
        "serial": None,
        "dynamic": {},
        "comparisons": {},
    }
    report["serial"] = run_serial(kwargs)
    write_checkpoint(args.output, report)
    print("serial complete", flush=True)

    for worker_count in workers:
        with contextlib.redirect_stdout(io.StringIO()):
            raw = run_dynamic_queue(
                args.block,
                args.pingroup,
                workers=worker_count,
                simulations=args.simulations,
                random_seed=args.seed,
                trace_feedthrough=args.trace_feedthrough,
                solver_overrides=kwargs,
            )
        dynamic = compact_dynamic(raw)
        key = str(worker_count)
        report["dynamic"][key] = dynamic
        report["comparisons"][key] = comparison(report["serial"], dynamic)
        write_checkpoint(args.output, report)
        print(
            f"workers={worker_count} complete; "
            f"core_speedup={report['comparisons'][key]['core_speedup']:.6f}",
            flush=True,
        )

    print(json.dumps(report["comparisons"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
