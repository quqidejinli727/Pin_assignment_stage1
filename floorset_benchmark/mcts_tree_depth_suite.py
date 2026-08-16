"""Generate and verify cross-Net-homology MCTS-depth calibration cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from assignment_solver import AssignmentSolver
from config import DEFAULT_CONFIG

from .hierarchical_suite import HierarchicalCaseConfig, write_hierarchical_case


SUITE_VERSION = "mcts-tree-depth-calibration-suite-v0.1"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assignment_consistency(case_dir: Path, assignment: dict[str, Any]) -> dict[str, Any]:
    pingroup = _read(case_dir / "pingroup.json")
    groups = {
        f"{pin['parent_module']}.{pin['pingroup_name']}"
        for net in pingroup
        for pin in net
    }
    return {
        "expected_group_count": len(groups),
        "assignment_group_count": assignment["summary"]["homology_group_count"],
        "unassigned_group_count": assignment["summary"]["unassigned_group_count"],
        "unexpected_assignment_count": max(
            0, assignment["summary"]["homology_group_count"] - len(groups)
        ),
    }


def run_case(
    source_dir: Path,
    output_root: Path,
    *,
    target: int,
    reuse_rate: float,
    seed: int,
    source_sample_id: str,
    source_commit: str,
    seed_namespace: str | None = None,
) -> dict[str, Any]:
    tier = f"r{int(reuse_rate * 100):02d}"
    tag = f"{target // 1000}k_{tier}"
    case_dir = output_root / tag
    started = time.perf_counter()
    generated = write_hierarchical_case(
        source_dir,
        case_dir,
        HierarchicalCaseConfig(
            case_id=f"mcts-depth-{tag}",
            target_pingroup_count=target,
            target_reuse_rate=reuse_rate,
            source_sample_id=source_sample_id,
            source_commit=source_commit,
            base_seed=seed,
            seed_namespace=seed_namespace or f"mcts-depth-{target}",
            capacity_profile="day6_30k" if target >= 30_000 else "day5",
            reuse_cluster_strategy=("mixed2to10" if reuse_rate == 0.60 else "pairs"),
            mcts_tree_depth_profile=True,
            tree_profile_group_budget_fraction=0.60,
            reuse_template_fraction=0.01,
        ),
    )
    assignment_started = time.perf_counter()
    solver = AssignmentSolver(
        str(case_dir / "block.json"),
        str(case_dir / "pingroup.json"),
        simulations=1,
        random_seed=seed,
        allow_overflow_fallback=False,
        enable_segment_subdivision=False,
        enable_feedthrough=False,
        feedthrough_weight=0.0,
        homology_use_fanout_reuse_for_sorting=(
            DEFAULT_CONFIG.homology_use_fanout_reuse_for_sorting
        ),
        homology_skip_uncovered_groups=DEFAULT_CONFIG.homology_skip_uncovered_groups,
        homology_skip_coverage_threshold=DEFAULT_CONFIG.homology_skip_coverage_threshold,
        homology_group_commit_coverage_threshold=(
            DEFAULT_CONFIG.homology_group_commit_coverage_threshold
        ),
        mcts_search_each_pingroup_once=DEFAULT_CONFIG.mcts_search_each_pingroup_once,
        assignment_rescan_until_stable=DEFAULT_CONFIG.assignment_rescan_until_stable,
    )
    assignment = solver.write_output(str(case_dir / "assignment_smoke.json"))
    assignment_seconds = time.perf_counter() - assignment_started
    consistency = _assignment_consistency(case_dir, assignment)
    depth = _read(case_dir / "mcts_tree_depth_report.json")
    capacity = _read(case_dir / "capacity_report.json")
    hierarchy = _read(case_dir / "hierarchy_report.json")
    validation = _read(case_dir / "validation.json")
    packing = capacity["packing"]
    acceptance = {
        "generated_validation": validation["valid"],
        "exact_pingroup_count": generated["pingroup_count"] == target,
        "exact_reuse_count": hierarchy["reused_module_instance_count"]
        == round(240 * reuse_rate),
        "tree_depth_acceptance": depth["acceptance_passed"],
        "multi_net_homology_nonzero": depth["multi_net_homology_group_count"] > 0,
        "single_net_max_below_runtime_max": (
            depth["net_pingroup_count"]["max"] < depth["effective_mcts_depth"]["max"]
        ),
        "capacity_group_deducted_once": (
            depth["capacity_deduction"]["deduction_count"] == target
        ),
        "capacity_rejected_zero": packing["rejected_group_count"] == 0,
        "hard_overflow_zero": packing["hard_overflow_count"] == 0,
        "assignment_complete": consistency["unassigned_group_count"] == 0,
        "assignment_capacity_valid": assignment["summary"]["capacity_violation_count"] == 0,
    }
    result = {
        "suite_version": SUITE_VERSION,
        "case_id": f"mcts-depth-{tag}",
        "output_dir": str(case_dir),
        "target_pingroup_count": target,
        "reuse_rate": reuse_rate,
        "reuse_cluster_size_histogram": hierarchy["reuse_cluster_size_histogram"],
        "effective_mcts_depth": depth["effective_mcts_depth"],
        "static_raw_depth": depth["static_raw_depth"],
        "ordered_raw_depth": depth["ordered_raw_depth"],
        "batch_component_depth": depth["batch_component_depth"],
        "batch_depth": depth["batch_depth"],
        "net_pingroup_count": depth["net_pingroup_count"],
        "multi_net_homology_group_count": depth["multi_net_homology_group_count"],
        "capacity_deduction": depth["capacity_deduction"],
        "assignment_summary": assignment["summary"],
        "assignment_consistency": consistency,
        "assignment_seconds": assignment_seconds,
        "total_seconds": time.perf_counter() - started,
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
    }
    _write(case_dir / "mcts_tree_depth_case_result.json", result)
    return result


def compare_cases(left: Path, right: Path) -> dict[str, Any]:
    files = (
        "block.json",
        "pingroup.json",
        "lineage.json",
        "capacity_report.json",
        "mcts_tree_depth_report.json",
    )
    checks = {
        name: {
            "left": _sha256(left / name),
            "right": _sha256(right / name),
        }
        for name in files
    }
    return {
        "files": checks,
        "all_equal": all(item["left"] == item["right"] for item in checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--targets", nargs="+", type=int, default=[1000, 3000])
    parser.add_argument("--reuse-rates", nargs="+", type=float, default=[0.60, 0.95])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--source-sample-id", default="floorset_config21_1")
    parser.add_argument("--source-commit", default="unknown")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    cases = [
        run_case(
            args.source_dir,
            args.output_root,
            target=target,
            reuse_rate=reuse_rate,
            seed=args.seed,
            source_sample_id=args.source_sample_id,
            source_commit=args.source_commit,
        )
        for target in args.targets
        for reuse_rate in args.reuse_rates
    ]
    summary = {
        "suite_version": SUITE_VERSION,
        "source_dir": str(args.source_dir),
        "cases": cases,
        "accepted": all(case["accepted"] for case in cases),
    }
    _write(args.output_root / "suite_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
