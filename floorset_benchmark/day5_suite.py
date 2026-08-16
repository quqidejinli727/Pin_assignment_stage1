"""Generate and validate Day-5 1K preflight and 20K R60/R95 cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from assignment_solver import AssignmentSolver

from .hierarchical_suite import HierarchicalCaseConfig, write_hierarchical_case


DAY5_SUITE_VERSION = "floorset-day5-20k-v0.1"
DEFAULT_CASES = (
    ("1k_r60", 1_000, 0.60),
    ("1k_r95", 1_000, 0.95),
    ("20k_r60", 20_000, 0.60),
    ("20k_r95", 20_000, 0.95),
)
CORE_DETERMINISM_FILES = (
    "block.json",
    "pingroup.json",
    "hierarchy_report.json",
    "net_distribution_report.json",
    "reuse_connectivity_report.json",
    "capacity_report.json",
    "lineage.json",
    "provenance.json",
    "manifest.json",
    "validation.json",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _assignment_consistency(case_dir: Path, assignment: dict[str, Any]) -> dict[str, Any]:
    pingroup = _read(case_dir / "pingroup.json")
    reuse = _read(case_dir / "reuse_connectivity_report.json")
    pin_to_segment: dict[str, str] = {}
    for segment_id, segment in assignment.get("segments", {}).items():
        for instance in segment.get("segment_instances", {}).values():
            for full_name in instance.get("assigned_pins", []):
                pin_to_segment[str(full_name)] = str(segment_id)
    group_segments: dict[str, set[str]] = {}
    singleton_pins: set[str] = set()
    for net in pingroup:
        if len(net) == 1:
            singleton_pins.add(f"{net[0]['parent_inst']}.{net[0]['pingroup_name']}")
        for pin in net:
            full_name = f"{pin['parent_inst']}.{pin['pingroup_name']}"
            group = f"{pin['parent_module']}.{pin['pingroup_name']}"
            segment_id = pin_to_segment.get(full_name)
            if segment_id is not None:
                group_segments.setdefault(group, set()).add(segment_id)
    mismatch_groups = sorted(
        group for group, segment_ids in group_segments.items() if len(segment_ids) != 1
    )
    unassigned_singletons = sorted(singleton_pins - set(pin_to_segment))
    return {
        "assigned_pin_lookup_count": len(pin_to_segment),
        "homology_relative_segment_mismatch_count": len(mismatch_groups),
        "homology_relative_segment_mismatch_groups": mismatch_groups,
        "singleton_pin_count": len(singleton_pins),
        "assigned_singleton_pin_count": len(singleton_pins) - len(unassigned_singletons),
        "unassigned_singleton_pin_count": len(unassigned_singletons),
        "unassigned_singleton_pins": unassigned_singletons,
        "expected_singleton_net_count": reuse["expected_singleton_net_count"],
    }


def _case_acceptance(
    hierarchy: dict[str, Any],
    nets: dict[str, Any],
    reuse: dict[str, Any],
    capacity: dict[str, Any],
    validation: dict[str, Any],
    assignment: dict[str, Any],
    consistency: dict[str, Any],
    target: int,
    reuse_rate: float,
) -> dict[str, bool]:
    expected_reused = 144 if reuse_rate == 0.60 else 228
    pattern_counts = reuse["pattern_template_counts"]
    remap_counts = reuse["remapped_endpoint_counts"]
    summary = assignment["summary"]
    return {
        "exact_240_nonroot_instances": hierarchy["nonroot_instance_count"] == 240,
        "exact_reused_instances": hierarchy["reused_module_instance_count"] == expected_reused,
        "exact_pingroup_count": nets["pingroup_count"] == target,
        "all_eight_hierarchical_types_nonzero": all(
            nets["net_type_counts"].get(name, 0) > 0
            for name in (
                "local_leaf_to_leaf",
                "leaf_to_parent_gateway",
                "intra_parent_gateway",
                "cross_parent_leaf_to_leaf",
                "parent_to_parent",
                "cross_parent_gateway_chain",
                "multicast_fanout",
                "ancestor_cross_leaf",
            )
        ),
        "all_reuse_patterns_nonzero": all(pattern_counts.get(name, 0) > 0 for name in (
            "aligned_pair", "dangling_counterpart", "remapped_counterpart"
        )),
        "aligned_pair_is_majority": pattern_counts.get("aligned_pair", 0) > sum(
            count for name, count in pattern_counts.items() if name != "aligned_pair"
        ),
        "all_remap_modes_nonzero": all(remap_counts.get(name, 0) > 0 for name in (
            "local", "cross_parent", "gateway"
        )),
        "singleton_contract_exact": (
            reuse["expected_singleton_net_count"] == reuse["actual_singleton_net_count"]
            and validation["nets"]["unprovenanced_singleton_net_count"] == 0
            and validation["nets"]["orphan_singleton_count"] == 0
        ),
        "full_pins_unique": reuse["full_pin_multi_net_violation_count"] == 0,
        "homology_widths_consistent": reuse["homology_width_mismatch_count"] == 0,
        "successors_valid": (
            nets["validation"]["missing_successor_count"] == 0
            and nets["validation"]["successor_cycle_count"] == 0
        ),
        "small_ratio_at_least_70_percent": capacity["small_group_ratio"] >= 0.70,
        "packing_valid": (
            capacity["packing"]["rejected_group_count"] == 0
            and capacity["packing"]["hard_overflow_count"] == 0
        ),
        "integration_valid": validation["valid"],
        "all_solver_groups_assigned": summary["unassigned_group_count"] == 0,
        "solver_capacity_valid": summary["capacity_violation_count"] == 0,
        "homology_relative_segments_consistent": (
            consistency["homology_relative_segment_mismatch_count"] == 0
        ),
        "all_singletons_assigned": consistency["unassigned_singleton_pin_count"] == 0,
    }


def run_case(
    source_dir: Path,
    output_root: Path,
    tag: str,
    target: int,
    reuse_rate: float,
    *,
    source_sample_id: str,
    source_commit: str,
    seed: int,
    seed_namespace: str,
    case_id_override: str | None = None,
) -> dict[str, Any]:
    case_dir = output_root / tag
    config = HierarchicalCaseConfig(
        case_id=case_id_override or f"day5-{tag}",
        target_pingroup_count=target,
        target_reuse_rate=reuse_rate,
        source_sample_id=source_sample_id,
        source_commit=source_commit,
        base_seed=seed,
        seed_namespace=seed_namespace,
    )
    rss_before = _rss_bytes()
    generated = write_hierarchical_case(source_dir, case_dir, config)
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
    )
    assignment = solver.write_output(str(case_dir / "assignment_smoke.json"))
    assignment_seconds = time.perf_counter() - assignment_started
    consistency = _assignment_consistency(case_dir, assignment)
    hierarchy = _read(case_dir / "hierarchy_report.json")
    nets = _read(case_dir / "net_distribution_report.json")
    reuse = _read(case_dir / "reuse_connectivity_report.json")
    capacity = _read(case_dir / "capacity_report.json")
    validation = _read(case_dir / "validation.json")
    acceptance = _case_acceptance(
        hierarchy,
        nets,
        reuse,
        capacity,
        validation,
        assignment,
        consistency,
        target,
        reuse_rate,
    )
    statistics_path = case_dir / "case_statistics.json"
    statistics = _read(statistics_path)
    statistics["assignment"] = {
        "summary": assignment["summary"],
        "consistency": consistency,
        "wall_seconds": assignment_seconds,
    }
    statistics["process_rss_bytes"] = {
        "before": rss_before,
        "after": _rss_bytes(),
    }
    statistics["acceptance"] = acceptance
    statistics["accepted"] = all(acceptance.values())
    statistics["artifact_bytes"] = {
        path.name: path.stat().st_size for path in sorted(case_dir.glob("*.json"))
        if path.name != "case_statistics.json"
    }
    statistics["artifact_sha256"] = {
        path.name: _sha256(path) for path in sorted(case_dir.glob("*.json"))
        if path.name != "case_statistics.json"
    }
    _write(statistics_path, statistics)
    if not statistics["accepted"]:
        failed = [name for name, passed in acceptance.items() if not passed]
        raise ValueError(f"Day-5 case {tag} failed acceptance: {failed!r}")
    return {
        "tag": tag,
        "config": asdict(config),
        "generated": generated,
        "assignment_summary": assignment["summary"],
        "assignment_consistency": consistency,
        "assignment_seconds": assignment_seconds,
        "acceptance": acceptance,
        "accepted": True,
        "case_statistics": str(statistics_path),
    }


def compare_deterministic(left: Path, right: Path) -> dict[str, Any]:
    comparisons = {
        name: {
            "left": _sha256(left / name),
            "right": _sha256(right / name),
            "equal": _sha256(left / name) == _sha256(right / name),
        }
        for name in CORE_DETERMINISM_FILES
    }
    return {
        "files": comparisons,
        "all_equal": all(item["equal"] for item in comparisons.values()),
    }


def run_suite(
    source_dir: str | Path,
    output_root: str | Path,
    *,
    selected_tags: set[str] | None = None,
    source_sample_id: str = "floorset_config21_1",
    source_commit: str = "unknown",
    seed: int = 7,
    seed_namespace: str = "day5-20k",
    deterministic_repeat: bool = True,
) -> dict[str, Any]:
    source_dir, output_root = Path(source_dir), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = selected_tags or {tag for tag, _, _ in DEFAULT_CASES}
    cases: list[dict[str, Any]] = []
    for tag, target, reuse_rate in DEFAULT_CASES:
        if tag not in selected:
            continue
        cases.append(
            run_case(
                source_dir,
                output_root,
                tag,
                target,
                reuse_rate,
                source_sample_id=source_sample_id,
                source_commit=source_commit,
                seed=seed,
                seed_namespace=seed_namespace,
            )
        )
    deterministic = None
    if deterministic_repeat and "20k_r60" in selected:
        repeat = run_case(
            source_dir,
            output_root,
            "repeat_20k_r60",
            20_000,
            0.60,
            source_sample_id=source_sample_id,
            source_commit=source_commit,
            seed=seed,
            seed_namespace=seed_namespace,
            case_id_override="day5-20k_r60",
        )
        deterministic = compare_deterministic(
            output_root / "20k_r60", output_root / "repeat_20k_r60"
        )
        repeat["deterministic_core_artifacts"] = deterministic
    summary = {
        "suite_version": DAY5_SUITE_VERSION,
        "source_dir": str(source_dir),
        "seed": seed,
        "seed_namespace": seed_namespace,
        "cases": cases,
        "deterministic_repeat_20k_r60": deterministic,
        "accepted": all(case["accepted"] for case in cases)
        and (deterministic is None or deterministic["all_equal"]),
    }
    _write(output_root / "suite_summary.json", summary)
    r60 = next((case for case in cases if case["tag"] == "20k_r60"), None)
    r95 = next((case for case in cases if case["tag"] == "20k_r95"), None)
    if r60 and r95:
        comparison = {
            "suite_version": DAY5_SUITE_VERSION,
            "r60": r60,
            "r95": r95,
            "shared_seed_namespace": seed_namespace,
            "assignment_seconds_delta": r95["assignment_seconds"] - r60["assignment_seconds"],
            "pin_count_delta": (
                r95["assignment_summary"]["pin_count"] - r60["assignment_summary"]["pin_count"]
            ),
        }
        _write(output_root / "20k_r60_vs_r95.json", comparison)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--cases",
        default=",".join(tag for tag, _, _ in DEFAULT_CASES),
        help="comma-separated subset of 1k_r60,1k_r95,20k_r60,20k_r95",
    )
    parser.add_argument("--source-sample-id", default="floorset_config21_1")
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seed-namespace", default="day5-20k")
    parser.add_argument("--no-deterministic-repeat", action="store_true")
    args = parser.parse_args()
    selected = {item.strip() for item in args.cases.split(",") if item.strip()}
    known = {tag for tag, _, _ in DEFAULT_CASES}
    if not selected <= known:
        raise ValueError(f"unknown case tags: {sorted(selected - known)!r}")
    result = run_suite(
        args.source_dir,
        args.output_root,
        selected_tags=selected,
        source_sample_id=args.source_sample_id,
        source_commit=args.source_commit,
        seed=args.seed,
        seed_namespace=args.seed_namespace,
        deterministic_repeat=not args.no_deterministic_repeat,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
