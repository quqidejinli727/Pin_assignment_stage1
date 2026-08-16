"""Summarize the six fixed-size hierarchical smoke cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CASE_TAGS = ("1k_r60", "1k_r95", "3k_r60", "3k_r95", "10k_r60", "10k_r95")
ARTIFACT_NAMES = (
    "block.json",
    "pingroup.json",
    "manifest.json",
    "lineage.json",
    "provenance.json",
    "validation.json",
    "capacity_report.json",
    "hierarchy_report.json",
    "net_distribution_report.json",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    cases: list[dict[str, Any]] = []
    for tag in CASE_TAGS:
        case_dir = root / tag
        hierarchy = _read(case_dir / "hierarchy_report.json")
        nets = _read(case_dir / "net_distribution_report.json")
        capacity = _read(case_dir / "capacity_report.json")
        validation = _read(case_dir / "validation.json")
        assignment = _read(case_dir / "assignment_smoke.json")
        cases.append(
            {
                "case": tag,
                "nonroot_module_instances": hierarchy["nonroot_instance_count"],
                "parent_count": hierarchy["parent_count"],
                "leaf_count": hierarchy["leaf_count"],
                "reused_module_instances": hierarchy["reused_module_instance_count"],
                "reuse_instance_ratio": hierarchy["reuse_instance_ratio"],
                "net_count": nets["net_count"],
                "pin_count": nets["pin_count"],
                "pingroup_count": nets["pingroup_count"],
                "net_type_counts": nets["net_type_counts"],
                "successor_edge_count": nets["validation"]["successor_edge_count"],
                "missing_successor_count": nets["validation"]["missing_successor_count"],
                "successor_cycle_count": nets["validation"]["successor_cycle_count"],
                "gateway_net_count": nets["validation"]["gateway_net_count"],
                "small_group_ratio": capacity["small_group_ratio"],
                "width_summary": capacity["width_summary"],
                "pressure_summary": capacity["pressure_summary"],
                "admission_scale": capacity["admission_scale"],
                "packing_rejected_group_count": capacity["packing"]["rejected_group_count"],
                "hard_overflow_count": capacity["packing"]["hard_overflow_count"],
                "solver_unassigned_group_count": assignment["summary"]["unassigned_group_count"],
                "solver_capacity_violation_count": assignment["summary"]["capacity_violation_count"],
                "feedthrough_child_net_count": validation["integration"]["feedthrough_child_net_count"],
                "feedthrough_missing_successors": validation["integration"]["feedthrough_missing_successors"],
                "validation_valid": validation["valid"],
            }
        )
    repeat = root / "repeat_1k_r60"
    deterministic = all(
        _sha256(root / "1k_r60" / name) == _sha256(repeat / name)
        for name in ARTIFACT_NAMES
    )
    return {
        "suite": "fixed_240_module_hierarchical_pingroup_smoke",
        "case_count": len(cases),
        "cases": cases,
        "deterministic_repeat_1k_r60": deterministic,
        "solver_settings": {
            "simulations_argument": 1,
            "seed": 7,
            "strict_capacity": True,
            "feedthrough_reward": False,
            "segment_subdivision": False,
            "purpose": "correctness smoke, not quality benchmark",
        },
        "acceptance": {
            "all_240_nonroot_instances": all(case["nonroot_module_instances"] == 240 for case in cases),
            "all_exact_pingroup_counts": all(
                case["pingroup_count"] in {1000, 3000, 10000} for case in cases
            ),
            "all_eight_net_types_nonzero": all(
                len(case["net_type_counts"]) >= 8
                and all(value > 0 for value in case["net_type_counts"].values())
                for case in cases
            ),
            "all_successors_valid": all(
                case["missing_successor_count"] == 0
                and case["successor_cycle_count"] == 0
                for case in cases
            ),
            "all_small_ratio_at_least_70_percent": all(
                case["small_group_ratio"] >= 0.70 for case in cases
            ),
            "all_solver_groups_assigned": all(
                case["solver_unassigned_group_count"] == 0 for case in cases
            ),
            "all_hard_overflow_zero": all(
                case["hard_overflow_count"] == 0
                and case["solver_capacity_violation_count"] == 0
                for case in cases
            ),
            "all_integration_valid": all(case["validation_valid"] for case in cases),
            "same_seed_byte_deterministic": deterministic,
        },
        "source_limitation": (
            "The checked-in legacy config21 artifact has no per-edge source_connectivity_weight; "
            "these cases use a clearly marked synthetic topology-weight prior until config21 is re-converted."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = summarize(args.root)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
