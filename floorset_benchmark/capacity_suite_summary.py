"""Summarize capacity-pressure smoke comparisons and solver assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assignment_map(output: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for segment_id, segment in output.get("segments", {}).items():
        for group_name in segment.get("assigned_groups", []):
            result[str(group_name)] = str(segment_id)
    return result


def summarize_suite(output_root: str | Path, targets: tuple[int, ...] = (1000, 3000, 10000)) -> dict[str, Any]:
    root = Path(output_root)
    cases: list[dict[str, Any]] = []
    for target in targets:
        comparison_dir = root / f"comparison_{target}"
        comparison = _read(comparison_dir / "comparison.json")
        baseline_report = _read(comparison_dir / "baseline" / "capacity_report.json")
        floorset_report = _read(comparison_dir / "floorset_first" / "capacity_report.json")
        validation = _read(comparison_dir / "floorset_first" / "validation_report.json")
        baseline_output = _read(root / f"assignment_{target}_baseline.json")
        floorset_output = _read(root / f"assignment_{target}_floorset_first.json")
        baseline_assignments = _assignment_map(baseline_output)
        floorset_assignments = _assignment_map(floorset_output)
        group_names = set(baseline_assignments) | set(floorset_assignments)
        solver_changes = sum(
            baseline_assignments.get(name) != floorset_assignments.get(name)
            for name in group_names
        )
        cases.append(
            {
                "target_pingroup_count": target,
                "module_instance_count": validation["validation"]["module_instance_count"],
                "validation_valid": validation["validation"]["valid"],
                "small_group_ratio": floorset_report["small_group_ratio"],
                "width_summary": floorset_report["width_summary"],
                "pressure_summary": floorset_report["pressure_summary"],
                "source_weight_to_pressure_monotonic": floorset_report[
                    "source_weight_to_pressure_monotonic"
                ],
                "legacy_rank_proxy_used": floorset_report["legacy_rank_proxy_used"],
                "admission_adjusted": floorset_report["admission_adjusted"],
                "admission_scale": floorset_report["admission_scale"],
                "packing_utilization_limit": floorset_report["packing"][
                    "packing_utilization_limit"
                ],
                "packing_assignment_changed_count": comparison["effect"][
                    "packing_assignment_changed_count"
                ],
                "solver_assignment_changed_count": solver_changes,
                "baseline_unassigned_group_count": baseline_output["summary"][
                    "unassigned_group_count"
                ],
                "floorset_first_unassigned_group_count": floorset_output["summary"][
                    "unassigned_group_count"
                ],
                "baseline_capacity_violation_count": baseline_output["summary"][
                    "capacity_violation_count"
                ],
                "floorset_first_capacity_violation_count": floorset_output["summary"][
                    "capacity_violation_count"
                ],
                "stress_enabled": comparison["stress_enabled"],
            }
        )
    summary = {
        "suite": "floorset_capacity_pressure_smoke",
        "fixed_module_instance_count": 200,
        "reuse_instance_ratio": 0.60,
        "same_topology_per_comparison": True,
        "solver_settings": {
            "seed": 7,
            "simulations_argument": 1,
            "strict_capacity": True,
            "feedthrough": False,
            "segment_subdivision": False,
            "purpose": "correctness smoke, not quality benchmark",
        },
        "source_limitation": (
            "The checked-in legacy config21 artifact predates source_connectivity_weight metadata; "
            "smoke cases use a deterministic rank proxy. Re-conversion with the updated converter "
            "is required to certify actual FloorSet-weight ordering."
        ),
        "cases": cases,
        "acceptance": {
            "all_valid": all(case["validation_valid"] for case in cases),
            "all_small_ratio_at_least_70_percent": all(
                case["small_group_ratio"] >= 0.70 for case in cases
            ),
            "all_solver_groups_assigned": all(
                case["floorset_first_unassigned_group_count"] == 0 for case in cases
            ),
            "all_hard_overflow_zero": all(
                case["floorset_first_capacity_violation_count"] == 0 for case in cases
            ),
            "all_have_observable_assignment_effect": all(
                case["solver_assignment_changed_count"] > 0 for case in cases
            ),
            "stress_required": any(case["stress_enabled"] for case in cases),
            "actual_floorset_weight_gate_complete": not any(
                case["legacy_rank_proxy_used"] for case in cases
            ),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = summarize_suite(args.output_root)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
