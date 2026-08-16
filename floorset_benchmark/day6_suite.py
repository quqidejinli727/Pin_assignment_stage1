"""Run Day-6 30K R60/R95 cases and a gated 200K feasibility experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .day5_suite import compare_deterministic, run_case
from .hierarchical_suite import HierarchicalCaseConfig, write_hierarchical_case


DAY6_SUITE_VERSION = "floorset-day6-30k-200k-gate-v0.1"
TARGET_30K = 30_000
TARGET_200K = 200_000
PRESSURE_LOW = {"small": 0.002, "normal": 0.020000001, "large": 0.100000001}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def capacity_projection(
    capacity_report: dict[str, Any],
    *,
    source_group_count: int = TARGET_30K,
    target_group_count: int = TARGET_200K,
) -> dict[str, Any]:
    """Project fixed-module lower-bound utilization from a measured source case."""
    scale = target_group_count / source_group_count
    modules = capacity_report["packing"]["module_types"]
    configured_demand: dict[str, float] = {name: 0.0 for name in modules}
    all_small_demand: dict[str, float] = {name: 0.0 for name in modules}
    for item in capacity_report["groups"].values():
        pressure = float(item["pressure"])
        width = float(item["width"])
        c_edge = width / pressure if pressure > 0 else 0.0
        module_name = str(item["module_name"])
        configured_demand[module_name] += c_edge * PRESSURE_LOW[str(item["bucket"])]
        all_small_demand[module_name] += c_edge * PRESSURE_LOW["small"]

    rows: dict[str, dict[str, float]] = {}
    for module_name, module in modules.items():
        usable_capacity = float(module["capacity"]) * float(
            capacity_report["packing"]["packing_utilization_limit"]
        )
        configured = configured_demand[module_name] * scale
        all_small = all_small_demand[module_name] * scale
        rows[module_name] = {
            "usable_capacity": usable_capacity,
            "configured_minimum_projected_demand": configured,
            "configured_minimum_projected_utilization": (
                configured / usable_capacity if usable_capacity else math.inf
            ),
            "all_small_projected_demand": all_small,
            "all_small_projected_utilization": (
                all_small / usable_capacity if usable_capacity else math.inf
            ),
        }
    configured_max = max(
        (row["configured_minimum_projected_utilization"] for row in rows.values()),
        default=0.0,
    )
    all_small_max = max(
        (row["all_small_projected_utilization"] for row in rows.values()),
        default=0.0,
    )
    configured_overloaded = sorted(
        name
        for name, row in rows.items()
        if row["configured_minimum_projected_utilization"] > 1.0 + 1e-9
    )
    all_small_overloaded = sorted(
        name
        for name, row in rows.items()
        if row["all_small_projected_utilization"] > 1.0 + 1e-9
    )
    return {
        "source_group_count": source_group_count,
        "target_group_count": target_group_count,
        "linear_scale": scale,
        "configured_profile": "small=80%, normal=19%, large=1% at bucket lower bounds",
        "configured_minimum_max_module_utilization": configured_max,
        "configured_minimum_overloaded_module_count": len(configured_overloaded),
        "configured_minimum_overloaded_modules": configured_overloaded,
        "all_small_absolute_optimistic_max_module_utilization": all_small_max,
        "all_small_absolute_optimistic_overloaded_module_count": len(all_small_overloaded),
        "all_small_absolute_optimistic_overloaded_modules": all_small_overloaded,
        "module_types": rows,
    }


def evaluate_200k_conditions(output_root: Path) -> dict[str, Any]:
    projections: dict[str, Any] = {}
    resource_estimates: dict[str, Any] = {}
    reasons: list[str] = []
    scale = TARGET_200K / TARGET_30K
    complexity_scale = scale * math.log(TARGET_200K) / math.log(TARGET_30K)
    for tag in ("30k_r60", "30k_r95"):
        case_dir = output_root / tag
        capacity = _read(case_dir / "capacity_report.json")
        statistics = _read(case_dir / "case_statistics.json")
        projection = capacity_projection(capacity)
        projections[tag] = projection
        measured_bytes = sum(int(value) for value in statistics["artifact_bytes"].values())
        measured_peak = int(statistics["peak_python_allocation_bytes"])
        generation_seconds = float(statistics["timings"]["total_seconds"])
        assignment_seconds = float(statistics["assignment"]["wall_seconds"])
        resource_estimates[tag] = {
            "projected_artifact_bytes_linear": int(math.ceil(measured_bytes * scale)),
            "projected_peak_python_bytes_linear": int(math.ceil(measured_peak * scale)),
            "projected_generation_seconds_nlogn": generation_seconds * complexity_scale,
            "projected_assignment_seconds_linear": assignment_seconds * scale,
        }
        if projection["configured_minimum_overloaded_module_count"]:
            reasons.append(
                f"{tag}: configured 80/19/1 pressure lower bound overloads "
                f"{projection['configured_minimum_overloaded_module_count']} module types"
            )
        if projection["all_small_absolute_optimistic_overloaded_module_count"]:
            reasons.append(
                f"{tag}: even the non-production all-small optimistic bound overloads "
                f"{projection['all_small_absolute_optimistic_overloaded_module_count']} module types"
            )
        estimate = resource_estimates[tag]
        if estimate["projected_peak_python_bytes_linear"] > 2 * 1024**3:
            reasons.append(f"{tag}: projected Python peak exceeds 2 GiB")
        if estimate["projected_artifact_bytes_linear"] > 1024**3:
            reasons.append(f"{tag}: projected JSON artifacts exceed 1 GiB")
        if estimate["projected_assignment_seconds_linear"] > 20 * 60:
            reasons.append(f"{tag}: projected correctness assignment exceeds 20 minutes")
    capacity_go = all(
        projection["configured_minimum_overloaded_module_count"] == 0
        and projection["all_small_absolute_optimistic_overloaded_module_count"] == 0
        for projection in projections.values()
    )
    resource_go = all(
        estimate["projected_peak_python_bytes_linear"] <= 2 * 1024**3
        and estimate["projected_artifact_bytes_linear"] <= 1024**3
        and estimate["projected_assignment_seconds_linear"] <= 20 * 60
        for estimate in resource_estimates.values()
    )
    return {
        "experiment": "200k_fixed_240_module_conditional_gate",
        "decision": "GO" if capacity_go and resource_go else "NO_GO",
        "formal_200k_generation_permitted": capacity_go and resource_go,
        "capacity_gate_passed": capacity_go,
        "resource_gate_passed": resource_go,
        "reasons": reasons,
        "method": {
            "capacity": (
                "Scale measured 30K per-module C_edge demand to 200K with fixed geometry; "
                "evaluate both the production 80/19/1 bucket minima and an all-small "
                "absolute optimistic lower bound."
            ),
            "resources": "Linear bytes/RSS/assignment and n-log-n generation projection.",
            "module_count_or_geometry_change": False,
        },
        "capacity_projections": projections,
        "resource_estimates": resource_estimates,
    }


def run_suite(
    source_dir: str | Path,
    output_root: str | Path,
    *,
    source_sample_id: str = "floorset_config21_1",
    source_commit: str = "unknown",
    seed: int = 7,
    seed_namespace: str = "day6-30k",
    deterministic_repeat: bool = True,
    generate_200k_on_go: bool = False,
) -> dict[str, Any]:
    source_dir, output_root = Path(source_dir), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for tag, reuse_rate in (("30k_r60", 0.60), ("30k_r95", 0.95)):
        cases.append(
            run_case(
                source_dir,
                output_root,
                tag,
                TARGET_30K,
                reuse_rate,
                source_sample_id=source_sample_id,
                source_commit=source_commit,
                seed=seed,
                seed_namespace=seed_namespace,
                case_id_override=f"day6-{tag}",
            )
        )
    deterministic = None
    if deterministic_repeat:
        run_case(
            source_dir,
            output_root,
            "repeat_30k_r60",
            TARGET_30K,
            0.60,
            source_sample_id=source_sample_id,
            source_commit=source_commit,
            seed=seed,
            seed_namespace=seed_namespace,
            case_id_override="day6-30k_r60",
        )
        deterministic = compare_deterministic(
            output_root / "30k_r60", output_root / "repeat_30k_r60"
        )
    evaluation = evaluate_200k_conditions(output_root)
    _write(output_root / "200k_condition_evaluation.json", evaluation)
    generated_200k: list[dict[str, Any]] = []
    if generate_200k_on_go and evaluation["formal_200k_generation_permitted"]:
        for tag, reuse_rate in (("200k_r60", 0.60), ("200k_r95", 0.95)):
            destination = output_root / tag
            generated_200k.append(
                write_hierarchical_case(
                    source_dir,
                    destination,
                    HierarchicalCaseConfig(
                        case_id=f"day6-{tag}",
                        target_pingroup_count=TARGET_200K,
                        target_reuse_rate=reuse_rate,
                        source_sample_id=source_sample_id,
                        source_commit=source_commit,
                        base_seed=seed,
                        seed_namespace="day6-200k",
                        capacity_profile="day6_30k",
                    ),
                )
            )
    summary = {
        "suite_version": DAY6_SUITE_VERSION,
        "source_dir": str(source_dir),
        "seed": seed,
        "seed_namespace": seed_namespace,
        "cases": cases,
        "deterministic_repeat_30k_r60": deterministic,
        "200k_condition_evaluation": evaluation,
        "generated_200k_cases": generated_200k,
        "accepted": all(case["accepted"] for case in cases)
        and (deterministic is None or deterministic["all_equal"]),
    }
    _write(output_root / "suite_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-sample-id", default="floorset_config21_1")
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seed-namespace", default="day6-30k")
    parser.add_argument("--no-deterministic-repeat", action="store_true")
    parser.add_argument("--generate-200k-on-go", action="store_true")
    args = parser.parse_args()
    result = run_suite(
        args.source_dir,
        args.output_root,
        source_sample_id=args.source_sample_id,
        source_commit=args.source_commit,
        seed=args.seed,
        seed_namespace=args.seed_namespace,
        deterministic_repeat=not args.no_deterministic_repeat,
        generate_200k_on_go=args.generate_200k_on_go,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
