"""Build, validate, and write fixed-size hierarchical FloorSet-derived cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PlaceDB import PlaceDB
from feedthrough.ftpred_loader import build_nets_text_all
from homology import HomologyManager
from plan_mcts_batches import BatchPlanner
from segment import SegmentManager

from .capacity_calibration import CapacityCalibrationConfig, calibrate_case
from .hierarchical_net_synthesizer import (
    HierarchicalNetSynthesizer,
    NetSynthesisConfig,
    build_reuse_connectivity_report,
    validate_hierarchical_nets,
)
from .hierarchy_composer import HierarchyConfig, compose_hierarchy, validate_hierarchy
from .mcts_tree_depth_analysis import analyze_mcts_tree_depths
from .single_converter import FLOORSET_DATA_LICENSE, FLOORSET_REPOSITORY, validate_artifacts


HIERARCHICAL_SUITE_VERSION = "floorset-hierarchical-suite-v0.3"


@dataclass(frozen=True)
class HierarchicalCaseConfig:
    case_id: str
    target_pingroup_count: int
    target_reuse_rate: float
    source_sample_id: str
    source_commit: str = "unknown"
    base_seed: int = 7
    seed_namespace: str | None = None
    capacity_profile: str = "auto"
    reuse_cluster_strategy: str = "pairs"
    mcts_tree_depth_profile: bool = False
    tree_profile_group_budget_fraction: float = 0.60
    reuse_template_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.capacity_profile not in {"auto", "day5", "day6_30k"}:
            raise ValueError("capacity_profile must be auto, day5, or day6_30k")
        if self.reuse_cluster_strategy not in {"pairs", "mixed2to10"}:
            raise ValueError("reuse_cluster_strategy must be pairs or mixed2to10")


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integration_report(case_dir: Path, expected_pingroup_count: int) -> dict[str, Any]:
    placedb = PlaceDB(str(case_dir / "block.json"), str(case_dir / "pingroup.json"))
    homology = HomologyManager(placedb)
    segments = SegmentManager(placedb)
    planner = BatchPlanner(placedb, homology)
    plan = planner.plan()
    _, feedthrough = build_nets_text_all(placedb, return_stats=True)
    singleton_net_count = sum(len(net.pins) == 1 for net in placedb.nets_list)
    multi_net_homology_group_count = sum(
        len(homology.get_related_nets(group)) > 1
        for group in homology.pin_groups.values()
    )
    return {
        "placedb_module_count": len(placedb.all_modules_list),
        "placedb_nonroot_module_count": len(placedb.all_modules_list) - 1,
        "net_count": len(placedb.nets_list),
        "pin_count": placedb.total_pin_count,
        "homology_pingroup_count": len(homology.pin_groups),
        "expected_pingroup_count": expected_pingroup_count,
        "abstract_segment_count": len(segments.abstract_segments),
        "capacity_violation_count": len(segments.capacity_violations()),
        "batch_summary": plan["summary"],
        "feedthrough_child_net_count": feedthrough["child_net_count"],
        "feedthrough_missing_successors": feedthrough["missing_successors"],
        "singleton_net_count": singleton_net_count,
        "feedthrough_skipped_singleton_net_count": sum(
            not feedthrough["orig_to_children"].get(net.net_id)
            for net in placedb.nets_list
            if len(net.pins) == 1
        ),
        "multi_net_homology_group_count": multi_net_homology_group_count,
    }


def build_hierarchical_case(
    source_block: dict[str, Any],
    source_pingroup: list[list[dict[str, Any]]],
    config: HierarchicalCaseConfig,
) -> dict[str, Any]:
    hierarchy_config = HierarchyConfig(
        case_id=config.case_id,
        target_reuse_rate=config.target_reuse_rate,
        base_seed=config.base_seed,
        seed_namespace=config.seed_namespace,
        reuse_cluster_strategy=config.reuse_cluster_strategy,
    )
    hierarchy = compose_hierarchy(source_block, hierarchy_config)
    net_config = NetSynthesisConfig(
        case_id=config.case_id,
        target_pingroup_count=config.target_pingroup_count,
        source_sample_id=config.source_sample_id,
        base_seed=config.base_seed,
        seed_namespace=config.seed_namespace,
        reuse_template_fraction=config.reuse_template_fraction,
        mcts_tree_depth_profile=config.mcts_tree_depth_profile,
        tree_profile_group_budget_fraction=config.tree_profile_group_budget_fraction,
    )
    synthesized = HierarchicalNetSynthesizer(
        hierarchy["block"], source_pingroup, net_config
    ).synthesize()
    capacity_profile = config.capacity_profile
    if capacity_profile == "auto":
        capacity_profile = "day6_30k" if config.target_pingroup_count >= 30_000 else "day5"
    capacity_config = (
        CapacityCalibrationConfig(
            policy="floorset_first",
            seed=net_config.width_seed,
            small_fraction=0.80,
            normal_fraction=0.19,
            large_fraction=0.01,
        )
        if capacity_profile == "day6_30k"
        else CapacityCalibrationConfig(policy="floorset_first", seed=net_config.width_seed)
    )
    calibrated = calibrate_case(
        hierarchy["block"],
        synthesized["pingroup"],
        capacity_config,
    )
    lineage_nets = synthesized["lineage_nets"]
    for record, net in zip(lineage_nets, calibrated["pingroup"]):
        record["width_provenance"] = sorted(
            {str(pin.get("width_provenance", "")) for pin in net}
        )
        record["width"] = max(float(pin["width"]) for pin in net)
        record["capacity_pressure"] = max(
            float(pin.get("capacity_pressure", 0.0)) for pin in net
        )

    hierarchy_validation = validate_hierarchy(calibrated["block"], hierarchy_config)
    net_validation = validate_hierarchical_nets(
        calibrated["pingroup"], lineage_nets, config.target_pingroup_count
    )
    reuse_connectivity_report = build_reuse_connectivity_report(
        calibrated["pingroup"], lineage_nets
    )
    allowed_singletons = {
        int(record["derived_net_index"])
        for record in lineage_nets
        if record.get("net_kind") == "dangling_singleton"
    }
    artifact_validation = validate_artifacts(
        calibrated["block"],
        calibrated["pingroup"],
        allowed_singleton_net_indices=allowed_singletons,
    )
    preliminary_errors: list[dict[str, str]] = []
    for name, report in (
        ("hierarchy", hierarchy_validation),
        ("net", net_validation),
        ("artifact", artifact_validation),
    ):
        preliminary_errors.extend(
            {"code": f"{name}:{error['code']}", "detail": str(error.get("detail", ""))}
            for error in report["errors"]
        )
    packing = calibrated["report"]["packing"]
    if packing["rejected_group_count"]:
        preliminary_errors.append(
            {
                "code": "capacity:rejected_groups",
                "detail": str(packing["rejected_group_count"]),
            }
        )
    if packing["hard_overflow_count"]:
        preliminary_errors.append(
            {
                "code": "capacity:hard_overflow",
                "detail": str(packing["hard_overflow_count"]),
            }
        )
    if calibrated["report"]["small_group_ratio"] < 0.70:
        preliminary_errors.append(
            {
                "code": "capacity:small_ratio",
                "detail": str(calibrated["report"]["small_group_ratio"]),
            }
        )
    validation = {
        "valid": not preliminary_errors,
        "error_count": len(preliminary_errors),
        "errors": preliminary_errors,
        "hierarchy": hierarchy_validation,
        "nets": net_validation,
        "artifacts": artifact_validation,
        "capacity": {
            "small_group_ratio": calibrated["report"]["small_group_ratio"],
            "rejected_group_count": packing["rejected_group_count"],
            "hard_overflow_count": packing["hard_overflow_count"],
            "admission_scale": calibrated["report"]["admission_scale"],
            "packing_utilization_limit": packing["packing_utilization_limit"],
        },
    }
    lineage = {
        "suite_version": HIERARCHICAL_SUITE_VERSION,
        "source_sample_id": config.source_sample_id,
        "modules": hierarchy["lineage_modules"],
        "nets": lineage_nets,
        "counts": {
            "modules": len(hierarchy["lineage_modules"]),
            "nets": len(lineage_nets),
            "fully_synthetic_nets": sum(record["fully_synthetic"] for record in lineage_nets),
        },
    }
    return {
        "block": calibrated["block"],
        "pingroup": calibrated["pingroup"],
        "hierarchy_report": hierarchy["report"],
        "net_distribution_report": synthesized["report"],
        "reuse_connectivity_report": reuse_connectivity_report,
        "cross_net_homology_report": synthesized["cross_net_homology_report"],
        "capacity_report": calibrated["report"],
        "validation": validation,
        "lineage": lineage,
        "phase_seeds": {
            **hierarchy["report"]["phase_seeds"],
            **synthesized["report"]["phase_seeds"],
        },
    }


def write_hierarchical_case(
    source_dir: str | Path,
    destination: str | Path,
    config: HierarchicalCaseConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    tracemalloc.start()
    source_dir, destination = Path(source_dir), Path(destination)
    load_started = time.perf_counter()
    source_block = json.loads((source_dir / "block.json").read_text(encoding="utf-8"))
    source_pingroup = json.loads((source_dir / "pingroup.json").read_text(encoding="utf-8"))
    load_seconds = time.perf_counter() - load_started
    build_started = time.perf_counter()
    result = build_hierarchical_case(source_block, source_pingroup, config)
    build_seconds = time.perf_counter() - build_started
    if not result["validation"]["valid"]:
        raise ValueError(f"hierarchical case failed validation: {result['validation']['errors'][:5]!r}")
    destination.mkdir(parents=True, exist_ok=True)
    _json_dump(destination / "block.json", result["block"])
    _json_dump(destination / "pingroup.json", result["pingroup"])
    _json_dump(destination / "hierarchy_report.json", result["hierarchy_report"])
    _json_dump(destination / "net_distribution_report.json", result["net_distribution_report"])
    _json_dump(destination / "reuse_connectivity_report.json", result["reuse_connectivity_report"])
    _json_dump(destination / "cross_net_homology_report.json", result["cross_net_homology_report"])
    _json_dump(destination / "capacity_report.json", result["capacity_report"])
    _json_dump(destination / "lineage.json", result["lineage"])
    provenance = {
        "derived_artifact_label": "FloorSet-derived hierarchical synthetic PinAssign benchmark",
        "source_repository": FLOORSET_REPOSITORY,
        "source_data_license": FLOORSET_DATA_LICENSE,
        "source_sample_id": config.source_sample_id,
        "source_commit": config.source_commit,
        "source_artifact": str(source_dir),
        "suite_version": HIERARCHICAL_SUITE_VERSION,
        "fully_synthetic_topology": True,
    }
    _json_dump(destination / "provenance.json", provenance)
    manifest = {
        "suite_version": HIERARCHICAL_SUITE_VERSION,
        "config": asdict(config),
        "phase_seeds": result["phase_seeds"],
        "files": {
            "block": "block.json",
            "pingroup": "pingroup.json",
            "lineage": "lineage.json",
            "provenance": "provenance.json",
            "validation": "validation.json",
            "capacity_report": "capacity_report.json",
            "hierarchy_report": "hierarchy_report.json",
            "net_distribution_report": "net_distribution_report.json",
            "reuse_connectivity_report": "reuse_connectivity_report.json",
            "cross_net_homology_report": "cross_net_homology_report.json",
            "case_statistics": "case_statistics.json",
            "mcts_tree_depth_report": "mcts_tree_depth_report.json",
            "batch_vs_runtime_tree_report": "batch_vs_runtime_tree_report.json",
            "tree_depth_profile_comparison": "tree_depth_profile_comparison.json",
        },
        "policies": {
            "module_count": "fixed 12 parents + 228 leaves = 240 non-root instances",
            "reuse": "instance-expanded aligned/dangling/remapped homology templates; no size changes",
            "topology": "independent synthetic hierarchical successor DAGs",
            "width": "existing FloorSet-first capacity-pressure calibrator",
            "connectivity_weight": "ordering prior only; not physical pin width",
        },
    }
    if not config.mcts_tree_depth_profile:
        for key in (
            "mcts_tree_depth_report",
            "batch_vs_runtime_tree_report",
            "tree_depth_profile_comparison",
        ):
            manifest["files"].pop(key, None)
    _json_dump(destination / "manifest.json", manifest)

    integration_started = time.perf_counter()
    integration = _integration_report(destination, config.target_pingroup_count)
    integration_seconds = time.perf_counter() - integration_started
    integration_errors: list[dict[str, str]] = []
    if integration["homology_pingroup_count"] != config.target_pingroup_count:
        integration_errors.append(
            {"code": "integration:pingroup_count", "detail": str(integration["homology_pingroup_count"])}
        )
    if integration["feedthrough_missing_successors"]:
        integration_errors.append(
            {
                "code": "integration:missing_successors",
                "detail": str(integration["feedthrough_missing_successors"]),
            }
        )
    if integration["feedthrough_child_net_count"] <= 0:
        integration_errors.append({"code": "integration:no_child_nets", "detail": "0"})
    if integration["capacity_violation_count"]:
        integration_errors.append(
            {
                "code": "integration:capacity_violation",
                "detail": str(integration["capacity_violation_count"]),
            }
        )
    expected_singletons = result["reuse_connectivity_report"]["expected_singleton_net_count"]
    if integration["singleton_net_count"] != expected_singletons:
        integration_errors.append(
            {
                "code": "integration:singleton_count",
                "detail": f"{integration['singleton_net_count']}!={expected_singletons}",
            }
        )
    if integration["feedthrough_skipped_singleton_net_count"] != expected_singletons:
        integration_errors.append(
            {
                "code": "integration:singleton_feedthrough_skip",
                "detail": str(integration["feedthrough_skipped_singleton_net_count"]),
            }
        )
    result["validation"]["integration"] = integration
    result["validation"]["errors"].extend(integration_errors)
    result["validation"]["error_count"] = len(result["validation"]["errors"])
    result["validation"]["valid"] = not result["validation"]["errors"]
    if config.mcts_tree_depth_profile and result["validation"]["valid"]:
        depth_report = analyze_mcts_tree_depths(
            destination / "block.json",
            destination / "pingroup.json",
            lineage_json=destination / "lineage.json",
            capacity_report_json=destination / "capacity_report.json",
        )
        _json_dump(destination / "mcts_tree_depth_report.json", depth_report)
        cross_report = {
            "generation": result["cross_net_homology_report"],
            "runtime_tree_records": depth_report["cross_net_tree_records"],
            "multi_net_homology_group_count": depth_report["multi_net_homology_group_count"],
            "capacity_deduction": depth_report["capacity_deduction"],
        }
        _json_dump(destination / "cross_net_homology_report.json", cross_report)
        _json_dump(
            destination / "batch_vs_runtime_tree_report.json",
            {
                "static_raw_depth": depth_report["static_raw_depth"],
                "ordered_raw_depth": depth_report["ordered_raw_depth"],
                "effective_mcts_depth": depth_report["effective_mcts_depth"],
                "batch_component_depth": depth_report["batch_component_depth"],
                "batch_depth": depth_report["batch_depth"],
                "batch_planner": depth_report["batch_planner"],
            },
        )
        _json_dump(
            destination / "tree_depth_profile_comparison.json",
            {
                "excel_conditional_sample": {
                    "tree_count": 4327,
                    "p50": 3,
                    "p90": 10,
                    "p95": 15,
                    "sampled_max": 22,
                    "scope": "partial small/medium-tree sample only; not a full-case distribution",
                },
                "generated_case": depth_report["effective_mcts_depth"],
                "acceptance": depth_report["acceptance"],
            },
        )
        if not depth_report["acceptance_passed"]:
            result["validation"]["errors"].append(
                {
                    "code": "mcts_tree_depth:acceptance",
                    "detail": json.dumps(depth_report["acceptance"], sort_keys=True),
                }
            )
            result["validation"]["error_count"] = len(result["validation"]["errors"])
            result["validation"]["valid"] = False
    _json_dump(destination / "validation.json", result["validation"])
    if not result["validation"]["valid"]:
        raise ValueError(f"hierarchical integration failed: {integration_errors[:5]!r}")
    _, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings = {
        "source_load_seconds": load_seconds,
        "build_and_calibration_seconds": build_seconds,
        "integration_seconds": integration_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    case_statistics = {
        "case_id": config.case_id,
        "target_pingroup_count": config.target_pingroup_count,
        "hierarchy": result["hierarchy_report"],
        "topology": result["net_distribution_report"],
        "reuse_connectivity": result["reuse_connectivity_report"],
        "capacity": {
            "small_group_ratio": result["capacity_report"]["small_group_ratio"],
            "bucket_group_counts": result["capacity_report"]["bucket_group_counts"],
            "width_summary": result["capacity_report"]["width_summary"],
            "pressure_summary": result["capacity_report"]["pressure_summary"],
            "admission_scale": result["capacity_report"]["admission_scale"],
            "packing": result["capacity_report"]["packing"],
        },
        "integration": integration,
        "timings": timings,
        "peak_python_allocation_bytes": peak_python_bytes,
    }
    _json_dump(destination / "case_statistics.json", case_statistics)
    hashes = {
        path.name: _sha256(path)
        for path in sorted(destination.glob("*.json"))
        if path.name != "case_statistics.json"
    }
    case_statistics["artifact_bytes"] = {
        path.name: path.stat().st_size for path in sorted(destination.glob("*.json"))
        if path.name != "case_statistics.json"
    }
    case_statistics["artifact_sha256"] = hashes
    _json_dump(destination / "case_statistics.json", case_statistics)
    hashes["case_statistics.json"] = _sha256(destination / "case_statistics.json")
    return {
        "output_dir": str(destination),
        "case_id": config.case_id,
        "target_pingroup_count": config.target_pingroup_count,
        "module_instance_count": result["hierarchy_report"]["nonroot_instance_count"],
        "reuse_instance_ratio": result["hierarchy_report"]["reuse_instance_ratio"],
        "net_count": result["net_distribution_report"]["net_count"],
        "pin_count": result["net_distribution_report"]["pin_count"],
        "pingroup_count": integration["homology_pingroup_count"],
        "small_group_ratio": result["capacity_report"]["small_group_ratio"],
        "capacity_rejected_group_count": result["capacity_report"]["packing"]["rejected_group_count"],
        "hard_overflow_count": result["capacity_report"]["packing"]["hard_overflow_count"],
        "feedthrough_child_net_count": integration["feedthrough_child_net_count"],
        "artifact_sha256": hashes,
        "reuse_connectivity": result["reuse_connectivity_report"],
        "timings": timings,
        "peak_python_allocation_bytes": peak_python_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--target-pingroups", required=True, type=int)
    parser.add_argument("--reuse-rate", required=True, type=float, choices=(0.60, 0.95))
    parser.add_argument("--source-sample-id", default="floorset_config21_1")
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seed-namespace")
    parser.add_argument(
        "--mcts-tree-depth-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reuse-cluster-strategy",
        choices=("pairs", "mixed2to10"),
        default="pairs",
    )
    args = parser.parse_args()
    result = write_hierarchical_case(
        args.source_dir,
        args.output_dir,
        HierarchicalCaseConfig(
            case_id=args.case_id,
            target_pingroup_count=args.target_pingroups,
            target_reuse_rate=args.reuse_rate,
            source_sample_id=args.source_sample_id,
            source_commit=args.source_commit,
            base_seed=args.seed,
            seed_namespace=args.seed_namespace,
            mcts_tree_depth_profile=args.mcts_tree_depth_profile,
            reuse_cluster_strategy=args.reuse_cluster_strategy,
            reuse_template_fraction=0.01 if args.mcts_tree_depth_profile else 0.05,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
