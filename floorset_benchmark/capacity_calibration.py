"""Capacity-aware PinGroup width calibration and deterministic comparison tools.

FloorSet connectivity weights are routing-cost weights, not physical pin widths.
This module therefore uses their order as a prior and maps it to a dimensionless
pressure relative to the real abstract-segment capacities of this project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from geometry_utils import canonical_vertex_mapping, segment_length
from segment_subdivision import interpolate_edge, subdivision_specs


CALIBRATOR_VERSION = "floorset-capacity-pressure-v0.1"


@dataclass(frozen=True)
class CapacityCalibrationConfig:
    """Width-pressure policy for one deterministic calibration pass."""

    policy: str = "floorset_first"
    seed: int = 7
    small_fraction: float = 0.75
    normal_fraction: float = 0.20
    large_fraction: float = 0.05
    small_range: tuple[float, float] = (0.002, 0.02)
    normal_range: tuple[float, float] = (0.020000001, 0.10)
    large_range: tuple[float, float] = (0.100000001, 0.35)
    stress_fraction: float = 0.005
    stress_range: tuple[float, float] = (0.350000001, 0.60)
    enable_stress: bool = False
    baseline_pressure: float = 0.01
    max_segment_length: float | None = None
    packing_utilization_limit: float = 0.90

    def __post_init__(self) -> None:
        if self.policy not in {"baseline", "floorset_first"}:
            raise ValueError("policy must be 'baseline' or 'floorset_first'")
        if not math.isclose(
            self.small_fraction + self.normal_fraction + self.large_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("small/normal/large fractions must sum to 1")
        if self.small_fraction < 0.70:
            raise ValueError("small_fraction must be at least 0.70")
        if not 0.0 <= self.stress_fraction <= 0.005:
            raise ValueError("stress_fraction must be in [0, 0.005]")
        if not self.small_range[0] <= self.baseline_pressure <= self.small_range[1]:
            raise ValueError("baseline_pressure must remain in the small range")
        if not 0.0 < self.packing_utilization_limit <= 1.0:
            raise ValueError("packing_utilization_limit must be in (0, 1]")


def _walk_modules(root: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield root
    for child in root.get("children", []):
        yield from _walk_modules(child)


def module_segment_capacities(
    block: dict[str, Any],
    max_segment_length: float | None = None,
) -> dict[str, list[float]]:
    """Reproduce SegmentManager's abstract capacities without temporary files."""
    result: dict[str, list[float]] = {}
    for module in _walk_modules(block):
        module_name = str(module.get("module_name", ""))
        vertices = module.get("vertex", [])
        if not module_name or not vertices or module_name in result:
            continue
        _, canonical = canonical_vertex_mapping(vertices, int(module.get("direction", 0)))
        capacities: list[float] = []
        for spec in subdivision_specs(canonical, max_segment_length):
            edge_start = canonical[spec.edge_id]
            edge_end = canonical[(spec.edge_id + 1) % len(canonical)]
            start = interpolate_edge(edge_start, edge_end, spec.t_start)
            end = interpolate_edge(edge_start, edge_end, spec.t_end)
            capacities.append(float(segment_length(start, end)))
        result[module_name] = capacities
    return result


def _group_key(pin: dict[str, Any]) -> str:
    return f"{pin.get('parent_module', '')}.{pin.get('pingroup_name', '')}"


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            self.parent[right_root] = left_root


def _stable_unit_interval(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def connectivity_weight_audit(record: dict[str, np.ndarray]) -> dict[str, Any]:
    """Summarize valid FloorSet b2b/p2b third-column connectivity weights."""
    result: dict[str, Any] = {}
    for name in ("b2b_connectivity", "p2b_connectivity"):
        array = np.asarray(record[name])
        valid = array[np.all(array[:, :2] >= 0, axis=1)] if array.size else array.reshape((0, array.shape[-1]))
        weights = valid[:, 2] if valid.shape[1] > 2 else np.ones(valid.shape[0], dtype=np.float64)
        result[name] = _quantiles(float(value) for value in weights)
    return result


def _component_model(
    pingroup: list[list[dict[str, Any]]],
    capacities: dict[str, list[float]],
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    dsu = _DisjointSet()
    group_pins: dict[str, list[dict[str, Any]]] = {}
    for net in pingroup:
        keys = sorted({_group_key(pin) for pin in net})
        for key in keys:
            dsu.add(key)
        for key in keys[1:]:
            dsu.union(keys[0], key)
        for pin in net:
            group_pins.setdefault(_group_key(pin), []).append(pin)

    component_groups: dict[str, list[str]] = {}
    for key in sorted(group_pins):
        component_groups.setdefault(dsu.find(key), []).append(key)

    observed_weights = sorted(
        float(pin["source_connectivity_weight"])
        for pins in group_pins.values()
        for pin in pins
        if pin.get("source_connectivity_weight") is not None
    )
    components: dict[str, dict[str, Any]] = {}
    group_to_component: dict[str, str] = {}
    for root, groups in sorted(component_groups.items()):
        module_names = sorted({key.rsplit(".", 1)[0] for key in groups})
        missing_modules = [name for name in module_names if not capacities.get(name)]
        if missing_modules:
            raise ValueError(f"No candidate segments for module types: {missing_modules[:5]!r}")
        c_edge = min(max(capacities[name]) for name in module_names)
        weights = [
            float(pin["source_connectivity_weight"])
            for group in groups
            for pin in group_pins[group]
            if pin.get("source_connectivity_weight") is not None
        ]
        if weights:
            source_weight = max(weights)
            pin_kinds = {
                str(pin.get("source_connectivity_kind", ""))
                for group in groups
                for pin in group_pins[group]
                if pin.get("source_connectivity_weight") is not None
            }
            if any(kind in {"b2b_weight", "p2b_weight", "floorset_empirical_weight_sample"} for kind in pin_kinds):
                source_kind = "floorset_connectivity_weight"
            else:
                source_kind = "synthetic_topology_weight_prior"
        elif observed_weights:
            index = min(int(_stable_unit_interval(seed, root) * len(observed_weights)), len(observed_weights) - 1)
            source_weight = observed_weights[index]
            source_kind = "synthetic_empirical_weight_sample"
        else:
            source_weight = _stable_unit_interval(seed, root)
            source_kind = "legacy_deterministic_rank_proxy"
        components[root] = {
            "groups": groups,
            "module_names": module_names,
            "c_edge": c_edge,
            "source_weight": source_weight,
            "source_kind": source_kind,
        }
        for group in groups:
            group_to_component[group] = root
    return components, group_to_component


def _packing_report(
    pingroup: list[list[dict[str, Any]]],
    capacities: dict[str, list[float]],
    utilization_limit: float = 1.0,
) -> dict[str, Any]:
    group_width: dict[str, float] = {}
    group_module: dict[str, str] = {}
    for net in pingroup:
        for pin in net:
            key = _group_key(pin)
            width = float(pin["width"])
            previous = group_width.setdefault(key, width)
            if not math.isclose(previous, width, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Homology group {key} has inconsistent widths")
            group_module[key] = str(pin["parent_module"])

    by_module: dict[str, list[str]] = {}
    for key, module_name in group_module.items():
        by_module.setdefault(module_name, []).append(key)

    assignments: dict[str, str] = {}
    rejected: list[str] = []
    module_reports: dict[str, dict[str, Any]] = {}
    initial_candidate_total = 0
    initial_capacity_pruned = 0
    for module_name, groups in sorted(by_module.items()):
        bins = list(capacities.get(module_name, []))
        remaining = [capacity * utilization_limit for capacity in bins]
        candidate_counts: dict[str, int] = {}
        for key in groups:
            count = sum(capacity + 1e-9 >= group_width[key] for capacity in bins)
            candidate_counts[key] = count
            initial_candidate_total += count
            initial_capacity_pruned += len(bins) - count
        assigned_count = 0
        for key in sorted(groups, key=lambda item: (-group_width[item], item)):
            width = group_width[key]
            feasible = [index for index, capacity in enumerate(remaining) if capacity + 1e-9 >= width]
            if not feasible:
                rejected.append(key)
                continue
            index = min(feasible, key=lambda item: (remaining[item] - width, item))
            remaining[index] -= width
            assignments[key] = f"{module_name}:S{index}"
            assigned_count += 1
        demand = sum(group_width[key] for key in groups)
        total_capacity = sum(bins)
        module_reports[module_name] = {
            "group_count": len(groups),
            "assigned_group_count": assigned_count,
            "rejected_group_count": len(groups) - assigned_count,
            "demand": demand,
            "capacity": total_capacity,
            "utilization": demand / total_capacity if total_capacity else None,
            "max_group_width": max((group_width[key] for key in groups), default=0.0),
            "initial_candidate_count": sum(candidate_counts.values()),
            "initial_capacity_pruned_count": sum(len(bins) - value for value in candidate_counts.values()),
        }
    return {
        "group_count": len(group_width),
        "assignment_count": len(assignments),
        "rejected_group_count": len(rejected),
        "rejected_groups": sorted(rejected),
        "initial_candidate_count": initial_candidate_total,
        "initial_capacity_pruned_count": initial_capacity_pruned,
        "hard_overflow_count": 0,
        "packing_utilization_limit": utilization_limit,
        "assignments": assignments,
        "module_types": module_reports,
    }


def calibrate_case(
    block: dict[str, Any],
    pingroup: list[list[dict[str, Any]]],
    config: CapacityCalibrationConfig,
) -> dict[str, Any]:
    """Return calibrated copies of block/pingroup plus a machine-readable report."""
    calibrated_block = deepcopy(block)
    calibrated_pingroup = deepcopy(pingroup)
    capacities = module_segment_capacities(calibrated_block, config.max_segment_length)
    components, group_to_component = _component_model(calibrated_pingroup, capacities, config.seed)

    ordered = sorted(
        components,
        key=lambda root: (components[root]["source_weight"], root),
    )
    total_group_count = sum(len(components[root]["groups"]) for root in ordered)
    ranked_pressure: dict[str, tuple[float, str]] = {}
    consumed_groups = 0
    normal_end = config.small_fraction + config.normal_fraction
    for root in ordered:
        item_group_count = len(components[root]["groups"])
        start_quantile = consumed_groups / total_group_count if total_group_count else 0.0
        midpoint_quantile = (consumed_groups + item_group_count / 2.0) / total_group_count if total_group_count else 0.0
        if start_quantile < config.small_fraction:
            low, high = config.small_range
            fraction = min(midpoint_quantile / config.small_fraction, 1.0)
            ranked_pressure[root] = (low + fraction * (high - low), "small")
        elif start_quantile < normal_end:
            low, high = config.normal_range
            fraction = min((midpoint_quantile - config.small_fraction) / config.normal_fraction, 1.0)
            ranked_pressure[root] = (low + fraction * (high - low), "normal")
        else:
            low, high = config.large_range
            fraction = min((midpoint_quantile - normal_end) / config.large_fraction, 1.0)
            ranked_pressure[root] = (low + fraction * (high - low), "large")
        consumed_groups += item_group_count

    stress_roots: set[str] = set()
    stress_group_budget = (
        int(math.floor(total_group_count * config.stress_fraction))
        if config.enable_stress else 0
    )
    stress_groups = 0
    for root in reversed(ordered):
        component_group_count = len(components[root]["groups"])
        if stress_groups + component_group_count > stress_group_budget:
            continue
        stress_roots.add(root)
        stress_groups += component_group_count
        if stress_groups == stress_group_budget:
            break
    component_reports: dict[str, dict[str, Any]] = {}
    for root, item in sorted(components.items()):
        if config.policy == "baseline":
            pressure, bucket = config.baseline_pressure, "small"
            provenance = "all_small_baseline"
        else:
            pressure, bucket = ranked_pressure[root]
            provenance = item["source_kind"]
        if root in stress_roots:
            stress_order = sorted(stress_roots, key=lambda value: (components[value]["source_weight"], value))
            position = stress_order.index(root)
            fraction = (position + 1) / max(len(stress_order), 1)
            low, high = config.stress_range
            pressure = low + fraction * (high - low)
            bucket = "stress"
            provenance = "synthetic_capacity_stress"
        width = pressure * float(item["c_edge"])
        component_reports[root] = {
            **item,
            "pressure": pressure,
            "width": width,
            "bucket": bucket,
            "width_provenance": provenance,
        }

    target_pressure = {
        root: float(item["pressure"])
        for root, item in component_reports.items()
    }

    def materialize(admission_scale: float) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        for root, component in component_reports.items():
            bucket = str(component["bucket"])
            if config.policy == "floorset_first" and bucket != "stress":
                low = {
                    "small": config.small_range[0],
                    "normal": config.normal_range[0],
                    "large": config.large_range[0],
                }[bucket]
                component["pressure"] = low + admission_scale * (target_pressure[root] - low)
                component["width"] = float(component["pressure"]) * float(component["c_edge"])
        reports: dict[str, dict[str, Any]] = {}
        for net in calibrated_pingroup:
            for pin in net:
                key = _group_key(pin)
                component = component_reports[group_to_component[key]]
                pin["width"] = float(component["width"])
                pin["capacity_pressure"] = float(component["pressure"])
                pin["width_provenance"] = component["width_provenance"]
                reports[key] = {
                    "component": group_to_component[key],
                    "module_name": str(pin["parent_module"]),
                    "width": float(component["width"]),
                    "pressure": float(component["pressure"]),
                    "bucket": component["bucket"],
                    "source_weight": float(component["source_weight"]),
                    "source_kind": component["source_kind"],
                }
        return reports, _packing_report(
            calibrated_pingroup,
            capacities,
            utilization_limit=config.packing_utilization_limit,
        )

    admission_scale = 1.0
    group_reports, packing = materialize(admission_scale)
    admission_adjusted = False
    if (
        config.policy == "floorset_first"
        and not stress_roots
        and packing["rejected_group_count"] > 0
    ):
        minimum_reports, minimum_packing = materialize(0.0)
        if minimum_packing["rejected_group_count"] == 0:
            low, high = 0.0, 1.0
            for _ in range(36):
                middle = (low + high) / 2.0
                _, trial = materialize(middle)
                if trial["rejected_group_count"] == 0:
                    low = middle
                else:
                    high = middle
            admission_scale = low
            group_reports, packing = materialize(admission_scale)
            admission_adjusted = True
        else:
            group_reports, packing = minimum_reports, minimum_packing
            admission_scale = 0.0
            admission_adjusted = True
    buckets: dict[str, int] = {}
    for item in group_reports.values():
        buckets[item["bucket"]] = buckets.get(item["bucket"], 0) + 1
    group_count = len(group_reports)
    report = {
        "calibrator_version": CALIBRATOR_VERSION,
        "config": asdict(config),
        "floorSet_weight_semantics": "connectivity cost prior; not a physical pin width",
        "observed_source_weight_pin_count": sum(
            pin.get("source_connectivity_weight") is not None
            for net in pingroup
            for pin in net
        ),
        "legacy_rank_proxy_used": any(
            item["source_kind"] == "legacy_deterministic_rank_proxy"
            for item in component_reports.values()
        ),
        "component_count": len(component_reports),
        "group_count": group_count,
        "bucket_group_counts": buckets,
        "small_group_ratio": buckets.get("small", 0) / group_count if group_count else 0.0,
        "width_summary": _quantiles(item["width"] for item in group_reports.values()),
        "pressure_summary": _quantiles(item["pressure"] for item in group_reports.values()),
        "source_weight_summary": _quantiles(item["source_weight"] for item in component_reports.values()),
        "source_weight_to_pressure_monotonic": all(
            left[1] <= right[1] + 1e-12
            for left, right in zip(
                sorted((item["source_weight"], item["pressure"]) for item in component_reports.values()),
                sorted((item["source_weight"], item["pressure"]) for item in component_reports.values())[1:],
            )
        ) if config.policy == "floorset_first" and not config.enable_stress else None,
        "stress_enabled": bool(stress_roots),
        "stress_component_count": len(stress_roots),
        "stress_group_count": stress_groups,
        "stress_group_ratio": stress_groups / group_count if group_count else 0.0,
        "admission_adjusted": admission_adjusted,
        "admission_scale": admission_scale,
        "minimum_profile_feasible": packing["rejected_group_count"] == 0,
        "packing": packing,
        "groups": group_reports,
    }
    return {"block": calibrated_block, "pingroup": calibrated_pingroup, "report": report}


def compare_calibrations(
    block: dict[str, Any],
    pingroup: list[list[dict[str, Any]]],
    *,
    seed: int = 7,
    auto_stress: bool = True,
) -> dict[str, Any]:
    """Build same-topology baseline/FloorSet-first cases and compare capacity effects."""
    baseline = calibrate_case(block, pingroup, CapacityCalibrationConfig(policy="baseline", seed=seed))
    floorset = calibrate_case(block, pingroup, CapacityCalibrationConfig(policy="floorset_first", seed=seed))
    left = baseline["report"]["packing"]
    right = floorset["report"]["packing"]
    all_groups = set(left["assignments"]) | set(right["assignments"])
    assignment_changes = sum(left["assignments"].get(key) != right["assignments"].get(key) for key in all_groups)
    effect = {
        "candidate_count_delta": right["initial_candidate_count"] - left["initial_candidate_count"],
        "capacity_pruned_delta": right["initial_capacity_pruned_count"] - left["initial_capacity_pruned_count"],
        "rejected_group_delta": right["rejected_group_count"] - left["rejected_group_count"],
        "packing_assignment_changed_count": assignment_changes,
    }
    effect["observable_capacity_effect"] = any(value != 0 for value in effect.values())
    stress = None
    if auto_stress and not effect["observable_capacity_effect"]:
        stress = calibrate_case(
            block,
            pingroup,
            CapacityCalibrationConfig(policy="floorset_first", seed=seed, enable_stress=True),
        )
    return {"baseline": baseline, "floorset_first": floorset, "effect": effect, "stress": stress}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_comparison(
    block_path: str | Path,
    pingroup_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 7,
    auto_stress: bool = True,
) -> dict[str, Any]:
    block = json.loads(Path(block_path).read_text(encoding="utf-8"))
    pingroup = json.loads(Path(pingroup_path).read_text(encoding="utf-8"))
    result = compare_calibrations(block, pingroup, seed=seed, auto_stress=auto_stress)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("baseline", "floorset_first"):
        case_dir = destination / name
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_json(case_dir / "block.json", result[name]["block"])
        _write_json(case_dir / "pingroup.json", result[name]["pingroup"])
        _write_json(case_dir / "capacity_report.json", result[name]["report"])
        _write_json(
            case_dir / "manifest.json",
            {
                "calibrator_version": CALIBRATOR_VERSION,
                "mode": name,
                "seed": seed,
                "same_topology_comparison": True,
                "source_block": str(Path(block_path)),
                "source_pingroup": str(Path(pingroup_path)),
                "capacity_report_file": "capacity_report.json",
                "lineage_file": "lineage.json",
                "synthetic_capacity_stress": False,
                "source_weight_semantics": "connectivity cost prior; not physical pin width",
            },
        )
        _write_json(
            case_dir / "lineage.json",
            {
                "calibrator_version": CALIBRATOR_VERSION,
                "source_block": str(Path(block_path)),
                "source_pingroup": str(Path(pingroup_path)),
                "width_policy": name,
                "group_count": result[name]["report"]["group_count"],
                "legacy_rank_proxy_used": result[name]["report"]["legacy_rank_proxy_used"],
                "synthetic_capacity_stress": False,
            },
        )
    if result["stress"] is not None:
        case_dir = destination / "synthetic_capacity_stress"
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_json(case_dir / "block.json", result["stress"]["block"])
        _write_json(case_dir / "pingroup.json", result["stress"]["pingroup"])
        _write_json(case_dir / "capacity_report.json", result["stress"]["report"])
        _write_json(
            case_dir / "manifest.json",
            {
                "calibrator_version": CALIBRATOR_VERSION,
                "mode": "synthetic_capacity_stress",
                "seed": seed,
                "same_topology_comparison": True,
                "source_block": str(Path(block_path)),
                "source_pingroup": str(Path(pingroup_path)),
                "capacity_report_file": "capacity_report.json",
                "lineage_file": "lineage.json",
                "synthetic_capacity_stress": True,
                "source_weight_semantics": "connectivity cost prior; not physical pin width",
            },
        )
        _write_json(
            case_dir / "lineage.json",
            {
                "calibrator_version": CALIBRATOR_VERSION,
                "source_block": str(Path(block_path)),
                "source_pingroup": str(Path(pingroup_path)),
                "width_policy": "floorset_first",
                "group_count": result["stress"]["report"]["group_count"],
                "synthetic_capacity_stress": True,
                "stress_group_count": result["stress"]["report"]["stress_group_count"],
            },
        )
    summary = {
        "calibrator_version": CALIBRATOR_VERSION,
        "seed": seed,
        "same_topology": True,
        "effect": result["effect"],
        "stress_enabled": result["stress"] is not None,
        "baseline_report": "baseline/capacity_report.json",
        "floorset_first_report": "floorset_first/capacity_report.json",
    }
    _write_json(destination / "comparison.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True, type=Path)
    parser.add_argument("--pingroup", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-auto-stress", action="store_true")
    args = parser.parse_args()
    result = write_comparison(
        args.block,
        args.pingroup,
        args.output_dir,
        seed=args.seed,
        auto_stress=not args.no_auto_stress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
