"""Deterministic small-scale composer for FloorSet-derived project JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from geometry_utils import canonical_vertex_mapping

from .capacity_calibration import CapacityCalibrationConfig, calibrate_case
from .single_converter import CONVERTER_VERSION, validate_artifacts


SCALER_VERSION = "floorset-scaler-v0.1"


@dataclass(frozen=True)
class ScaleConfig:
    target_module_count: int
    target_reuse_rate: float
    base_seed: int = 7
    case_id: str = "scale-smoke"
    source_sample_id: str = "unknown"
    spacing: float = 1.0
    source_commit: str = "unknown"
    direction_strategy: str = "cycle8"
    target_pingroup_count: int | None = None
    width_policy: str = "preserve"

    def __post_init__(self) -> None:
        if self.target_module_count < 1:
            raise ValueError("target_module_count must be positive")
        if not 0.0 <= self.target_reuse_rate <= 1.0:
            raise ValueError("target_reuse_rate must be between 0 and 1")
        if self.spacing < 0:
            raise ValueError("spacing must be non-negative")
        if self.direction_strategy not in {"cycle8", "r0-only"}:
            raise ValueError("direction_strategy must be cycle8 or r0-only")
        if self.target_pingroup_count is not None and self.target_pingroup_count < 1:
            raise ValueError("target_pingroup_count must be positive when provided")
        if self.width_policy not in {"preserve", "baseline", "floorset_first"}:
            raise ValueError("width_policy must be preserve, baseline, or floorset_first")

    @property
    def derived_seed(self) -> int:
        material = (
            f"{self.base_seed}|{self.case_id}|{self.source_sample_id}|{SCALER_VERSION}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _polygon_bounds(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _translate_polygon(points: list[list[float]], dx: float, dy: float) -> list[list[float]]:
    return [[float(point[0]) + dx, float(point[1]) + dy] for point in points]


def _forward_direction_point(point: list[float], direction: int) -> tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    return {
        0: (x, y), 1: (-x, -y), 2: (-y, x), 3: (y, -x),
        4: (-x, y), 5: (x, -y), 6: (y, x), 7: (-y, -x),
    }[direction]


def _transform_polygon(points: list[list[float]], direction: int) -> list[list[float]]:
    transformed = [_forward_direction_point(point, direction) for point in points]
    min_x = min(point[0] for point in transformed)
    min_y = min(point[1] for point in transformed)
    return [[x - min_x, y - min_y] for x, y in transformed]


def _source_base_polygon(source: dict[str, Any]) -> list[list[float]]:
    """Undo the source placement direction before applying a derived direction."""
    normalized, _ = canonical_vertex_mapping(
        source["vertex"], int(source.get("direction", 0))
    )
    return [[float(x), float(y)] for x, y in normalized]


def _directions_for_positions(positions: list[tuple[int, int]], strategy: str) -> list[int]:
    if strategy == "r0-only":
        return [0] * len(positions)
    return [index % 8 for index in range(len(positions))]


def _instance_list(block: dict[str, Any]) -> list[dict[str, Any]]:
    return [child for child in block.get("children", []) if child.get("name") != "TOP"]


def _module_count(block: dict[str, Any]) -> int:
    return len(_instance_list(block))


def _allocate_template_names(positions: list[tuple[int, int]], reuse_rate: float) -> tuple[list[str], dict[int, int]]:
    """Allocate reuse names only among copies of one source template."""
    target = int(round(len(positions) * reuse_rate))
    if target == 1 or target > len(positions):
        raise ValueError("requested reuse count cannot be represented")
    by_template: dict[int, list[int]] = {}
    for slot, (_, source_index) in enumerate(positions):
        by_template.setdefault(source_index, []).append(slot)
    templates = sorted(by_template)
    # Dynamic programming chooses 0 or 2..c copies for each source template.
    choices: dict[int, list[int]] = {0: []}
    for source_index in templates:
        next_choices: dict[int, list[int]] = {}
        capacity = len(by_template[source_index])
        for total, allocation in choices.items():
            for reused_count in [0, *range(2, capacity + 1)]:
                new_total = total + reused_count
                if new_total <= target and new_total not in next_choices:
                    next_choices[new_total] = allocation + [reused_count]
        choices = next_choices
    allocation = choices.get(target)
    if allocation is None:
        raise ValueError("target reuse rate is not representable by source templates")
    names = [f"SCALE_U{index:06d}" for index in range(len(positions))]
    groups: dict[int, int] = {}
    group_index = 0
    for source_index, reused_count in zip(templates, allocation):
        if reused_count < 2:
            continue
        slots = list(by_template[source_index])
        for slot in slots[:reused_count]:
            names[slot] = f"SCALE_R{group_index:06d}"
        groups[group_index] = source_index
        group_index += 1
    return names, groups


def _rewrite_pin(pin: dict[str, Any], instance_map: dict[str, str], module_map: dict[str, str]) -> dict[str, Any]:
    rewritten = deepcopy(pin)
    old_instance = str(pin["parent_inst"])
    rewritten["parent_inst"] = instance_map[old_instance]
    old_module = str(pin.get("parent_module", ""))
    rewritten["parent_module"] = module_map[old_instance] if old_instance in module_map else old_module
    return rewritten


def _reuse_stats(block: dict[str, Any]) -> dict[str, Any]:
    instances = _instance_list(block)
    counts = Counter(str(instance.get("module_name", "")) for instance in instances)
    reused_instances = sum(count for count in counts.values() if count >= 2)
    total = len(instances)
    return {
        "module_instance_count": total,
        "reused_module_instance_count": reused_instances,
        "reuse_instance_ratio": reused_instances / total if total else 0.0,
        "reused_module_type_count": sum(1 for count in counts.values() if count >= 2),
        "module_type_count": len(counts),
        "reuse_module_type_ratio": (
            sum(1 for count in counts.values() if count >= 2) / len(counts) if counts else 0.0
        ),
    }


def _pingroup_keys(pingroup: list[list[dict[str, Any]]]) -> set[str]:
    return {
        f"{pin.get('parent_module', '')}.{pin.get('pingroup_name', '')}"
        for net in pingroup for pin in net
    }


def _append_synthetic_pingroup_nets(
    block: dict[str, Any], pingroup: list[list[dict[str, Any]]], target: int, lineage_nets: list[dict[str, Any]]
) -> tuple[int, int]:
    """Add deterministic two-pin nets until the homology-group count is exact."""
    current = _pingroup_keys(pingroup)
    if len(current) > target:
        raise ValueError(f"source scaling already has {len(current)} PinGroups, target is {target}")
    children = _instance_list(block)
    by_module: dict[str, list[dict[str, Any]]] = {}
    for child in children:
        by_module.setdefault(str(child["module_name"]), []).append(child)
    reusable = [(name, sorted(items, key=lambda item: item["name"])) for name, items in sorted(by_module.items()) if len(items) >= 2]
    if not reusable:
        if (target - len(current)) % 2:
            raise ValueError("odd PinGroup target is not representable without a reused module type")
        reusable = []
    added_nets = 0
    next_index = 0
    while len(current) < target:
        name = f"CAL_PG{next_index:08d}"
        next_index += 1
        if reusable:
            module_name, items = reusable[next_index % len(reusable)]
            pins = [
                {"parent_inst": items[0]["name"], "parent_module": module_name, "pingroup_name": name, "scope": [], "successors": [], "width": 0.001, "source_connectivity_weight": None, "source_connectivity_kind": "synthetic_calibration", "width_provenance": "pending_capacity_calibration"},
                {"parent_inst": items[1]["name"], "parent_module": module_name, "pingroup_name": name, "scope": [], "successors": [], "width": 0.001, "source_connectivity_weight": None, "source_connectivity_kind": "synthetic_calibration", "width_provenance": "pending_capacity_calibration"},
            ]
            added_groups = 1
        else:
            pairs = sorted(children, key=lambda item: item["name"])
            left, right = pairs[(2 * added_nets) % len(pairs)], pairs[(2 * added_nets + 1) % len(pairs)]
            if left["module_name"] == right["module_name"]:
                continue
            pins = [
                {"parent_inst": left["name"], "parent_module": left["module_name"], "pingroup_name": name, "scope": [], "successors": [], "width": 0.001, "source_connectivity_weight": None, "source_connectivity_kind": "synthetic_calibration", "width_provenance": "pending_capacity_calibration"},
                {"parent_inst": right["name"], "parent_module": right["module_name"], "pingroup_name": name, "scope": [], "successors": [], "width": 0.001, "source_connectivity_weight": None, "source_connectivity_kind": "synthetic_calibration", "width_provenance": "pending_capacity_calibration"},
            ]
            added_groups = 2
        pingroup.append(pins)
        for pin in pins:
            current.add(f"{pin['parent_module']}.{pin['pingroup_name']}")
        lineage_nets.append({"source_net_index": None, "derived_net_index": len(pingroup) - 1, "status": "synthetic_calibration", "pin_count": len(pins), "added_pingroup_count": added_groups})
        added_nets += 1
    return added_nets, len(current)


def _validate_template_geometry(block: dict[str, Any]) -> list[dict[str, str]]:
    grouped: dict[str, list[list[list[float]]]] = {}
    for child in _instance_list(block):
        normalized, _ = canonical_vertex_mapping(child["vertex"], int(child.get("direction", 0)))
        grouped.setdefault(str(child.get("module_name", "")), []).append(normalized)
    errors: list[dict[str, str]] = []
    for module_name, polygons in grouped.items():
        reference = polygons[0]
        if any(polygon != reference for polygon in polygons[1:]):
            errors.append({"code": "module_geometry_mismatch", "detail": module_name})
    return errors


def validate_scaled_artifacts(block: dict[str, Any], pingroup: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Run project validation plus the same-module normalized-geometry invariant."""
    validation = validate_artifacts(block, pingroup)
    validation["errors"].extend(_validate_template_geometry(block))
    validation["error_count"] = len(validation["errors"])
    validation["valid"] = not validation["errors"]
    return validation


def scale_case(block: dict[str, Any], pingroup: list[list[dict[str, Any]]], config: ScaleConfig) -> dict[str, Any]:
    """Replicate a converted case into a deterministic, non-overlapping small case."""
    source_children = _instance_list(block)
    if not source_children:
        raise ValueError("source block has no non-root instances")
    if config.target_module_count < len(source_children):
        raise ValueError("target_module_count cannot be smaller than source instance count")

    copies: list[dict[str, Any]] = []
    tile_maps: list[tuple[dict[str, str], dict[str, str], int]] = []
    tile_count = (config.target_module_count + len(source_children) - 1) // len(source_children)
    positions = [(tile, source_index) for tile in range(tile_count) for source_index in range(len(source_children))][: config.target_module_count]
    target_names, reuse_groups = _allocate_template_names(positions, config.target_reuse_rate)
    directions = _directions_for_positions(positions, config.direction_strategy)
    transformed_templates = [
        _transform_polygon(_source_base_polygon(source_children[source_index]), direction)
        for (_, source_index), direction in zip(positions, directions)
    ]
    cell_width = max(max(point[0] for point in polygon) for polygon in transformed_templates) + config.spacing
    cell_height = max(max(point[1] for point in polygon) for polygon in transformed_templates) + config.spacing
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("source geometry has zero extent")
    target_index = 0
    for tile_index in range(tile_count):
        instance_map: dict[str, str] = {}
        module_map: dict[str, str] = {}
        for source in source_children:
            if target_index >= config.target_module_count:
                break
            source_name = str(source["name"])
            new_name = f"TOP.S{target_index:06d}"
            child = deepcopy(source)
            child["name"] = new_name
            child["module_name"] = target_names[target_index]
            direction = directions[target_index]
            source_shape = _transform_polygon(_source_base_polygon(source), direction)
            source_index = target_index % len(source_children)
            dx = source_index * cell_width
            dy = tile_index * cell_height
            child["vertex"] = _translate_polygon(source_shape, dx, dy)
            child["direction"] = direction
            child["children"] = []
            copies.append(child)
            instance_map[source_name] = new_name
            module_map[source_name] = target_names[target_index]
            target_index += 1
        tile_maps.append((instance_map, module_map, tile_index))

    scaled_nets: list[list[dict[str, Any]]] = []
    lineage_nets: list[dict[str, Any]] = []
    retained_net_count = filtered_net_count = 0
    for instance_map, module_map, tile_index in tile_maps:
        for source_net_index, net in enumerate(pingroup):
            if not all(str(pin.get("parent_inst", "")) in instance_map for pin in net):
                filtered_net_count += 1
                lineage_nets.append({"source_net_index": source_net_index, "tile": tile_index, "status": "filtered", "pin_count": len(net)})
                continue
            scaled_nets.append([_rewrite_pin(pin, instance_map, module_map) for pin in net])
            retained_net_count += 1
            lineage_nets.append({"derived_net_index": len(scaled_nets) - 1, "source_net_index": source_net_index, "tile": tile_index, "status": "retained", "pin_count": len(net)})

    min_scaled_x = min(float(point[0]) for child in copies for point in child["vertex"])
    max_scaled_x = max(float(point[0]) for child in copies for point in child["vertex"])
    min_scaled_y = min(float(point[1]) for child in copies for point in child["vertex"])
    max_scaled_y = max(float(point[1]) for child in copies for point in child["vertex"])
    root = deepcopy(block)
    root["children"] = copies
    root["vertex"] = [
        [min_scaled_x - config.spacing, min_scaled_y - config.spacing],
        [max_scaled_x + config.spacing, min_scaled_y - config.spacing],
        [max_scaled_x + config.spacing, max_scaled_y + config.spacing],
        [min_scaled_x - config.spacing, max_scaled_y + config.spacing],
    ]
    synthetic_net_count = 0
    if config.target_pingroup_count is not None:
        synthetic_net_count, actual_pingroup_count = _append_synthetic_pingroup_nets(
            root, scaled_nets, config.target_pingroup_count, lineage_nets
        )
    else:
        actual_pingroup_count = len(_pingroup_keys(scaled_nets))
    capacity_calibration = None
    if config.width_policy != "preserve":
        calibrated = calibrate_case(
            root,
            scaled_nets,
            CapacityCalibrationConfig(policy=config.width_policy, seed=config.derived_seed),
        )
        root = calibrated["block"]
        scaled_nets = calibrated["pingroup"]
        capacity_calibration = calibrated["report"]
    validation = validate_scaled_artifacts(root, scaled_nets)
    stats = _reuse_stats(root)
    target_reused = int(round(config.target_module_count * config.target_reuse_rate))
    stats.update({
        "source_module_instance_count": len(source_children),
        "target_reuse_instance_ratio": config.target_reuse_rate,
        "target_reused_module_instance_count": target_reused,
        "source_sample_id": config.source_sample_id,
        "seed": config.derived_seed,
        "direction_strategy": config.direction_strategy,
        "retained_net_count": retained_net_count,
        "filtered_net_count": filtered_net_count,
        "pingroup_count": actual_pingroup_count,
        "target_pingroup_count": config.target_pingroup_count,
        "synthetic_calibration_net_count": synthetic_net_count,
        "width_policy": config.width_policy,
        "width_small_group_ratio": (
            capacity_calibration["small_group_ratio"] if capacity_calibration else None
        ),
        "capacity_rejected_group_count": (
            capacity_calibration["packing"]["rejected_group_count"]
            if capacity_calibration else None
        ),
    })
    lineage_modules = []
    for index, child in enumerate(copies):
        tile_index, source_index = positions[index]
        lineage_modules.append({
            "derived_instance": child["name"], "source_instance": source_children[source_index]["name"],
            "source_sample_id": config.source_sample_id, "tile": tile_index, "copy_index": index,
            "source_direction": source_children[source_index].get("direction", 0),
            "derived_direction": child.get("direction", 0),
            "transform": "orthogonal_forward_then_translate",
            "offset": [source_index * cell_width, tile_index * cell_height],
            "derived_module_name": child["module_name"],
        })
    lineage = {
        "scaler_version": SCALER_VERSION, "source_sample_id": config.source_sample_id,
        "modules": lineage_modules, "nets": lineage_nets,
        "counts": {"module_copies": len(lineage_modules), "retained_nets": retained_net_count,
                   "filtered_nets": filtered_net_count, "renamed_instances": len(lineage_modules),
                   "offset_applied": len(lineage_modules), "reuse_groups": len(reuse_groups),
                   "synthetic_calibration_nets": sum(1 for item in lineage_nets if item.get("status") == "synthetic_calibration")},
        "width_calibration": {
            "policy": config.width_policy,
            "capacity_report_file": "capacity_report.json" if capacity_calibration else None,
            "synthetic_capacity_stress": False,
        },
    }
    return {"block": root, "pingroup": scaled_nets, "validation": validation, "statistics": stats, "lineage": lineage, "capacity_calibration": capacity_calibration}


def write_scaled_case(
    block: dict[str, Any],
    pingroup: list[list[dict[str, Any]]],
    destination: str | Path,
    config: ScaleConfig,
    *,
    source_artifact: str = "single-converter output",
) -> dict[str, Any]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    scaled = scale_case(block, pingroup, config)
    if not scaled["validation"]["valid"]:
        raise ValueError(f"scaled case failed validation: {scaled['validation']['errors'][:5]!r}")
    provenance = {
        "derived_artifact_label": "FloorSet-derived synthetic PinAssign benchmark",
        "source_artifact": source_artifact,
        "source_sample_id": config.source_sample_id,
        "source_commit": config.source_commit,
        "scaler_version": SCALER_VERSION,
        "source_data_license": "CC-BY-4.0 (if source is FloorSet data)",
        "synthetic_scaling": True,
        "lineage_file": "lineage.json",
        "capacity_report_file": "capacity_report.json" if scaled["capacity_calibration"] else None,
    }
    manifest = {
        "scaler_version": SCALER_VERSION,
        "converter_version": CONVERTER_VERSION,
        "config": asdict(config),
        "statistics": scaled["statistics"],
        "lineage_file": "lineage.json",
        "policies": {
            "reuse_metric": "reused module instances / all non-root module instances",
            "geometry": "each copied template translated to a disjoint tile; root bounds recomputed",
            "pin_rewrite": "instance names and parent_module rewritten per copy",
            "direction": "real orthogonal transform before translation; direction field records the same transform",
            "target_matrix": "day4_calibration_smoke" if config.target_pingroup_count is not None else "not_applicable_single_scale",
            "pin_width": "capacity-pressure calibrated after scaling" if config.width_policy != "preserve" else "preserved from source and legacy synthetic default",
        },
    }
    _json_dump(destination / "block.json", scaled["block"])
    _json_dump(destination / "pingroup.json", scaled["pingroup"])
    _json_dump(destination / "manifest.json", manifest)
    _json_dump(destination / "provenance.json", provenance)
    _json_dump(destination / "validation.json", scaled["validation"])
    _json_dump(destination / "lineage.json", scaled["lineage"])
    if scaled["capacity_calibration"] is not None:
        _json_dump(destination / "capacity_report.json", scaled["capacity_calibration"])
    return scaled["statistics"] | {"output_dir": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True, type=Path)
    parser.add_argument("--pingroup", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-modules", required=True, type=int)
    parser.add_argument("--reuse-rate", required=True, type=float)
    parser.add_argument("--case-id", default="scale-smoke")
    parser.add_argument("--source-sample-id", default="unknown")
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--direction-strategy", choices=("cycle8", "r0-only"), default="cycle8")
    parser.add_argument("--target-pingroups", type=int, default=None)
    parser.add_argument("--width-policy", choices=("preserve", "baseline", "floorset_first"), default="preserve")
    args = parser.parse_args()
    config = ScaleConfig(args.target_modules, args.reuse_rate, args.seed, args.case_id, args.source_sample_id, source_commit=args.source_commit, direction_strategy=args.direction_strategy, target_pingroup_count=args.target_pingroups, width_policy=args.width_policy)
    block = json.loads(args.block.read_text(encoding="utf-8"))
    pingroup = json.loads(args.pingroup.read_text(encoding="utf-8"))
    print(json.dumps(write_scaled_case(block, pingroup, args.output_dir, config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
