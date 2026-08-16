"""Compose a fixed 240-instance two-level hierarchy from FloorSet geometry."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from shapely.geometry import Polygon

from geometry_utils import canonical_vertex_mapping


HIERARCHY_COMPOSER_VERSION = "floorset-hierarchy-composer-v0.2"


def derive_phase_seed(base_seed: int, case_id: str, phase: str) -> int:
    material = f"{base_seed}|{case_id}|{phase}|{HIERARCHY_COMPOSER_VERSION}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


@dataclass(frozen=True)
class HierarchyConfig:
    case_id: str
    target_reuse_rate: float
    base_seed: int = 7
    seed_namespace: str | None = None
    parent_count: int = 12
    leaves_per_parent: int = 19
    leaf_spacing: float = 1.0
    parent_margin: float = 1.0
    parent_spacing: float = 4.0
    direction_strategy: str = "cycle8"
    reuse_cluster_strategy: str = "pairs"

    def __post_init__(self) -> None:
        if self.parent_count != 12 or self.leaves_per_parent != 19:
            raise ValueError("v1 hierarchy is fixed at 12 parents x 19 leaves")
        if self.target_reuse_rate not in {0.60, 0.95}:
            raise ValueError("v1 supports target reuse rates 0.60 and 0.95")
        if min(self.leaf_spacing, self.parent_margin, self.parent_spacing) < 0:
            raise ValueError("hierarchy spacing and margins must be non-negative")
        if self.direction_strategy not in {"cycle8", "r0-only"}:
            raise ValueError("direction_strategy must be cycle8 or r0-only")
        if self.reuse_cluster_strategy not in {"pairs", "mixed2to10"}:
            raise ValueError("reuse_cluster_strategy must be pairs or mixed2to10")

    @property
    def nonroot_instance_count(self) -> int:
        return self.parent_count * (self.leaves_per_parent + 1)

    @property
    def target_reused_instance_count(self) -> int:
        return int(round(self.nonroot_instance_count * self.target_reuse_rate))

    @property
    def hierarchy_seed(self) -> int:
        return derive_phase_seed(
            self.base_seed, self.seed_namespace or self.case_id, "hierarchy"
        )

    @property
    def reuse_seed(self) -> int:
        return derive_phase_seed(
            self.base_seed, self.seed_namespace or self.case_id, "reuse"
        )


def _walk(module: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield module
    for child in module.get("children", []):
        yield from _walk(child)


def _base_polygon(module: dict[str, Any]) -> list[list[float]]:
    normalized, _ = canonical_vertex_mapping(
        module["vertex"], int(module.get("direction", 0))
    )
    return [[float(x), float(y)] for x, y in normalized]


def _forward(point: list[float], direction: int) -> tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    return {
        0: (x, y),
        1: (-x, -y),
        2: (-y, x),
        3: (y, -x),
        4: (-x, y),
        5: (x, -y),
        6: (y, x),
        7: (-y, -x),
    }[direction]


def _transform(points: list[list[float]], direction: int) -> list[list[float]]:
    transformed = [_forward(point, direction) for point in points]
    min_x = min(point[0] for point in transformed)
    min_y = min(point[1] for point in transformed)
    return [[x - min_x, y - min_y] for x, y in transformed]


def _translate(points: list[list[float]], dx: float, dy: float) -> list[list[float]]:
    return [[float(x) + dx, float(y) + dy] for x, y in points]


def _source_templates(source_block: dict[str, Any]) -> list[dict[str, Any]]:
    templates = [
        child
        for child in source_block.get("children", [])
        if child.get("vertex")
        and not str(child.get("module_name", "")).startswith("IO_")
    ]
    if not templates:
        templates = [child for child in source_block.get("children", []) if child.get("vertex")]
    if not templates:
        raise ValueError("source FloorSet block has no usable leaf geometry")
    return sorted(templates, key=lambda item: str(item.get("name", "")))


def _reuse_pairs(config: HierarchyConfig) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pair_count = config.target_reused_instance_count // 2
    local_pairs = [
        ((parent, leaf), (parent, leaf + 1))
        for parent in range(config.parent_count)
        for leaf in range(0, config.leaves_per_parent - 1, 2)
    ]
    tail_slots = [(parent, config.leaves_per_parent - 1) for parent in range(config.parent_count)]
    tail_pairs = [
        (tail_slots[index], tail_slots[index + 1])
        for index in range(0, len(tail_slots), 2)
    ]
    candidates = local_pairs + tail_pairs
    if pair_count > len(candidates):
        raise ValueError("target reused instance count is not representable by pairs")
    offset = config.reuse_seed % len(local_pairs)
    rotated_local = local_pairs[offset:] + local_pairs[:offset]
    candidates = rotated_local + tail_pairs
    return candidates[:pair_count]


def _partition_cluster_sizes(total: int, offset: int) -> list[int]:
    """Partition an exact reused-instance budget into deterministic clusters of 2..10."""
    # Keep nearly all clusters as pairs so fixed module-type capacity remains
    # close to the proven R60/R95 baseline. One size-10 cluster is sufficient
    # for the deliberately rare 21-100 depth hubs.
    large = [10]
    remaining = total - 10
    if remaining < 0 or remaining % 2:
        raise ValueError(f"reused instance count {total} is not cluster-representable")
    return large + [2] * (remaining // 2)


def _reuse_clusters(config: HierarchyConfig) -> list[tuple[tuple[int, int], ...]]:
    if config.reuse_cluster_strategy == "pairs":
        return [tuple(pair) for pair in _reuse_pairs(config)]

    slots = [
        (parent, leaf)
        for parent in range(config.parent_count)
        for leaf in range(config.leaves_per_parent)
    ]
    offset = config.reuse_seed % len(slots)
    rotated = slots[offset:] + slots[:offset]
    selected = rotated[: config.target_reused_instance_count]
    sizes = _partition_cluster_sizes(
        config.target_reused_instance_count,
        config.reuse_seed % 9,
    )
    clusters: list[tuple[tuple[int, int], ...]] = []
    cursor = 0
    for size in sizes:
        clusters.append(tuple(selected[cursor : cursor + size]))
        cursor += size
    return clusters


def compose_hierarchy(source_block: dict[str, Any], config: HierarchyConfig) -> dict[str, Any]:
    """Build TOP/12 parents/19 leaves while preserving source template sizes."""
    templates = _source_templates(source_block)
    clusters = _reuse_clusters(config)
    cluster_by_slot: dict[tuple[int, int], int] = {}
    for cluster_index, cluster in enumerate(clusters):
        for slot in cluster:
            cluster_by_slot[slot] = cluster_index

    direction_offset = config.hierarchy_seed % 8
    slots: list[dict[str, Any]] = []
    unique_index = 0
    for parent_index in range(config.parent_count):
        for leaf_index in range(config.leaves_per_parent):
            slot = (parent_index, leaf_index)
            cluster_index = cluster_by_slot.get(slot)
            if cluster_index is not None:
                module_name = f"HIER_REUSE_{cluster_index:04d}"
                template_index = (cluster_index + config.reuse_seed) % len(templates)
                reuse_cluster = cluster_index
            else:
                module_name = f"HIER_UNIQUE_{unique_index:04d}"
                template_index = (len(clusters) + unique_index + config.reuse_seed) % len(templates)
                reuse_cluster = None
                unique_index += 1
            direction = 0 if config.direction_strategy == "r0-only" else (
                len(slots) + direction_offset
            ) % 8
            template = templates[template_index]
            polygon = _transform(_base_polygon(template), direction)
            slots.append(
                {
                    "parent_index": parent_index,
                    "leaf_index": leaf_index,
                    "module_name": module_name,
                    "direction": direction,
                    "polygon": polygon,
                    "source_instance": str(template.get("name", "")),
                    "source_module_name": str(template.get("module_name", "")),
                    "reuse_cluster": reuse_cluster,
                }
            )

    max_width = max(max(point[0] for point in slot["polygon"]) for slot in slots)
    max_height = max(max(point[1] for point in slot["polygon"]) for slot in slots)
    cell_width = max_width + config.leaf_spacing
    cell_height = max_height + config.leaf_spacing
    columns, rows = 5, 4
    parent_width = 2 * config.parent_margin + columns * cell_width
    parent_height = 2 * config.parent_margin + rows * cell_height
    parent_columns = 4

    parents: list[dict[str, Any]] = []
    lineage_modules: list[dict[str, Any]] = []
    for parent_index in range(config.parent_count):
        parent_col = parent_index % parent_columns
        parent_row = parent_index // parent_columns
        parent_x = parent_col * (parent_width + config.parent_spacing)
        parent_y = parent_row * (parent_height + config.parent_spacing)
        parent_name = f"TOP.P{parent_index:02d}"
        children: list[dict[str, Any]] = []
        parent_slots = [slot for slot in slots if slot["parent_index"] == parent_index]
        for slot in parent_slots:
            leaf_index = int(slot["leaf_index"])
            col, row = leaf_index % columns, leaf_index // columns
            x = parent_x + config.parent_margin + col * cell_width
            y = parent_y + config.parent_margin + row * cell_height
            name = f"{parent_name}.L{parent_index:02d}_{leaf_index:02d}"
            child = {
                "name": name,
                "module_name": slot["module_name"],
                "direction": slot["direction"],
                "color": f"#{(len(lineage_modules) * 2654435761) & 0xFFFFFF:06x}",
                "vertex": _translate(slot["polygon"], x, y),
                "children": [],
            }
            children.append(child)
            lineage_modules.append(
                {
                    "derived_instance": name,
                    "derived_module_name": slot["module_name"],
                    "hierarchy_path": ["TOP", parent_name, name],
                    "source_instance": slot["source_instance"],
                    "source_module_name": slot["source_module_name"],
                    "direction": slot["direction"],
                    "reuse_pair": slot["reuse_cluster"],
                    "reuse_cluster": slot["reuse_cluster"],
                    "size_changed": False,
                    "transform": "orthogonal_forward_then_translate",
                }
            )
        parents.append(
            {
                "name": parent_name,
                "module_name": f"HIER_PARENT_{parent_index:02d}",
                "direction": 0,
                "color": f"#{(parent_index * 8191) & 0xFFFFFF:06x}",
                "vertex": [
                    [parent_x, parent_y],
                    [parent_x + parent_width, parent_y],
                    [parent_x + parent_width, parent_y + parent_height],
                    [parent_x, parent_y + parent_height],
                ],
                "children": children,
            }
        )
        lineage_modules.append(
            {
                "derived_instance": parent_name,
                "derived_module_name": f"HIER_PARENT_{parent_index:02d}",
                "hierarchy_path": ["TOP", parent_name],
                "source_instance": None,
                "direction": 0,
                "reuse_pair": None,
                "reuse_cluster": None,
                "size_changed": False,
                "transform": "synthetic_parent_bounds",
            }
        )

    root_max_x = max(point[0] for parent in parents for point in parent["vertex"])
    root_max_y = max(point[1] for parent in parents for point in parent["vertex"])
    block = {
        "name": "TOP",
        "module_name": "TOP",
        "direction": 0,
        "color": "#000000",
        "vertex": [
            [-config.parent_spacing, -config.parent_spacing],
            [root_max_x + config.parent_spacing, -config.parent_spacing],
            [root_max_x + config.parent_spacing, root_max_y + config.parent_spacing],
            [-config.parent_spacing, root_max_y + config.parent_spacing],
        ],
        "children": parents,
    }
    validation = validate_hierarchy(block, config)
    if not validation["valid"]:
        raise ValueError(f"composed hierarchy is invalid: {validation['errors'][:5]!r}")
    counts = Counter(
        str(module["module_name"])
        for module in _walk(block)
        if module is not block
    )
    reused_instances = sum(count for count in counts.values() if count >= 2)
    report = {
        "composer_version": HIERARCHY_COMPOSER_VERSION,
        "config": asdict(config),
        "phase_seeds": {
            "hierarchy_seed": config.hierarchy_seed,
            "reuse_seed": config.reuse_seed,
        },
        "parent_count": config.parent_count,
        "leaf_count": config.parent_count * config.leaves_per_parent,
        "nonroot_instance_count": config.nonroot_instance_count,
        "reused_module_instance_count": reused_instances,
        "reuse_instance_ratio": reused_instances / config.nonroot_instance_count,
        "module_type_count": len(counts),
        "reused_module_type_count": sum(count >= 2 for count in counts.values()),
        "reuse_cluster_size_histogram": dict(
            sorted(Counter(count for count in counts.values() if count >= 2).items())
        ),
        "module_type_instance_counts": dict(sorted(counts.items())),
        "source_template_count": len(templates),
        "validation": validation,
    }
    return {"block": block, "report": report, "lineage_modules": lineage_modules}


def validate_hierarchy(block: dict[str, Any], config: HierarchyConfig | None = None) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    modules = list(_walk(block))
    nonroot = modules[1:]
    parents = list(block.get("children", []))
    if config is not None:
        if len(nonroot) != config.nonroot_instance_count:
            errors.append({"code": "module_count", "detail": str(len(nonroot))})
        if len(parents) != config.parent_count:
            errors.append({"code": "parent_count", "detail": str(len(parents))})
        for parent in parents:
            if len(parent.get("children", [])) != config.leaves_per_parent:
                errors.append({"code": "leaf_count", "detail": str(parent.get("name"))})

    for parent in modules:
        parent_polygon = Polygon(parent.get("vertex", []))
        children = list(parent.get("children", []))
        child_polygons = [(child, Polygon(child.get("vertex", []))) for child in children]
        for child, polygon in child_polygons:
            if not parent_polygon.covers(polygon):
                errors.append(
                    {"code": "child_outside_parent", "detail": str(child.get("name", ""))}
                )
        for index, (left, left_polygon) in enumerate(child_polygons):
            for right, right_polygon in child_polygons[index + 1 :]:
                if left_polygon.intersection(right_polygon).area > 1e-9:
                    errors.append(
                        {
                            "code": "sibling_overlap",
                            "detail": f"{left.get('name')}|{right.get('name')}",
                        }
                    )

    normalized_by_type: dict[str, list[list[tuple[float, float]]]] = {}
    for module in nonroot:
        normalized, _ = canonical_vertex_mapping(
            module.get("vertex", []), int(module.get("direction", 0))
        )
        normalized_by_type.setdefault(str(module.get("module_name", "")), []).append(normalized)
    for module_name, polygons in normalized_by_type.items():
        if any(polygon != polygons[0] for polygon in polygons[1:]):
            errors.append({"code": "module_geometry_mismatch", "detail": module_name})
    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "nonroot_instance_count": len(nonroot),
        "parent_count": len(parents),
        "leaf_count": sum(len(parent.get("children", [])) for parent in parents),
    }
