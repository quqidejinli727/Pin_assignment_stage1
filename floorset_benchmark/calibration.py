"""Independent Day 4 calibration validation and resource statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

from PlaceDB import PlaceDB
from homology import HomologyManager
from plan_mcts_batches import BatchPlanner
from segment import SegmentManager
from shapely.geometry import Polygon

from .scaler import validate_scaled_artifacts


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _overlap_count(block: dict[str, Any]) -> int:
    children = [child for child in block.get("children", []) if child.get("name") != "TOP"]
    polygons = []
    for child in children:
        polygons.append(Polygon(child["vertex"]))
    count = 0
    for index, left in enumerate(polygons):
        for right in polygons[index + 1 :]:
            if left.intersection(right).area > 1e-9:
                count += 1
    return count


def _graph_stats(placedb: PlaceDB, homology: HomologyManager) -> dict[str, Any]:
    group_names = sorted(homology.pin_groups)
    net_ids = list(range(len(placedb.nets_list)))
    edges: set[tuple[str, int]] = set()
    group_degree = {name: 0 for name in group_names}
    net_degree = {net_id: 0 for net_id in net_ids}
    for net_id, net in enumerate(placedb.nets_list):
        names = {homology.group_by_pin_full_name[pin.full_name].name for pin in net.pins}
        for name in names:
            edges.add((name, net_id))
            group_degree[name] += 1
            net_degree[net_id] += 1
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for name, net_id in edges:
        g, n = ("g", name), ("n", net_id)
        adjacency.setdefault(g, set()).add(n)
        adjacency.setdefault(n, set()).add(g)
    components = []
    unseen = set(adjacency)
    while unseen:
        root = unseen.pop()
        stack, size = [root], 1
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, set()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    size += 1
        components.append(size)
    def degree_summary(values: list[int]) -> dict[str, float | int]:
        return {"min": min(values, default=0), "avg": sum(values) / len(values) if values else 0.0, "max": max(values, default=0)}
    return {
        "group_node_count": len(group_names), "net_node_count": len(net_ids), "unique_group_net_edge_count": len(edges),
        "group_degree": degree_summary(list(group_degree.values())), "net_degree": degree_summary(list(net_degree.values())),
        "component_count": len(components), "largest_component_nodes": max(components, default=0),
        "component_size_distribution": sorted(components),
    }


def _full_batches(planner: BatchPlanner) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    globally_batched: set[str] = set()
    for component_index, component in enumerate(planner._components()):
        planned = planner._plan_component(component, component_index, globally_batched)
        batches.extend(planned)
        for batch in planned:
            globally_batched.update(batch["group_names"])
    return batches


def _resource_stats(placedb: PlaceDB, homology: HomologyManager, segments: SegmentManager, plan: dict[str, Any], full_batches: list[dict[str, Any]]) -> dict[str, Any]:
    groups = homology.pin_groups
    batch_resources = []
    for batch in full_batches:
        resources = set()
        for group_name in batch["group_names"]:
            resources.update(segment.segment_id for segment in segments.candidates_for_module(groups[group_name].module_name))
        batch_resources.append(resources)
    conflict_edges = 0
    for index, resources in enumerate(batch_resources):
        conflict_edges += sum(1 for other in batch_resources[index + 1 :] if resources & other)
    waves: list[set[str]] = []
    wave_batches: list[int] = []
    for resources in batch_resources:
        placed = False
        for index, used in enumerate(waves):
            if not resources & used:
                waves[index] |= resources
                wave_batches[index] += 1
                placed = True
                break
        if not placed:
            waves.append(set(resources))
            wave_batches.append(1)
    return {
        "resource_id_count": len(segments.abstract_segments),
        "resource_conflict_edge_count": conflict_edges,
        "batch_count": len(batch_resources),
        "component_count": plan["summary"]["component_count"],
        "wave_count_greedy": len(waves),
        "wave_batch_count_distribution": sorted(wave_batches),
        "max_wave_parallel_batches": max(wave_batches, default=0),
    }


def validate_case(case_dir: str | Path, expected_pingroup_count: int) -> dict[str, Any]:
    case_dir = Path(case_dir)
    tracemalloc.start()
    block = _read(case_dir / "block.json")
    pingroup = _read(case_dir / "pingroup.json")
    started = time.perf_counter()
    validation = validate_scaled_artifacts(block, pingroup)
    validation["overlap_pair_count"] = _overlap_count(block)
    validation["pingroup_count"] = len({f"{pin.get('parent_module')}.{pin.get('pingroup_name')}" for net in pingroup for pin in net})
    validation["expected_pingroup_count"] = expected_pingroup_count
    validation["errors"] = list(validation["errors"])
    if validation["overlap_pair_count"]:
        validation["errors"].append({"code": "polygon_overlap", "detail": str(validation["overlap_pair_count"])})
    if validation["pingroup_count"] != expected_pingroup_count:
        validation["errors"].append({"code": "pingroup_count_mismatch", "detail": str(validation["pingroup_count"])})
    validation["valid"] = not validation["errors"]
    validation["error_count"] = len(validation["errors"])
    validation_seconds = time.perf_counter() - started

    if not validation["valid"]:
        report = {
            "case_id": case_dir.name,
            "validation": validation,
            "statistics": {},
            "timing_seconds": {"validation": validation_seconds, "placedb_load": None, "segment_build": None, "batch_plan": None},
            "peak_tracemalloc_mb": round(tracemalloc.get_traced_memory()[1] / (1024 * 1024), 3),
            "resource_stats": {},
            "batch_summary": {},
            "artifact_sha256": {path.name: _sha256(path) for path in sorted(case_dir.glob("*.json")) if path.name != "calibration_report.json"},
        }
        tracemalloc.stop()
        return report

    load_start = time.perf_counter()
    placedb = PlaceDB(str(case_dir / "block.json"), str(case_dir / "pingroup.json"))
    load_seconds = time.perf_counter() - load_start
    segment_start = time.perf_counter()
    segments = SegmentManager(placedb)
    segment_seconds = time.perf_counter() - segment_start
    homology = HomologyManager(placedb)
    plan_start = time.perf_counter()
    planner = BatchPlanner(placedb, homology)
    plan = planner.plan()
    full_batches = _full_batches(planner)
    plan_seconds = time.perf_counter() - plan_start
    artifacts = {path.name: _sha256(path) for path in sorted(case_dir.glob("*.json")) if path.name != "calibration_report.json"}
    report = {
        "case_id": case_dir.name, "validation": validation,
        "statistics": {"place_db_modules": len(placedb.all_modules_list), "nets": len(placedb.nets_list), "pins": placedb.total_pin_count},
        "timing_seconds": {"validation": validation_seconds, "placedb_load": load_seconds, "segment_build": segment_seconds, "batch_plan": plan_seconds},
        "graph_stats": _graph_stats(placedb, homology),
        "resource_stats": _resource_stats(placedb, homology, segments, plan, full_batches),
        "batch_summary": plan["summary"], "artifact_sha256": artifacts,
    }
    report["peak_tracemalloc_mb"] = round(tracemalloc.get_traced_memory()[1] / (1024 * 1024), 3)
    tracemalloc.stop()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--expected-pingroups", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = validate_case(args.case_dir, args.expected_pingroups)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
