"""Analyze static, ordered, runtime-effective, and BatchPlanner tree depths separately."""

from __future__ import annotations

import json
import math
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from PlaceDB import PlaceDB
from analyze_mcts_tree_batches import analyze_batches
from config import DEFAULT_CONFIG
from homology import HomologyManager
from plan_mcts_batches import BatchPlanner


ANALYZER_VERSION = "mcts-tree-depth-analyzer-v0.1"


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items(), key=lambda item: int(item[0])))


def _summary(values: list[int]) -> dict[str, Any]:
    count = len(values)
    buckets = {
        "depth_1": sum(value == 1 for value in values),
        "depth_2_10": sum(2 <= value <= 10 for value in values),
        "depth_11_20": sum(11 <= value <= 20 for value in values),
        "depth_21_100": sum(21 <= value <= 100 for value in values),
        "depth_gt_100": sum(value > 100 for value in values),
    }
    return {
        "count": count,
        "min": min(values, default=None),
        "mean": mean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values, default=None),
        "histogram": _histogram(values),
        "buckets": {
            name: {"count": value, "ratio": value / count if count else 0.0}
            for name, value in buckets.items()
        },
    }


def _dag_longest_path(net: list[dict[str, Any]]) -> int:
    names = {f"{pin['parent_inst']}.{pin['pingroup_name']}" for pin in net}
    edges = {
        f"{pin['parent_inst']}.{pin['pingroup_name']}": [
            successor for successor in pin.get("successors", []) if successor in names
        ]
        for pin in net
    }
    memo: dict[str, int] = {}

    def visit(name: str, active: set[str]) -> int:
        if name in memo:
            return memo[name]
        if name in active:
            raise ValueError(f"successor cycle at {name}")
        memo[name] = max(
            (1 + visit(successor, active | {name}) for successor in edges[name]),
            default=0,
        )
        return memo[name]

    return max((visit(name, set()) for name in names), default=0)


def _runtime_config_snapshot() -> dict[str, Any]:
    return {
        "homology_use_fanout_reuse_for_sorting": DEFAULT_CONFIG.homology_use_fanout_reuse_for_sorting,
        "homology_skip_uncovered_groups": DEFAULT_CONFIG.homology_skip_uncovered_groups,
        "homology_skip_coverage_threshold": DEFAULT_CONFIG.homology_skip_coverage_threshold,
        "homology_group_commit_coverage_threshold": DEFAULT_CONFIG.homology_group_commit_coverage_threshold,
        "mcts_search_each_pingroup_once": DEFAULT_CONFIG.mcts_search_each_pingroup_once,
        "assignment_rescan_until_stable": DEFAULT_CONFIG.assignment_rescan_until_stable,
    }


def analyze_mcts_tree_depths(
    block_json: str | Path,
    pingroup_json: str | Path,
    *,
    lineage_json: str | Path | None = None,
    capacity_report_json: str | Path | None = None,
) -> dict[str, Any]:
    block_path, pingroup_path = Path(block_json), Path(pingroup_json)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    settings = _runtime_config_snapshot()
    if settings["mcts_search_each_pingroup_once"] or settings["assignment_rescan_until_stable"]:
        raise ValueError(
            "depth analyzer currently freezes the formal default ordering: search_each_once=false, rescan=false"
        )
    homology = HomologyManager(
        placedb,
        use_fanout_reuse_for_sorting=settings["homology_use_fanout_reuse_for_sorting"],
    )

    static_records: list[dict[str, Any]] = []
    related_net_counts: list[int] = []
    group_net_counts: dict[str, int] = {}
    for group_name in homology.sorted_group_names:
        group = homology.pin_groups[group_name]
        nets = homology.get_related_nets(group)
        pins = {pin.full_name: pin for net in nets for pin in net.pins}
        related_groups = homology.groups_for_pins(pins.values())
        related_net_counts.append(len(nets))
        group_net_counts[group_name] = len(nets)
        static_records.append(
            {
                "seed_group": group_name,
                "static_raw_depth": len(related_groups),
                "related_net_count": len(nets),
                "pin_count": len(pins),
                "hub_instance_pin_count": len(group.pins),
                "related_net_ids": [net.net_id for net in nets],
                "related_groups": [item.name for item in related_groups],
            }
        )

    runtime = analyze_batches(
        block_path,
        pingroup_path,
        coverage_threshold=settings["homology_group_commit_coverage_threshold"],
        use_fanout_reuse_for_sorting=settings["homology_use_fanout_reuse_for_sorting"],
        skip_uncovered_groups=settings["homology_skip_uncovered_groups"],
        skip_coverage_threshold=settings["homology_skip_coverage_threshold"],
        include_tree_details=True,
    )
    ordered_raw = [int(tree["raw_search_group_count"]) for tree in runtime["tree_reports"]]
    effective = [int(tree["search_group_count"]) for tree in runtime["tree_reports"]]

    planner_homology = HomologyManager(
        placedb,
        use_fanout_reuse_for_sorting=settings["homology_use_fanout_reuse_for_sorting"],
    )
    planner = BatchPlanner(placedb, planner_homology)
    components = planner._components()
    batch_plan = planner.plan()
    component_depths = [len(component["groups"]) for component in components]
    batch_depths = [int(batch["group_count"]) for batch in batch_plan["batches"]]

    raw_pingroup = json.loads(pingroup_path.read_text(encoding="utf-8"))
    net_records = []
    for net_index, net in enumerate(raw_pingroup):
        groups = {
            f"{pin['parent_module']}.{pin['pingroup_name']}" for pin in net
        }
        net_records.append(
            {
                "net_index": net_index,
                "pin_count": len(net),
                "net_pingroup_count": len(groups),
                "successor_dag_longest_path": _dag_longest_path(net),
            }
        )

    lineage = {}
    if lineage_json and Path(lineage_json).exists():
        lineage = json.loads(Path(lineage_json).read_text(encoding="utf-8"))
    tree_lineage = {
        str(item["tree_template_id"]): item
        for item in lineage.get("nets", [])
        if item.get("tree_template_id")
    }
    runtime_by_seed = {tree["seed_group"]: tree for tree in runtime["tree_reports"]}
    cross_tree_records = []
    seen_templates: set[str] = set()
    for record in tree_lineage.values():
        template_id = str(record["tree_template_id"])
        if template_id in seen_templates:
            continue
        seen_templates.add(template_id)
        hub = str(record["hub_homology_group"])
        actual = runtime_by_seed.get(hub)
        cross_tree_records.append(
            {
                "template_id": template_id,
                "template": record.get("tree_template"),
                "target_depth_bucket": record.get("target_depth_bucket"),
                "hub_homology_group": hub,
                "expected_effective_depth": record.get("target_effective_depth"),
                "actual_effective_depth": actual.get("search_group_count") if actual else None,
                "ordered_raw_depth": actual.get("raw_search_group_count") if actual else None,
                "related_net_count": actual.get("related_net_count") if actual else group_net_counts.get(hub),
                "committable_group_count": actual.get("committable_group_count") if actual else None,
                "deferred_group_count": actual.get("deferred_group_count") if actual else None,
                "skipped_group_count": actual.get("true_skipped_group_count") if actual else None,
                "topology_seed": record.get("topology_seed"),
                "source_sample_id": record.get("source_sample_id"),
                "width_provenance": record.get("width_provenance"),
            }
        )

    capacity = {}
    if capacity_report_json and Path(capacity_report_json).exists():
        capacity = json.loads(Path(capacity_report_json).read_text(encoding="utf-8"))
    packing = capacity.get("packing", {})
    capacity_deduction = {
        "semantic_key": "parent_module.pingroup_name",
        "homology_group_count": len(homology.pin_groups),
        "deduction_count": len(homology.pin_groups),
        "multi_net_homology_group_count": sum(value > 1 for value in group_net_counts.values()),
        "group_instance_pin_count_summary": _summary([len(group.pins) for group in homology.pin_groups.values()]),
        "related_net_count_summary": _summary(related_net_counts),
        "packing_rejected_group_count": packing.get("rejected_group_count"),
        "hard_overflow_count": packing.get("hard_overflow_count"),
    }

    effective_summary = _summary(effective)
    acceptance = {
        "depth_2_10_ratio_at_least_60pct": effective_summary["buckets"]["depth_2_10"]["ratio"] >= 0.60,
        "depth_11_20_nonzero": effective_summary["buckets"]["depth_11_20"]["count"] > 0,
        "depth_21_100_nonzero_at_most_5pct": (
            0 < effective_summary["buckets"]["depth_21_100"]["ratio"] <= 0.05
        ),
        "depth_gt_100_zero": effective_summary["buckets"]["depth_gt_100"]["count"] == 0,
        "p50_between_2_and_5": effective_summary["p50"] is not None and 2 <= effective_summary["p50"] <= 5,
        "p90_between_8_and_20": effective_summary["p90"] is not None and 8 <= effective_summary["p90"] <= 20,
        "max_gt_20_at_most_100": effective_summary["max"] is not None and 20 < effective_summary["max"] <= 100,
    }
    return {
        "analyzer_version": ANALYZER_VERSION,
        "formal_assignment_solver_config": settings,
        "formal_tree_construction_path": [
            "seed homology group",
            "HomologyManager.get_related_nets(seed_group)",
            "collect all Pins in those Nets",
            "HomologyManager.groups_for_pins(pins)",
            "remove assigned/search-once groups",
            "apply coverage skip",
            "MCTSSolver(groups=search_groups, nets=related_nets)",
        ],
        "static_raw_depth": _summary([record["static_raw_depth"] for record in static_records]),
        "ordered_raw_depth": _summary(ordered_raw),
        "effective_mcts_depth": effective_summary,
        "batch_component_depth": _summary(component_depths),
        "batch_depth": _summary(batch_depths),
        "net_pingroup_count": _summary([record["net_pingroup_count"] for record in net_records]),
        "net_pin_count": _summary([record["pin_count"] for record in net_records]),
        "successor_dag_longest_path": _summary(
            [record["successor_dag_longest_path"] for record in net_records]
        ),
        "related_net_count": _summary(related_net_counts),
        "multi_net_homology_group_count": sum(value > 1 for value in group_net_counts.values()),
        "depth_source": {
            "cross_net_homology_tree_count": sum(tree["related_net_count"] > 1 for tree in runtime["tree_reports"]),
            "single_net_multi_group_tree_count": sum(
                tree["related_net_count"] == 1 and tree["search_group_count"] > 1
                for tree in runtime["tree_reports"]
            ),
        },
        "capacity_deduction": capacity_deduction,
        "acceptance": acceptance,
        "acceptance_passed": all(acceptance.values()),
        "runtime_tree_records": runtime["tree_reports"],
        "static_tree_records": static_records,
        "cross_net_tree_records": cross_tree_records,
        "net_records": net_records,
        "batch_planner": {
            "consumed_by_assignment_solver": False,
            "note": "AssignmentSolver does not consume BatchPlanner output; one batch is not one runtime MCTS tree.",
            "summary": batch_plan["summary"],
            "component_histogram": batch_plan["component_histogram"],
            "batch_depth_histogram": batch_plan["batch_depth_histogram"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True, type=Path)
    parser.add_argument("--pingroup", required=True, type=Path)
    parser.add_argument("--lineage", type=Path)
    parser.add_argument("--capacity-report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_mcts_tree_depths(
        args.block,
        args.pingroup,
        lineage_json=args.lineage,
        capacity_report_json=args.capacity_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "effective_mcts_depth": report["effective_mcts_depth"],
        "acceptance": report["acceptance"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
