"""Analyze Stage1 local MCTS tree batches without running MCTS search.

The real solver builds many local MCTS trees.  Each tree starts from one
unassigned homology group, collects related nets and pins, searches all
currently unassigned homology groups touched by those pins, and only commits
groups fully covered by the current ``pins_in`` set.  This script simulates
that batching/commit logic and writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from PlaceDB import Net, Pin, PlaceDB
from config import DEFAULT_CONFIG
from homology import HomologyManager, PinHomologyGroup


def _collect_pins(nets: Iterable[Net]) -> List[Pin]:
    pins_by_name: Dict[str, Pin] = {}
    for net in nets:
        for pin in net.pins:
            pins_by_name[pin.full_name] = pin
    return list(pins_by_name.values())


def _group_pin_count(groups: Iterable[PinHomologyGroup]) -> int:
    return sum(len(group.pins) for group in groups)


def _group_records(groups: Iterable[PinHomologyGroup]) -> List[dict]:
    return [
        {
            "name": group.name,
            "pin_count": len(group.pins),
            "module_name": group.module_name,
            "reuse_count": group.reuse_count,
        }
        for group in groups
    ]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _committable_groups(
    groups: Iterable[PinHomologyGroup],
    pin_full_names: set[str],
    coverage_threshold: float,
) -> List[PinHomologyGroup]:
    threshold = max(0.0, min(1.0, coverage_threshold))
    committable = []
    for group in groups:
        if not group.pins:
            continue
        covered_count = sum(1 for pin in group.pins if pin.full_name in pin_full_names)
        if _ratio(covered_count, len(group.pins)) >= threshold:
            committable.append(group)
    return committable


def analyze_batches(
    block_json: str | Path,
    pingroup_json: str | Path,
    coverage_threshold: float = DEFAULT_CONFIG.homology_group_commit_coverage_threshold,
    min_committable_group_ratio: float = DEFAULT_CONFIG.mcts_tree_min_committable_group_ratio,
    use_fanout_reuse_for_sorting: bool = DEFAULT_CONFIG.homology_use_fanout_reuse_for_sorting,
) -> dict:
    placedb = PlaceDB(str(block_json), str(pingroup_json))
    homology = HomologyManager(
        placedb,
        use_fanout_reuse_for_sorting=use_fanout_reuse_for_sorting,
    )

    tree_reports: List[dict] = []
    skipped_tree_reports: List[dict] = []
    non_mcts_assignments: List[dict] = []
    tree_index = 0
    candidate_tree_index = 0

    for seed_group in homology.unassigned_groups():
        if seed_group.assigned:
            continue

        nets = homology.get_related_nets(seed_group)
        if not nets:
            homology.mark_assigned(seed_group, "simulated_no_related_net")
            non_mcts_assignments.append(
                {
                    "reason": "no_related_net",
                    "group": seed_group.name,
                    "pin_count": len(seed_group.pins),
                }
            )
            continue

        pins_in = _collect_pins(nets)
        related_groups = [
            group
            for group in homology.groups_for_pins(pins_in)
            if not group.assigned
        ]
        if not related_groups:
            homology.mark_assigned(seed_group, "simulated_no_related_unassigned_group")
            non_mcts_assignments.append(
                {
                    "reason": "no_related_unassigned_group",
                    "group": seed_group.name,
                    "pin_count": len(seed_group.pins),
                }
            )
            continue

        pin_full_names = {pin.full_name for pin in pins_in}
        committable_groups = _committable_groups(
            related_groups,
            pin_full_names,
            coverage_threshold,
        )
        committable_group_names = {group.name for group in committable_groups}
        deferred_groups = [
            group for group in related_groups if group.name not in committable_group_names
        ]

        search_group_count = len(related_groups)
        search_pin_count = _group_pin_count(related_groups)
        committable_pin_count = _group_pin_count(committable_groups)
        committable_group_ratio = _ratio(len(committable_groups), search_group_count)
        pins_in_count = len(pin_full_names)

        report = {
            "candidate_tree_index": candidate_tree_index,
            "tree_index": tree_index,
            "seed_group": seed_group.name,
            "related_net_count": len(nets),
            "related_net_ids": [net.net_id for net in nets],
            "pins_in_count": pins_in_count,
            "mcts_depth": search_group_count,
            "search_group_count": search_group_count,
            "search_pin_count": search_pin_count,
            "committable_group_count": len(committable_groups),
            "committable_pin_count": committable_pin_count,
            "committable_group_ratio_of_search_groups": committable_group_ratio,
            "committable_pin_ratio_of_search_pins": _ratio(
                committable_pin_count,
                search_pin_count,
            ),
            "committable_pin_ratio_of_pins_in": _ratio(
                committable_pin_count,
                pins_in_count,
            ),
            "deferred_group_count": len(deferred_groups),
            "deferred_pin_count": _group_pin_count(deferred_groups),
            "skipped_by_low_committable_ratio": (
                committable_group_ratio <= min_committable_group_ratio
            ),
            "search_groups": _group_records(related_groups),
            "committable_groups": _group_records(committable_groups),
            "deferred_groups": _group_records(deferred_groups),
        }

        if report["skipped_by_low_committable_ratio"]:
            report["tree_index"] = None
            skipped_tree_reports.append(report)
            candidate_tree_index += 1
            continue

        tree_reports.append(report)
        tree_index += 1
        candidate_tree_index += 1

        for group in committable_groups:
            homology.mark_assigned(group, f"simulated_tree_{tree_index - 1}")

    final_greedy_groups = homology.unassigned_groups()
    for group in final_greedy_groups:
        homology.mark_assigned(group, "simulated_final_greedy_sweep")

    depth_values = [report["mcts_depth"] for report in tree_reports]
    search_pin_values = [report["search_pin_count"] for report in tree_reports]
    committable_pin_values = [report["committable_pin_count"] for report in tree_reports]
    final_greedy_pin_count = _group_pin_count(final_greedy_groups)
    total_search_pin_count = sum(search_pin_values)
    total_committable_pin_count = sum(committable_pin_values)
    skipped_search_group_count = sum(report["search_group_count"] for report in skipped_tree_reports)
    skipped_search_pin_count = sum(report["search_pin_count"] for report in skipped_tree_reports)

    return {
        "input": {
            "block_json": str(Path(block_json)),
            "pingroup_json": str(Path(pingroup_json)),
        },
        "parameters": {
            "use_fanout_reuse_for_sorting": use_fanout_reuse_for_sorting,
            "coverage_threshold": coverage_threshold,
            "min_committable_group_ratio": min_committable_group_ratio,
        },
        "summary": {
            "total_homology_group_count": len(homology.pin_groups),
            "total_pin_count": placedb.total_pin_count,
            "mcts_tree_count": len(tree_reports),
            "total_mcts_search_group_visits": sum(depth_values),
            "total_mcts_search_pin_visits": total_search_pin_count,
            "total_mcts_committable_pin_count": total_committable_pin_count,
            "overall_committable_group_ratio_of_built_trees": _ratio(
                sum(report["committable_group_count"] for report in tree_reports),
                sum(report["search_group_count"] for report in tree_reports),
            ),
            "overall_committable_ratio_of_search_pin_visits": _ratio(
                total_committable_pin_count,
                total_search_pin_count,
            ),
            "skipped_mcts_tree_count": len(skipped_tree_reports),
            "skipped_mcts_search_group_count": skipped_search_group_count,
            "skipped_mcts_search_pin_count": skipped_search_pin_count,
            "final_greedy_group_count": len(final_greedy_groups),
            "final_greedy_pin_count": final_greedy_pin_count,
            "non_mcts_assignment_count": len(non_mcts_assignments),
            "non_mcts_pin_count": sum(item["pin_count"] for item in non_mcts_assignments),
            "max_mcts_depth": max(depth_values, default=0),
            "min_mcts_depth": min(depth_values, default=0),
            "average_mcts_depth": _ratio(sum(depth_values), len(depth_values)),
            "average_tree_committable_pin_ratio": _ratio(
                sum(report["committable_pin_ratio_of_search_pins"] for report in tree_reports),
                len(tree_reports),
            ),
        },
        "depth_histogram": _histogram(depth_values),
        "tree_reports": tree_reports,
        "skipped_tree_reports": skipped_tree_reports,
        "non_mcts_assignments": non_mcts_assignments,
        "final_greedy_groups": _group_records(final_greedy_groups),
    }


def _histogram(values: Iterable[int]) -> dict:
    histogram: Dict[str, int] = {}
    for value in values:
        key = str(value)
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: int(item[0])))


def default_output_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return output_dir / f"mcts_tree_batch_analysis_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate Stage1 local MCTS tree batching and write a JSON report."
    )
    parser.add_argument("--block", required=True, help="Path to block.json.")
    parser.add_argument("--pingroup", required=True, help="Path to pingroup.json.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output report path. Defaults to mcts_tree_batch_analysis_<timestamp>.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory used when --output is not provided.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=DEFAULT_CONFIG.homology_group_commit_coverage_threshold,
        help="Pin coverage ratio required for a homology group to be committable.",
    )
    parser.add_argument(
        "--fanout-reuse-sorting",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.homology_use_fanout_reuse_for_sorting,
        help="Whether multi-fanout pins boost homology sorting reuse count.",
    )
    parser.add_argument(
        "--min-committable-group-ratio",
        type=float,
        default=DEFAULT_CONFIG.mcts_tree_min_committable_group_ratio,
        help="Skip a candidate MCTS tree when committable/search group ratio is <= this value.",
    )
    args = parser.parse_args()

    report = analyze_batches(
        args.block,
        args.pingroup,
        use_fanout_reuse_for_sorting=args.fanout_reuse_sorting,
        coverage_threshold=args.coverage_threshold,
        min_committable_group_ratio=args.min_committable_group_ratio,
    )
    output_path = Path(args.output) if args.output else default_output_path(Path(args.output_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote MCTS tree batch analysis: {output_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
