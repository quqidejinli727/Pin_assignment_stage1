"""Analyze equal-size large nets and the PinGroups they touch.

For each set of nets with the same pin count above a threshold, this script
collects all PinGroups touched by those nets and classifies them into:

- internal PinGroups: all related nets of the PinGroup are inside the large-net set.
- external PinGroups: the PinGroup also touches at least one net outside the set.

Only counts are reported for the external nets; full external net lists are
intentionally omitted to keep the report compact.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Set

from PlaceDB import Net, PlaceDB
from homology import HomologyManager, PinHomologyGroup


def _group_nets(group: PinHomologyGroup, placedb: PlaceDB) -> Set[int]:
    net_ids = set()
    for pin in group.pins:
        for net in placedb.iter_pin_nets(pin):
            net_ids.add(net.net_id)
    return net_ids


def _pin_names(nets: Iterable[Net]) -> Set[str]:
    return {pin.full_name for net in nets for pin in net.pins}


def analyze_equal_large_nets(
    block_json: str | Path,
    pingroup_json: str | Path,
    *,
    pin_threshold: int = 1500,
    min_equal_net_count: int = 2,
    use_fanout_reuse_for_sorting: bool = False,
) -> dict:
    placedb = PlaceDB(str(block_json), str(pingroup_json))
    homology = HomologyManager(
        placedb,
        use_fanout_reuse_for_sorting=use_fanout_reuse_for_sorting,
    )
    nets_by_pin_count: Dict[int, list[Net]] = {}
    for net in placedb.nets_list:
        pin_count = len(net.pins)
        if pin_count > pin_threshold:
            nets_by_pin_count.setdefault(pin_count, []).append(net)

    analyses = []
    for pin_count, nets in sorted(nets_by_pin_count.items()):
        if len(nets) < min_equal_net_count:
            continue
        large_net_ids = {net.net_id for net in nets}
        large_pin_names = _pin_names(nets)
        touched_groups = {
            homology.group_by_pin_full_name[pin_name].name
            for pin_name in large_pin_names
        }
        internal_groups = []
        external_groups = []
        external_net_ids = set()
        partial_pin_groups = 0

        for group_name in sorted(touched_groups, key=lambda name: homology.sort_rank[name]):
            group = homology.pin_groups[group_name]
            group_net_ids = _group_nets(group, placedb)
            outside_net_ids = group_net_ids - large_net_ids
            all_group_pins_in_large_nets = all(
                pin.full_name in large_pin_names for pin in group.pins
            )
            if outside_net_ids:
                external_groups.append(group_name)
                external_net_ids.update(outside_net_ids)
            elif all_group_pins_in_large_nets:
                internal_groups.append(group_name)
            else:
                partial_pin_groups += 1

        analyses.append(
            {
                "pin_count": pin_count,
                "equal_large_net_count": len(nets),
                "large_net_ids": sorted(large_net_ids),
                "large_net_total_pin_visits": sum(len(net.pins) for net in nets),
                "touched_pin_group_count": len(touched_groups),
                "internal_pin_group_count": len(internal_groups),
                "external_pin_group_count": len(external_groups),
                "partial_unclassified_pin_group_count": partial_pin_groups,
                "external_unique_net_count": len(external_net_ids),
            }
        )

    return {
        "input": {
            "block_json": str(Path(block_json)),
            "pingroup_json": str(Path(pingroup_json)),
        },
        "parameters": {
            "pin_threshold": pin_threshold,
            "min_equal_net_count": min_equal_net_count,
            "use_fanout_reuse_for_sorting": use_fanout_reuse_for_sorting,
        },
        "summary": {
            "large_equal_pin_count_bucket_count": len(analyses),
            "large_equal_net_count": sum(item["equal_large_net_count"] for item in analyses),
            "total_touched_pin_group_count": sum(
                item["touched_pin_group_count"] for item in analyses
            ),
            "total_internal_pin_group_count": sum(
                item["internal_pin_group_count"] for item in analyses
            ),
            "total_external_pin_group_count": sum(
                item["external_pin_group_count"] for item in analyses
            ),
            "total_external_unique_net_count_sum_by_bucket": sum(
                item["external_unique_net_count"] for item in analyses
            ),
        },
        "analyses": analyses,
    }


def default_output_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return output_dir / f"equal_large_net_analysis_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze equal-size large nets.")
    parser.add_argument("--block", required=True, help="Path to block.json.")
    parser.add_argument("--pingroup", required=True, help="Path to pingroup.json.")
    parser.add_argument("--output", default=None, help="Output JSON path.")
    parser.add_argument("--output-dir", default=".", help="Output directory when --output is omitted.")
    parser.add_argument(
        "--pin-threshold",
        type=int,
        default=1500,
        help="Only nets with pin count greater than this threshold are analyzed.",
    )
    parser.add_argument(
        "--min-equal-net-count",
        type=int,
        default=2,
        help="Minimum number of nets sharing the same pin count to form an analysis group.",
    )
    parser.add_argument(
        "--fanout-reuse-sorting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether multi-fanout pins boost homology sorting reuse count.",
    )
    args = parser.parse_args()

    report = analyze_equal_large_nets(
        args.block,
        args.pingroup,
        pin_threshold=args.pin_threshold,
        min_equal_net_count=args.min_equal_net_count,
        use_fanout_reuse_for_sorting=args.fanout_reuse_sorting,
    )
    output_path = Path(args.output) if args.output else default_output_path(Path(args.output_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote equal large net analysis: {output_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
