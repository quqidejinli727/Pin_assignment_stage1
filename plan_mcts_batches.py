"""Plan MCTS pin-group batches from the PinGroup-Net bipartite graph.

This is an offline planner.  It does not run MCTS or assign pins.  The goal is
to group homology pin groups into batches whose nets are as complete as
possible, while handling very large nets before ordinary balanced batches.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set

from PlaceDB import Net, PlaceDB
from homology import HomologyManager, PinHomologyGroup


def _histogram(values: Iterable[int], bucket_size: int | None = None) -> dict:
    histogram: Dict[str, int] = {}
    for value in values:
        if bucket_size and value > bucket_size:
            lower = ((value - 1) // bucket_size) * bucket_size + 1
            upper = lower + bucket_size - 1
            key = f"{lower}-{upper}"
        else:
            key = str(value)
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: int(item[0].split("-", 1)[0])))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _group_pin_count(group_names: Iterable[str], groups_by_name: Dict[str, PinHomologyGroup]) -> int:
    return sum(len(groups_by_name[name].pins) for name in group_names)


class BatchPlanner:
    def __init__(
        self,
        placedb: PlaceDB,
        homology: HomologyManager,
        *,
        target_depth: int = 128,
        max_depth: int = 256,
        large_net_pin_threshold: int = 500,
        large_net_group_threshold: int = 500,
        large_net_max_depth: int = 2500,
        absorb_net_pin_threshold: int = 100,
    ):
        self.placedb = placedb
        self.homology = homology
        self.target_depth = max(1, target_depth)
        self.max_depth = max(1, max_depth)
        self.large_net_pin_threshold = max(1, large_net_pin_threshold)
        self.large_net_group_threshold = max(1, large_net_group_threshold)
        self.large_net_max_depth = max(0, large_net_max_depth)
        self.absorb_net_pin_threshold = max(1, absorb_net_pin_threshold)
        self.groups_by_name = homology.pin_groups
        self.group_rank = homology.sort_rank
        self.nets_by_id = {net.net_id: net for net in placedb.nets_list}
        self.group_to_nets: Dict[str, Set[str]] = {name: set() for name in self.groups_by_name}
        self.net_to_groups: Dict[str, Set[str]] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        for net in self.placedb.nets_list:
            group_names = set()
            for pin in net.pins:
                group = self.homology.group_by_pin_full_name[pin.full_name]
                group_names.add(group.name)
                self.group_to_nets.setdefault(group.name, set()).add(net.net_id)
            self.net_to_groups[net.net_id] = group_names

    def plan(self) -> dict:
        components = self._components()
        batches = []
        batched_groups: Set[str] = set()
        for component_index, component in enumerate(components):
            component_batches = self._plan_component(component, component_index, batched_groups)
            batches.extend(component_batches)
            for batch in component_batches:
                batched_groups.update(batch["group_names"])
        return self._build_report(components, batches)

    def _components(self) -> List[dict]:
        unvisited_groups = set(self.groups_by_name)
        components = []
        while unvisited_groups:
            seed_group = min(unvisited_groups, key=lambda name: self.group_rank[name])
            groups = set()
            nets = set()
            queue = deque([("group", seed_group)])
            unvisited_groups.remove(seed_group)
            while queue:
                kind, name = queue.popleft()
                if kind == "group":
                    groups.add(name)
                    for net_id in self.group_to_nets.get(name, set()):
                        if net_id not in nets:
                            nets.add(net_id)
                            queue.append(("net", net_id))
                else:
                    for group_name in self.net_to_groups.get(name, set()):
                        if group_name in unvisited_groups:
                            unvisited_groups.remove(group_name)
                            queue.append(("group", group_name))
            components.append({"groups": groups, "nets": nets})
        components.sort(key=lambda item: (-len(item["groups"]), -len(item["nets"])))
        return components

    def _plan_component(
        self,
        component: dict,
        component_index: int,
        globally_batched_groups: Set[str],
    ) -> List[dict]:
        unbatched = set(component["groups"]) - globally_batched_groups
        if not unbatched:
            return []
        if len(unbatched) <= self.max_depth:
            return [self._make_batch(component_index, "component", unbatched)]

        batches = []
        while True:
            seed_net = self._largest_unbatched_large_net(component["nets"], unbatched)
            if seed_net is None:
                break
            groups = self.net_to_groups[seed_net] & unbatched
            if self.large_net_max_depth and len(groups) > self.large_net_max_depth:
                chunks = self._chunk_groups(groups, self.large_net_max_depth)
                for chunk in chunks:
                    batches.append(self._make_batch(component_index, "large_net_chunk", set(chunk)))
                    unbatched.difference_update(chunk)
            else:
                batches.append(self._make_batch(component_index, "large_net", groups))
                unbatched.difference_update(groups)

        while unbatched:
            seed_net = self._largest_net(component["nets"], unbatched)
            if seed_net is None:
                seed_group = min(unbatched, key=lambda name: self.group_rank[name])
                groups = {seed_group}
            else:
                groups = self.net_to_groups[seed_net] & unbatched
                if len(groups) > self.max_depth:
                    chunk = set(self._chunk_groups(groups, self.max_depth)[0])
                    batches.append(self._make_batch(component_index, "oversized_net_chunk", chunk))
                    unbatched.difference_update(chunk)
                    continue
            groups = self._absorb_neighbor_nets(component["nets"], groups, unbatched)
            batches.append(self._make_batch(component_index, "balanced", groups))
            unbatched.difference_update(groups)
        return batches

    def _largest_unbatched_large_net(self, net_ids: Iterable[str], unbatched: Set[str]) -> str | None:
        candidates = []
        for net_id in net_ids:
            net = self.nets_by_id[net_id]
            groups = self.net_to_groups[net_id] & unbatched
            if not groups:
                continue
            if len(net.pins) >= self.large_net_pin_threshold or len(groups) >= self.large_net_group_threshold:
                candidates.append((len(net.pins), len(groups), net_id))
        if not candidates:
            return None
        return max(candidates)[2]

    def _largest_net(self, net_ids: Iterable[str], unbatched: Set[str]) -> str | None:
        candidates = [
            (len(self.net_to_groups[net_id] & unbatched), len(self.nets_by_id[net_id].pins), net_id)
            for net_id in net_ids
            if self.net_to_groups[net_id] & unbatched
        ]
        if not candidates:
            return None
        return max(candidates)[2]

    def _absorb_neighbor_nets(
        self,
        component_net_ids: Iterable[str],
        initial_groups: Set[str],
        unbatched: Set[str],
    ) -> Set[str]:
        groups = set(initial_groups)
        changed = True
        while changed and len(groups) < self.target_depth:
            changed = False
            touching_nets = {
                net_id
                for group_name in groups
                for net_id in self.group_to_nets.get(group_name, set())
            } & set(component_net_ids)
            candidates = sorted(
                touching_nets,
                key=lambda net_id: (
                    -len(self.net_to_groups[net_id] & unbatched),
                    -len(self.nets_by_id[net_id].pins),
                    net_id,
                ),
            )
            for net_id in candidates:
                net = self.nets_by_id[net_id]
                if len(net.pins) > self.absorb_net_pin_threshold:
                    continue
                candidate_groups = groups | (self.net_to_groups[net_id] & unbatched)
                if len(candidate_groups) <= self.max_depth and len(candidate_groups) > len(groups):
                    groups = candidate_groups
                    changed = True
                    if len(groups) >= self.target_depth:
                        break
        return groups

    def _chunk_groups(self, group_names: Iterable[str], size: int) -> List[List[str]]:
        ordered = sorted(group_names, key=lambda name: self.group_rank[name])
        return [ordered[index : index + size] for index in range(0, len(ordered), size)]

    def _make_batch(self, component_index: int, kind: str, group_names: Set[str]) -> dict:
        nets = {
            net_id
            for group_name in group_names
            for net_id in self.group_to_nets.get(group_name, set())
        }
        complete_nets = [
            net_id for net_id in nets if self.net_to_groups[net_id].issubset(group_names)
        ]
        partial_nets = sorted(nets - set(complete_nets))
        return {
            "component_index": component_index,
            "batch_type": kind,
            "group_count": len(group_names),
            "pin_count": _group_pin_count(group_names, self.groups_by_name),
            "net_count": len(nets),
            "complete_net_count": len(complete_nets),
            "partial_net_count": len(partial_nets),
            "net_completeness_ratio": _ratio(len(complete_nets), len(nets)),
            "max_net_pin_count": max((len(self.nets_by_id[net_id].pins) for net_id in nets), default=0),
            "group_names": sorted(group_names, key=lambda name: self.group_rank[name]),
            "net_ids": sorted(nets),
            "partial_net_ids": partial_nets,
        }

    def _build_report(self, components: List[dict], batches: List[dict]) -> dict:
        compact_batches = [
            {key: value for key, value in batch.items() if key not in {"group_names", "net_ids", "partial_net_ids"}}
            for batch in batches
        ]
        group_counts = [batch["group_count"] for batch in batches]
        net_counts = [batch["net_count"] for batch in batches]
        completeness = [batch["net_completeness_ratio"] for batch in batches]
        return {
            "summary": {
                "total_group_count": len(self.groups_by_name),
                "total_net_count": len(self.nets_by_id),
                "component_count": len(components),
                "batch_count": len(batches),
                "max_batch_depth": max(group_counts, default=0),
                "average_batch_depth": _ratio(sum(group_counts), len(group_counts)),
                "max_batch_net_count": max(net_counts, default=0),
                "average_batch_net_count": _ratio(sum(net_counts), len(net_counts)),
                "average_net_completeness_ratio": _ratio(sum(completeness), len(completeness)),
                "large_net_batch_count": sum(1 for batch in batches if batch["batch_type"].startswith("large_net")),
                "partial_net_batch_count": sum(1 for batch in batches if batch["partial_net_count"] > 0),
            },
            "component_histogram": _histogram([len(component["groups"]) for component in components], 100),
            "batch_depth_histogram": _histogram(group_counts, 20),
            "batch_net_count_histogram": _histogram(net_counts, 20),
            "batch_type_histogram": self._batch_type_histogram(batches),
            "batches": compact_batches,
            "complexity": {
                "graph_build": "O(E)",
                "connected_components": "O(G + N + E)",
                "large_net_first_batching": "O(E log N) worst-case with sorted scans",
                "memory": "O(G + N + E)",
                "G": "pin-group count",
                "N": "net count",
                "E": "unique PinGroup-Net edges",
            },
        }

    @staticmethod
    def _batch_type_histogram(batches: Iterable[dict]) -> dict:
        histogram: Dict[str, int] = {}
        for batch in batches:
            key = batch["batch_type"]
            histogram[key] = histogram.get(key, 0) + 1
        return dict(sorted(histogram.items()))


def default_output_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return output_dir / f"mcts_batch_plan_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan MCTS batches from the PinGroup-Net graph.")
    parser.add_argument("--block", required=True, help="Path to block.json.")
    parser.add_argument("--pingroup", required=True, help="Path to pingroup.json.")
    parser.add_argument("--output", default=None, help="Output JSON path.")
    parser.add_argument("--output-dir", default=".", help="Output directory when --output is omitted.")
    parser.add_argument("--target-depth", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=256)
    parser.add_argument("--large-net-pin-threshold", type=int, default=500)
    parser.add_argument("--large-net-group-threshold", type=int, default=500)
    parser.add_argument("--large-net-max-depth", type=int, default=2500)
    parser.add_argument("--absorb-net-pin-threshold", type=int, default=100)
    parser.add_argument(
        "--fanout-reuse-sorting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether multi-fanout pins boost homology sorting reuse count.",
    )
    args = parser.parse_args()

    placedb = PlaceDB(args.block, args.pingroup)
    homology = HomologyManager(placedb, use_fanout_reuse_for_sorting=args.fanout_reuse_sorting)
    planner = BatchPlanner(
        placedb,
        homology,
        target_depth=args.target_depth,
        max_depth=args.max_depth,
        large_net_pin_threshold=args.large_net_pin_threshold,
        large_net_group_threshold=args.large_net_group_threshold,
        large_net_max_depth=args.large_net_max_depth,
        absorb_net_pin_threshold=args.absorb_net_pin_threshold,
    )
    report = planner.plan()
    report["input"] = {
        "block": str(Path(args.block)),
        "pingroup": str(Path(args.pingroup)),
    }
    report["parameters"] = {
        "target_depth": args.target_depth,
        "max_depth": args.max_depth,
        "large_net_pin_threshold": args.large_net_pin_threshold,
        "large_net_group_threshold": args.large_net_group_threshold,
        "large_net_max_depth": args.large_net_max_depth,
        "absorb_net_pin_threshold": args.absorb_net_pin_threshold,
        "fanout_reuse_sorting": args.fanout_reuse_sorting,
    }
    output_path = Path(args.output) if args.output else default_output_path(Path(args.output_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote MCTS batch plan: {output_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
