"""Deterministic cross-Net homology templates for runtime MCTS depth calibration."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable


TREE_PROFILE_VERSION = "mcts-tree-depth-profile-v0.1"

TREE_TEMPLATES = (
    "aligned_two_net",
    "remapped_two_net",
    "multi_instance_star",
    "multi_net_branched",
    "hierarchical_gateway_tree",
    "deferred_homology_tree",
)

DEFAULT_DEPTH_CYCLE = (
    2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 10,
    11, 13, 15, 18, 20,
    24,
)
DEEP_DEPTH_CYCLE = (24, 32, 48, 64)


def _derive_seed(base_seed: int, case_id: str, phase: str) -> int:
    material = f"{base_seed}|{case_id}|{phase}|{TREE_PROFILE_VERSION}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


@dataclass(frozen=True)
class MCTSTreeDepthProfile:
    enabled: bool = False
    group_budget_fraction: float = 0.60
    depth_cycle: tuple[int, ...] = DEFAULT_DEPTH_CYCLE
    max_depth: int = 100

    def __post_init__(self) -> None:
        if not 0.0 <= self.group_budget_fraction <= 0.90:
            raise ValueError("group_budget_fraction must be in [0, 0.90]")
        if any(depth < 2 or depth > self.max_depth for depth in self.depth_cycle):
            raise ValueError("profile depths must be between 2 and max_depth")
        if self.max_depth > 100:
            raise ValueError("current calibration phase forbids MCTS depth above 100")


def _group_key(pin: dict[str, Any]) -> str:
    return f"{pin['parent_module']}.{pin['pingroup_name']}"


def _full_name(pin: dict[str, Any]) -> str:
    return f"{pin['parent_inst']}.{pin['pingroup_name']}"


def _pin(module: dict[str, Any], name: str, weight: float) -> dict[str, Any]:
    return {
        "parent_inst": str(module["name"]),
        "parent_module": str(module["module_name"]),
        "pingroup_name": name,
        "scope": str(module["name"]).split("."),
        "successors": [],
        "width": 0.001,
        "source_connectivity_weight": float(weight),
        "source_connectivity_kind": "synthetic_tree_depth_low_pressure_prior",
        "width_provenance": "pending_capacity_calibration",
    }


def _chain(pins: list[dict[str, Any]]) -> None:
    for left, right in zip(pins, pins[1:]):
        left["successors"] = [_full_name(right)]


class CrossNetHomologyComposer:
    """Build isolated MCTS trees from one-hop cross-Net homology relationships."""

    def __init__(
        self,
        block: dict[str, Any],
        *,
        case_id: str,
        base_seed: int,
        source_sample_id: str,
        profile: MCTSTreeDepthProfile,
        source_weights: list[float],
    ) -> None:
        self.profile = profile
        self.source_sample_id = source_sample_id
        self.tree_profile_seed = _derive_seed(base_seed, case_id, "tree_profile")
        self.reuse_cluster_seed = _derive_seed(base_seed, case_id, "reuse_cluster")
        self.cross_net_homology_seed = _derive_seed(
            base_seed, case_id, "cross_net_homology"
        )
        self.rng = random.Random(self.cross_net_homology_seed)
        self.source_weights = sorted(float(value) for value in source_weights)
        self.current_weight = self.source_weights[0] if self.source_weights else 0.0005
        self.current_cost = 0.002
        self.parents = sorted(block.get("children", []), key=lambda item: str(item["name"]))
        self.leaves = sorted(
            [leaf for parent in self.parents for leaf in parent.get("children", [])],
            key=lambda item: str(item["name"]),
        )
        by_type: dict[str, list[dict[str, Any]]] = {}
        for leaf in self.leaves:
            by_type.setdefault(str(leaf["module_name"]), []).append(leaf)
        self.clusters = [
            tuple(sorted(instances, key=lambda item: str(item["name"])))
            for _, instances in sorted(by_type.items())
            if len(instances) >= 2
        ]
        if len(self.clusters) < 2:
            raise ValueError("cross-Net homology composition requires two reuse clusters")
        self.modules = sorted([*self.parents, *self.leaves], key=lambda item: str(item["name"]))
        self.module_type_load: Counter[str] = Counter()
        self.module_type_capacity: dict[str, float] = {}
        for module in self.modules:
            module_name = str(module["module_name"])
            if module_name in self.module_type_capacity:
                continue
            vertices = module.get("vertex", [])
            self.module_type_capacity[module_name] = max(
                sum(
                    math.hypot(
                        float(vertices[(index + 1) % len(vertices)][0]) - float(point[0]),
                        float(vertices[(index + 1) % len(vertices)][1]) - float(point[1]),
                    )
                    for index, point in enumerate(vertices)
                ),
                1e-9,
            )

    def _type_score(self, module_name: str) -> tuple[float, float, str]:
        return (
            self.module_type_load[module_name] / self.module_type_capacity[module_name],
            self.module_type_load[module_name],
            module_name,
        )

    def _select_tree_weight(self, tree_index: int, target_depth: int) -> tuple[float, float]:
        digest = hashlib.sha256(
            f"{self.tree_profile_seed}|weight-rank|{tree_index}".encode("utf-8")
        ).digest()
        quantile = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if target_depth > 20:
            # Wide components remain in the small bucket; capacity pressure is
            # spread by shallow/medium trees instead of multiplied by 21-64 groups.
            quantile *= 0.65
        if self.source_weights:
            index = min(int(quantile * len(self.source_weights)), len(self.source_weights) - 1)
            weight = self.source_weights[index]
        else:
            weight = 0.0005 + quantile * 0.011
        if quantile < 0.75:
            cost = 0.011
        elif quantile < 0.95:
            cost = 0.060
        else:
            cost = 0.225
        return weight, cost

    def _cluster(
        self,
        index: int,
        *,
        exclude: str | None = None,
        min_size: int = 2,
    ) -> tuple[dict[str, Any], ...]:
        choices = [
            cluster
            for cluster in self.clusters
            if str(cluster[0]["module_name"]) != exclude
            and len(cluster) >= min_size
        ]
        if not choices:
            choices = [
                cluster
                for cluster in self.clusters
                if str(cluster[0]["module_name"]) != exclude
            ]
        if not choices:
            raise ValueError("no reuse cluster is available")
        best = min(self._type_score(str(cluster[0]["module_name"]))[:2] for cluster in choices)
        choices = [
            cluster
            for cluster in choices
            if self._type_score(str(cluster[0]["module_name"]))[:2] == best
        ]
        return choices[(index + self.reuse_cluster_seed) % len(choices)]

    def _neighbor(self, token: str, index: int, excluded_instances: set[str]) -> dict[str, Any]:
        candidates = [
            module for module in self.modules if str(module["name"]) not in excluded_instances
        ]
        by_type: dict[str, list[dict[str, Any]]] = {}
        for module in candidates:
            by_type.setdefault(str(module["module_name"]), []).append(module)
        best = min(self._type_score(module_name)[:2] for module_name in by_type)
        tied_types = sorted(
            module_name
            for module_name in by_type
            if self._type_score(module_name)[:2] == best
        )
        digest = hashlib.sha256(
            f"{self.cross_net_homology_seed}|{token}|{index}".encode("utf-8")
        ).digest()
        module_type = tied_types[int.from_bytes(digest[:8], "big") % len(tied_types)]
        instances = sorted(by_type[module_type], key=lambda item: str(item["name"]))
        return instances[int.from_bytes(digest[8:16], "big") % len(instances)]

    def _record(
        self,
        net_index: int,
        template_id: str,
        template: str,
        target_depth: int,
        hub_group: str,
        pins: list[dict[str, Any]],
        role: str,
    ) -> dict[str, Any]:
        gateways = [
            str(pin["parent_inst"])
            for pin in pins
            if str(pin["parent_module"]).startswith("HIER_PARENT_")
        ]
        record = {
            "derived_net_index": net_index,
            "net_type": "cross_net_homology_tree",
            "net_kind": "dangling_singleton" if len(pins) == 1 else "connected",
            "tree_template_id": template_id,
            "reuse_template_id": template_id,
            "tree_template": template,
            "target_depth_bucket": (
                "2-10" if target_depth <= 10 else "11-20" if target_depth <= 20 else "21-100"
            ),
            "target_effective_depth": target_depth,
            "hub_homology_group": hub_group,
            "tree_net_role": role,
            "endpoint_hierarchy_paths": [str(pin["parent_inst"]).split(".") for pin in pins],
            "gateway_path": gateways,
            "source_sample_id": self.source_sample_id,
            "fully_synthetic": True,
            "topology_seed": self.cross_net_homology_seed,
            "tree_profile_seed": self.tree_profile_seed,
            "source_weight": self.current_weight,
            "source_weight_kind": "synthetic_tree_depth_low_pressure_prior",
            "width_provenance": "pending_capacity_calibration",
            "pin_count": len(pins),
        }
        if len(pins) == 1:
            record["reason"] = "missing_corresponding_connection"
            record["reuse_pattern"] = "deferred_homology_tree"
        return record

    def _compose_tree(
        self,
        tree_index: int,
        target_depth: int,
        first_net_index: int,
    ) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
        template = TREE_TEMPLATES[tree_index % len(TREE_TEMPLATES)]
        if target_depth == 2:
            template = "aligned_two_net"
        elif target_depth == 3:
            template = "remapped_two_net"
        elif template in {"aligned_two_net", "remapped_two_net"}:
            template = TREE_TEMPLATES[2 + (tree_index % 4)]
        token = f"MT{tree_index:08d}"
        self.current_weight, self.current_cost = self._select_tree_weight(
            tree_index, target_depth
        )
        hub_cluster = self._cluster(
            tree_index,
            min_size=(
                min(10, max(2, math.ceil((target_depth - 1) / 8)))
                if target_depth > 20
                else 2
            ),
        )
        hub_name = f"{token}_H"
        hub_group = f"{hub_cluster[0]['module_name']}.{hub_name}"
        nets: list[list[dict[str, Any]]] = []
        records: list[dict[str, Any]] = []

        if template == "aligned_two_net":
            peer_cluster = self._cluster(
                tree_index + 1, exclude=str(hub_cluster[0]["module_name"])
            )
            peer_name = f"{token}_P"
            for pair_index in range(2):
                pins = [
                    _pin(hub_cluster[pair_index], hub_name, self.current_weight),
                    _pin(peer_cluster[pair_index], peer_name, self.current_weight),
                ]
                _chain(pins)
                records.append(
                    self._record(
                        first_net_index + len(nets), token, template, target_depth,
                        hub_group, pins, "aligned",
                    )
                )
                nets.append(pins)
            group_names = {hub_group, f"{peer_cluster[0]['module_name']}.{peer_name}"}
        else:
            net_count = min(len(hub_cluster), target_depth - 1)
            group_names = {hub_group}
            peer_cluster: tuple[dict[str, Any], ...] | None = None
            peer_name: str | None = None
            if template == "multi_net_branched":
                net_count = min(len(hub_cluster), max(2, target_depth - 2))
                peer_cluster = self._cluster(
                    tree_index + 1, exclude=str(hub_cluster[0]["module_name"])
                )
                peer_name = f"{token}_B"
                group_names.add(f"{peer_cluster[0]['module_name']}.{peer_name}")
                unique_neighbor_total = target_depth - 2
            else:
                unique_neighbor_total = target_depth - 1
            base, extra = divmod(unique_neighbor_total, net_count)
            neighbor_index = 0
            excluded = {str(module["name"]) for module in hub_cluster}
            if peer_cluster:
                excluded.update(str(module["name"]) for module in peer_cluster)
            for net_offset in range(net_count):
                pins = [_pin(hub_cluster[net_offset], hub_name, self.current_weight)]
                if peer_cluster is not None and peer_name is not None and net_offset < 2:
                    pins.append(_pin(peer_cluster[net_offset], peer_name, self.current_weight))
                count = base + (1 if net_offset < extra else 0)
                for _ in range(count):
                    if template == "hierarchical_gateway_tree" and neighbor_index == 0:
                        module = self.parents[tree_index % len(self.parents)]
                    elif (
                        template == "deferred_homology_tree"
                        and neighbor_index == unique_neighbor_total - 1
                    ):
                        module = self._cluster(tree_index + 3)[0]
                    else:
                        module = self._neighbor(token, neighbor_index, excluded)
                    name = f"{token}_N{neighbor_index:03d}"
                    pins.append(_pin(module, name, self.current_weight))
                    group_names.add(f"{module['module_name']}.{name}")
                    neighbor_index += 1
                _chain(pins)
                records.append(
                    self._record(
                        first_net_index + len(nets), token, template, target_depth,
                        hub_group, pins, "hub_branch",
                    )
                )
                nets.append(pins)

            if template == "deferred_homology_tree":
                # Add a legal counterpart outside the hub-related Nets. The group is
                # visible but not fully covered during the hub's first runtime tree.
                source = nets[-1][-1]
                # Preserve homology by using the source module type only when possible.
                same_type = [
                    leaf
                    for leaf in self.leaves
                    if str(leaf["module_name"]) == str(source["parent_module"])
                    and str(leaf["name"]) != str(source["parent_inst"])
                ]
                if not same_type:
                    raise AssertionError("deferred template endpoint must use a reused module type")
                counterpart = _pin(same_type[0], str(source["pingroup_name"]), self.current_weight)
                records.append(
                    self._record(
                        first_net_index + len(nets), token, template, target_depth,
                        hub_group, [counterpart], "deferred_counterpart",
                    )
                )
                nets.append([counterpart])

        for group_name in group_names:
            module_name = group_name.rsplit(".", 1)[0]
            self.module_type_load[module_name] += self.current_cost
        tree = {
            "template_id": token,
            "template": template,
            "target_effective_depth": target_depth,
            "target_depth_bucket": (
                "2-10" if target_depth <= 10 else "11-20" if target_depth <= 20 else "21-100"
            ),
            "hub_homology_group": hub_group,
            "hub_instance_pins": [
                _full_name(pin)
                for net in nets
                for pin in net
                if _group_key(pin) == hub_group
            ],
            "related_net_indices": list(range(first_net_index, first_net_index + len(nets))),
            "neighbor_homology_groups": sorted(group_names - {hub_group}),
            "group_count": len(group_names),
            "topology_seed": self.cross_net_homology_seed,
            "source_sample_id": self.source_sample_id,
            "width_provenance": "pending_capacity_calibration",
        }
        if len(group_names) != target_depth:
            raise AssertionError(f"tree {token} has {len(group_names)} groups, expected {target_depth}")
        return nets, records, tree

    def compose(
        self,
        target_pingroup_count: int,
        *,
        first_net_index: int = 0,
    ) -> dict[str, Any]:
        if not self.profile.enabled:
            return {"pingroup": [], "lineage_nets": [], "trees": [], "group_count": 0}
        budget = int(target_pingroup_count * self.profile.group_budget_fraction)
        nets: list[list[dict[str, Any]]] = []
        records: list[dict[str, Any]] = []
        trees: list[dict[str, Any]] = []
        group_count = 0
        tree_index = 0
        deep_index = 0
        while True:
            depth = self.profile.depth_cycle[tree_index % len(self.profile.depth_cycle)]
            if depth > 20:
                depth = DEEP_DEPTH_CYCLE[deep_index % len(DEEP_DEPTH_CYCLE)]
                deep_index += 1
            if group_count + depth > budget:
                break
            tree_nets, tree_records, tree = self._compose_tree(
                tree_index, depth, first_net_index + len(nets)
            )
            nets.extend(tree_nets)
            records.extend(tree_records)
            trees.append(tree)
            group_count += depth
            tree_index += 1
        return {
            "pingroup": nets,
            "lineage_nets": records,
            "trees": trees,
            "group_count": group_count,
            "report": {
                "profile_version": TREE_PROFILE_VERSION,
                "profile": asdict(self.profile),
                "phase_seeds": {
                    "tree_profile_seed": self.tree_profile_seed,
                    "reuse_cluster_seed": self.reuse_cluster_seed,
                    "cross_net_homology_seed": self.cross_net_homology_seed,
                },
                "tree_count": len(trees),
                "group_count": group_count,
                "template_counts": dict(sorted(Counter(tree["template"] for tree in trees).items())),
                "target_depth_histogram": dict(
                    sorted(Counter(tree["target_effective_depth"] for tree in trees).items())
                ),
                "max_target_depth": max((tree["target_effective_depth"] for tree in trees), default=0),
            },
        }
