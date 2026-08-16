"""Independent deterministic hierarchical PinGroup/Net synthesizer."""

from __future__ import annotations

import hashlib
import math
import random
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from .hierarchy_composer import derive_phase_seed
from .mcts_tree_depth_profile import (
    CrossNetHomologyComposer,
    MCTSTreeDepthProfile,
)


NET_SYNTHESIZER_VERSION = "floorset-hierarchical-net-v0.3"

ARCHETYPES = (
    "local_leaf_to_leaf",
    "leaf_to_parent_gateway",
    "intra_parent_gateway",
    "cross_parent_leaf_to_leaf",
    "parent_to_parent",
    "cross_parent_gateway_chain",
    "multicast_fanout",
    "ancestor_cross_leaf",
)

REUSE_PATTERNS = (
    "aligned_pair",
    "dangling_counterpart",
    "remapped_counterpart",
)

ARCHETYPE_GROUP_COUNTS = {
    "local_leaf_to_leaf": 2,
    "leaf_to_parent_gateway": 2,
    "intra_parent_gateway": 3,
    "cross_parent_leaf_to_leaf": 2,
    "parent_to_parent": 2,
    "cross_parent_gateway_chain": 4,
    "multicast_fanout": 4,
    "ancestor_cross_leaf": 3,
}

CAPACITY_AWARE_ARCHETYPE_CYCLE = (
    "local_leaf_to_leaf",
    "cross_parent_leaf_to_leaf",
    "multicast_fanout",
    "leaf_to_parent_gateway",
    "local_leaf_to_leaf",
    "cross_parent_leaf_to_leaf",
    "multicast_fanout",
    "intra_parent_gateway",
    "local_leaf_to_leaf",
    "cross_parent_leaf_to_leaf",
    "multicast_fanout",
    "parent_to_parent",
    "local_leaf_to_leaf",
    "cross_parent_leaf_to_leaf",
    "multicast_fanout",
    "cross_parent_gateway_chain",
    "local_leaf_to_leaf",
    "cross_parent_leaf_to_leaf",
    "multicast_fanout",
    "ancestor_cross_leaf",
)


@dataclass(frozen=True)
class NetSynthesisConfig:
    case_id: str
    target_pingroup_count: int
    source_sample_id: str
    base_seed: int = 7
    seed_namespace: str | None = None
    reuse_template_fraction: float = 0.05
    aligned_fraction: float = 0.85
    dangling_fraction: float = 0.05
    remapped_fraction: float = 0.10
    capacity_aware_residual: bool | None = None
    mcts_tree_depth_profile: bool = False
    tree_profile_group_budget_fraction: float = 0.60

    def __post_init__(self) -> None:
        if self.target_pingroup_count < 32:
            raise ValueError("target_pingroup_count must be at least 32 for all archetypes")
        if not 0.0 < self.reuse_template_fraction <= 0.25:
            raise ValueError("reuse_template_fraction must be in (0, 0.25]")
        if not math.isclose(
            self.aligned_fraction + self.dangling_fraction + self.remapped_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("reuse pattern fractions must sum to 1")
        if not 0.0 <= self.tree_profile_group_budget_fraction <= 0.90:
            raise ValueError("tree_profile_group_budget_fraction must be in [0, 0.90]")

    @property
    def seed_case_id(self) -> str:
        return self.seed_namespace or self.case_id

    @property
    def topology_seed(self) -> int:
        return derive_phase_seed(self.base_seed, self.seed_case_id, "topology")

    @property
    def width_seed(self) -> int:
        return derive_phase_seed(self.base_seed, self.seed_case_id, "width")

    @property
    def residual_archetype_cycle(self) -> tuple[str, ...]:
        enabled = (
            self.target_pingroup_count >= 30_000
            if self.capacity_aware_residual is None
            else self.capacity_aware_residual
        )
        return CAPACITY_AWARE_ARCHETYPE_CYCLE if enabled else ARCHETYPES


def _group_key(pin: dict[str, Any]) -> str:
    return f"{pin['parent_module']}.{pin['pingroup_name']}"


def _full_name(pin: dict[str, Any]) -> str:
    return f"{pin['parent_inst']}.{pin['pingroup_name']}"


def _pin(
    module: dict[str, Any],
    pingroup_name: str,
    source_weight: float,
    source_kind: str,
) -> dict[str, Any]:
    return {
        "parent_inst": str(module["name"]),
        "parent_module": str(module["module_name"]),
        "pingroup_name": pingroup_name,
        "scope": str(module["name"]).split("."),
        "successors": [],
        "width": 0.001,
        "source_connectivity_weight": float(source_weight),
        "source_connectivity_kind": source_kind,
        "width_provenance": "pending_capacity_calibration",
    }


def _empirical_weights(source_pingroup: list[list[dict[str, Any]]]) -> list[float]:
    return sorted(
        float(pin["source_connectivity_weight"])
        for net in source_pingroup
        for pin in net
        if pin.get("source_connectivity_weight") is not None
        and math.isfinite(float(pin["source_connectivity_weight"]))
    )


class HierarchicalNetSynthesizer:
    def __init__(
        self,
        block: dict[str, Any],
        source_pingroup: list[list[dict[str, Any]]],
        config: NetSynthesisConfig,
    ) -> None:
        self.block = block
        self.config = config
        self.rng = random.Random(config.topology_seed)
        self.parents = sorted(block.get("children", []), key=lambda item: str(item["name"]))
        self.leaves_by_parent = {
            str(parent["name"]): sorted(
                parent.get("children", []), key=lambda item: str(item["name"])
            )
            for parent in self.parents
        }
        self.parent_by_name = {str(parent["name"]): parent for parent in self.parents}
        self.parent_for_leaf = {
            str(leaf["name"]): parent
            for parent in self.parents
            for leaf in self.leaves_by_parent[str(parent["name"])]
        }
        by_module: dict[str, list[dict[str, Any]]] = {}
        for leaves in self.leaves_by_parent.values():
            for leaf in leaves:
                by_module.setdefault(str(leaf["module_name"]), []).append(leaf)
        self.reused_pairs = [
            tuple(sorted(leaves, key=lambda item: str(item["name"])))
            for _, leaves in sorted(by_module.items())
            if len(leaves) >= 2
        ]
        self.local_reused_pairs = []
        for cluster in self.reused_pairs:
            by_parent: dict[str, list[dict[str, Any]]] = {}
            for leaf in cluster:
                parent = self.parent_for_leaf[str(leaf["name"])]
                by_parent.setdefault(str(parent["name"]), []).append(leaf)
            self.local_reused_pairs.extend(
                tuple(sorted(leaves, key=lambda item: str(item["name"])))
                for _, leaves in sorted(by_parent.items())
                if len(leaves) >= 2
            )
        if not self.local_reused_pairs:
            raise ValueError("hierarchy has no same-parent reused leaf pair")
        if len(self.reused_pairs) < 2:
            raise ValueError("hierarchy needs at least two reused leaf pairs")
        self.all_leaves = sorted(
            [leaf for leaves in self.leaves_by_parent.values() for leaf in leaves],
            key=lambda item: str(item["name"]),
        )
        self.module_capacity = self._module_capacities()
        self.planned_group_load: Counter[str] = Counter()
        self.planned_groups: set[str] = set()
        self.weights = _empirical_weights(source_pingroup)

    def _module_capacities(self) -> dict[str, float]:
        capacities: dict[str, float] = {}
        modules = [*self.parents, *self.all_leaves]
        for module in modules:
            module_name = str(module["module_name"])
            if module_name in capacities:
                continue
            vertices = module.get("vertex", [])
            perimeter = sum(
                math.hypot(
                    float(vertices[(index + 1) % len(vertices)][0]) - float(point[0]),
                    float(vertices[(index + 1) % len(vertices)][1]) - float(point[1]),
                )
                for index, point in enumerate(vertices)
            )
            capacities[module_name] = max(perimeter, 1e-9)
        return capacities

    def _load_score(self, module: dict[str, Any]) -> tuple[float, int, str]:
        module_name = str(module["module_name"])
        capacity = self.module_capacity[module_name]
        return (
            self.planned_group_load[module_name] / capacity,
            self.planned_group_load[module_name],
            str(module["name"]),
        )

    def _choose_module(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            raise ValueError("no endpoint candidate is available")
        best_score = min(self._load_score(module)[:2] for module in candidates)
        tied = [
            module for module in candidates if self._load_score(module)[:2] == best_score
        ]
        if self.config.mcts_tree_depth_profile:
            by_type: dict[str, list[dict[str, Any]]] = {}
            for module in tied:
                by_type.setdefault(str(module["module_name"]), []).append(module)
            module_types = sorted(by_type)
            chosen_type = module_types[self.rng.randrange(len(module_types))]
            instances = sorted(by_type[chosen_type], key=lambda item: str(item["name"]))
            return instances[self.rng.randrange(len(instances))]
        return tied[self.rng.randrange(len(tied))]

    def _choose_reused_pair(
        self,
        candidates: list[tuple[dict[str, Any], ...]],
        *,
        excluded_module_names: set[str] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        excluded = excluded_module_names or set()
        available = [
            pair for pair in candidates if str(pair[0]["module_name"]) not in excluded
        ]
        if not available:
            raise ValueError("no reused module pair is available")
        chosen = self._choose_module([pair[0] for pair in available])
        chosen_name = str(chosen["module_name"])
        return next(pair for pair in available if str(pair[0]["module_name"]) == chosen_name)

    def _register_groups(self, nets: list[list[dict[str, Any]]]) -> None:
        for net in nets:
            for pin in net:
                key = _group_key(pin)
                if key in self.planned_groups:
                    continue
                self.planned_groups.add(key)
                cost = 1.0
                if self.config.mcts_tree_depth_profile:
                    weight = float(pin.get("source_connectivity_weight", 0.0))
                    quantile = (
                        bisect_right(self.weights, weight) / len(self.weights)
                        if self.weights
                        else 0.5
                    )
                    cost = 0.011 if quantile < 0.75 else 0.060 if quantile < 0.95 else 0.225
                self.planned_group_load[str(pin["parent_module"])] += cost

    def _weight(self, net_index: int) -> tuple[float, str]:
        if self.weights:
            position = int(
                hashlib.sha256(
                    f"{self.config.topology_seed}|weight|{net_index}".encode("utf-8")
                ).hexdigest()[:16],
                16,
            ) % len(self.weights)
            return self.weights[position], "floorset_empirical_weight_sample"
        raw = int(
            hashlib.sha256(
                f"{self.config.topology_seed}|synthetic-weight|{net_index}".encode("utf-8")
            ).hexdigest()[:16],
            16,
        ) / float(1 << 64)
        return 0.0005 + raw * 0.011, "synthetic_topology_weight_prior"

    def _different_parents(self) -> tuple[dict[str, Any], dict[str, Any]]:
        left = self._choose_module(self.parents)
        right = self._choose_module([parent for parent in self.parents if parent is not left])
        return left, right

    def _two_leaves(self, parent: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        leaves = self.leaves_by_parent[str(parent["name"])]
        left = self._choose_module(leaves)
        right = self._choose_module([leaf for leaf in leaves if leaf is not left])
        return left, right

    @staticmethod
    def _chain(pins: list[dict[str, Any]]) -> None:
        for left, right in zip(pins, pins[1:]):
            left["successors"] = [_full_name(right)]

    def _build(self, archetype: str, net_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        token = f"HN{net_index:08d}"
        weight, weight_kind = self._weight(net_index)
        gateways: list[str] = []
        if archetype == "local_leaf_to_leaf":
            pair = self._choose_reused_pair(self.local_reused_pairs)
            pins = [
                _pin(pair[0], f"{token}_A", weight, weight_kind),
                _pin(pair[1], f"{token}_B", weight, weight_kind),
            ]
            self._chain(pins)
        elif archetype == "leaf_to_parent_gateway":
            parent = self.parents[net_index % len(self.parents)]
            leaf = self._choose_module(self.leaves_by_parent[str(parent["name"])])
            pins = [
                _pin(leaf, f"{token}_L", weight, weight_kind),
                _pin(parent, f"{token}_GW", weight, weight_kind),
            ]
            gateways = [str(parent["name"])]
            self._chain(pins)
        elif archetype == "intra_parent_gateway":
            parent = self.parents[net_index % len(self.parents)]
            left, right = self._two_leaves(parent)
            pins = [
                _pin(left, f"{token}_A", weight, weight_kind),
                _pin(parent, f"{token}_GW", weight, weight_kind),
                _pin(right, f"{token}_B", weight, weight_kind),
            ]
            gateways = [str(parent["name"])]
            self._chain(pins)
        elif archetype == "cross_parent_leaf_to_leaf":
            left_parent, right_parent = self._different_parents()
            pins = [
                _pin(
                    self._choose_module(self.leaves_by_parent[str(left_parent["name"])]),
                    f"{token}_A",
                    weight,
                    weight_kind,
                ),
                _pin(
                    self._choose_module(self.leaves_by_parent[str(right_parent["name"])]),
                    f"{token}_B",
                    weight,
                    weight_kind,
                ),
            ]
            self._chain(pins)
        elif archetype == "parent_to_parent":
            left_parent, right_parent = self._different_parents()
            pins = [
                _pin(left_parent, f"{token}_PA", weight, weight_kind),
                _pin(right_parent, f"{token}_PB", weight, weight_kind),
            ]
            gateways = [str(left_parent["name"]), str(right_parent["name"])]
            self._chain(pins)
        elif archetype == "cross_parent_gateway_chain":
            left_parent, right_parent = self._different_parents()
            left_leaf = self._choose_module(self.leaves_by_parent[str(left_parent["name"])])
            right_leaf = self._choose_module(self.leaves_by_parent[str(right_parent["name"])])
            pins = [
                _pin(left_leaf, f"{token}_LA", weight, weight_kind),
                _pin(left_parent, f"{token}_GWA", weight, weight_kind),
                _pin(right_parent, f"{token}_GWB", weight, weight_kind),
                _pin(right_leaf, f"{token}_LB", weight, weight_kind),
            ]
            gateways = [str(left_parent["name"]), str(right_parent["name"])]
            self._chain(pins)
        elif archetype == "multicast_fanout":
            source_parent, target_parent = self._different_parents()
            source = self._choose_module(self.leaves_by_parent[str(source_parent["name"])])
            target_a, target_b = self._two_leaves(target_parent)
            target_c = self._choose_module(
                [
                    leaf
                    for leaf in self.leaves_by_parent[str(source_parent["name"])]
                    if leaf is not source
                ]
            )
            pins = [
                _pin(source, f"{token}_SRC", weight, weight_kind),
                _pin(target_a, f"{token}_T0", weight, weight_kind),
                _pin(target_b, f"{token}_T1", weight, weight_kind),
                _pin(target_c, f"{token}_T2", weight, weight_kind),
            ]
            pins[0]["successors"] = [_full_name(pin) for pin in pins[1:]]
        elif archetype == "ancestor_cross_leaf":
            left_parent, right_parent = self._different_parents()
            left_leaf = self._choose_module(self.leaves_by_parent[str(left_parent["name"])])
            right_leaf = self._choose_module(self.leaves_by_parent[str(right_parent["name"])])
            pins = [
                _pin(left_leaf, f"{token}_LA", weight, weight_kind),
                _pin(left_parent, f"{token}_ANCESTOR", weight, weight_kind),
                _pin(right_leaf, f"{token}_LB", weight, weight_kind),
            ]
            gateways = [str(left_parent["name"])]
            self._chain(pins)
        else:
            raise ValueError(f"unknown archetype {archetype!r}")
        return pins, {
            "derived_net_index": net_index,
            "net_type": archetype,
            "endpoint_hierarchy_paths": [str(pin["parent_inst"]).split(".") for pin in pins],
            "gateway_path": gateways,
            "source_sample_id": self.config.source_sample_id,
            "fully_synthetic": True,
            "topology_seed": self.config.topology_seed,
            "source_weight": weight,
            "source_weight_kind": weight_kind,
            "width_provenance": "pending_capacity_calibration",
            "pin_count": len(pins),
        }

    def _remap_endpoint(
        self,
        a2: dict[str, Any],
        template_index: int,
        excluded_instances: set[str],
    ) -> tuple[dict[str, Any], str, list[str]]:
        mode = ("local", "cross_parent", "gateway")[template_index % 3]
        source_parent = self.parent_for_leaf[str(a2["name"])]
        if mode == "gateway":
            return source_parent, mode, [str(source_parent["name"])]
        if mode == "local":
            candidates = [
                leaf
                for leaf in self.leaves_by_parent[str(source_parent["name"])]
                if str(leaf["name"]) not in excluded_instances
            ]
        else:
            candidates = [
                leaf
                for leaf in self.all_leaves
                if self.parent_for_leaf[str(leaf["name"])] is not source_parent
                and str(leaf["name"]) not in excluded_instances
            ]
        if not candidates:
            raise ValueError(f"no {mode} remap endpoint is available")
        endpoint = self._choose_module(candidates)
        return endpoint, mode, []

    def _reuse_net_record(
        self,
        *,
        net_index: int,
        pins: list[dict[str, Any]],
        template_id: str,
        pattern: str,
        role: str,
        weight: float,
        weight_kind: str,
        net_kind: str = "connected",
        remap_mode: str | None = None,
        gateways: list[str] | None = None,
    ) -> dict[str, Any]:
        record = {
            "derived_net_index": net_index,
            "net_type": "reuse_connectivity",
            "net_kind": net_kind,
            "reuse_template_id": template_id,
            "reuse_pattern": pattern,
            "reuse_role": role,
            "endpoint_hierarchy_paths": [str(pin["parent_inst"]).split(".") for pin in pins],
            "gateway_path": gateways or [],
            "source_sample_id": self.config.source_sample_id,
            "fully_synthetic": True,
            "topology_seed": self.config.topology_seed,
            "source_weight": weight,
            "source_weight_kind": weight_kind,
            "width_provenance": "pending_capacity_calibration",
            "pin_count": len(pins),
        }
        if net_kind == "dangling_singleton":
            record["reason"] = "missing_corresponding_connection"
        if remap_mode is not None:
            record["remap_mode"] = remap_mode
        return record

    def _build_reuse_template(
        self,
        pattern: str,
        template_index: int,
        first_net_index: int,
    ) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
        if pattern not in REUSE_PATTERNS:
            raise ValueError(f"unknown reuse pattern {pattern!r}")
        template_id = f"RT{template_index:08d}"
        a_pair = self._choose_reused_pair(self.reused_pairs)
        b_pair = self._choose_reused_pair(
            self.reused_pairs,
            excluded_module_names={str(a_pair[0]["module_name"])},
        )
        a1, a2 = a_pair[0], a_pair[1]
        b1, b2 = b_pair[0], b_pair[1]
        weight, weight_kind = self._weight(first_net_index)
        a_name, b_name = f"{template_id}_A", f"{template_id}_B"
        nets: list[list[dict[str, Any]]] = []
        records: list[dict[str, Any]] = []

        def add(
            pins: list[dict[str, Any]],
            role: str,
            *,
            net_kind: str = "connected",
            remap_mode: str | None = None,
            gateways: list[str] | None = None,
        ) -> None:
            net_index = first_net_index + len(nets)
            if len(pins) >= 2:
                self._chain(pins)
            nets.append(pins)
            records.append(
                self._reuse_net_record(
                    net_index=net_index,
                    pins=pins,
                    template_id=template_id,
                    pattern=pattern,
                    role=role,
                    weight=weight,
                    weight_kind=weight_kind,
                    net_kind=net_kind,
                    remap_mode=remap_mode,
                    gateways=gateways,
                )
            )

        add(
            [_pin(a1, a_name, weight, weight_kind), _pin(b1, b_name, weight, weight_kind)],
            "primary",
        )
        if pattern == "aligned_pair":
            add(
                [_pin(a2, a_name, weight, weight_kind), _pin(b2, b_name, weight, weight_kind)],
                "corresponding",
            )
        elif pattern == "dangling_counterpart":
            add(
                [_pin(a2, a_name, weight, weight_kind)],
                "a_counterpart",
                net_kind="dangling_singleton",
            )
            add(
                [_pin(b2, b_name, weight, weight_kind)],
                "b_counterpart",
                net_kind="dangling_singleton",
            )
        else:
            endpoint, remap_mode, gateways = self._remap_endpoint(
                a2,
                template_index,
                {str(a1["name"]), str(a2["name"]), str(b1["name"]), str(b2["name"])},
            )
            add(
                [
                    _pin(a2, a_name, weight, weight_kind),
                    _pin(endpoint, f"{template_id}_C", weight, weight_kind),
                ],
                "remapped",
                remap_mode=remap_mode,
                gateways=gateways,
            )
            add(
                [_pin(b2, b_name, weight, weight_kind)],
                "b_counterpart",
                net_kind="dangling_singleton",
                remap_mode=remap_mode,
            )
        return nets, records

    def _reuse_pattern_schedule(self) -> list[str]:
        template_count = max(
            len(REUSE_PATTERNS),
            int(round(self.config.target_pingroup_count * self.config.reuse_template_fraction)),
        )
        dangling = max(1, int(round(template_count * self.config.dangling_fraction)))
        remapped = max(1, int(round(template_count * self.config.remapped_fraction)))
        aligned = template_count - dangling - remapped
        if aligned < 1:
            raise ValueError("reuse template budget is too small")
        counts = {
            "aligned_pair": aligned,
            "dangling_counterpart": dangling,
            "remapped_counterpart": remapped,
        }
        schedule: list[str] = []
        remaining = dict(counts)
        while any(remaining.values()):
            for pattern in REUSE_PATTERNS:
                if remaining[pattern]:
                    schedule.append(pattern)
                    remaining[pattern] -= 1
        return schedule

    def synthesize(self) -> dict[str, Any]:
        nets: list[list[dict[str, Any]]] = []
        lineage: list[dict[str, Any]] = []
        groups: set[str] = set()
        cross_net_report: dict[str, Any] = {
            "enabled": False,
            "trees": [],
            "group_count": 0,
        }

        def append(archetype: str) -> int:
            pins, record = self._build(archetype, len(nets))
            new_groups = {_group_key(pin) for pin in pins} - groups
            if not new_groups:
                raise ValueError(f"net {len(nets)} contributes no new PinGroups")
            nets.append(pins)
            groups.update(new_groups)
            record["added_pingroup_count"] = len(new_groups)
            lineage.append(record)
            self._register_groups([pins])
            return len(new_groups)

        if self.config.mcts_tree_depth_profile:
            composer = CrossNetHomologyComposer(
                self.block,
                case_id=self.config.seed_case_id,
                base_seed=self.config.base_seed,
                source_sample_id=self.config.source_sample_id,
                profile=MCTSTreeDepthProfile(
                    enabled=True,
                    group_budget_fraction=self.config.tree_profile_group_budget_fraction,
                ),
                source_weights=self.weights,
            )
            composed = composer.compose(self.config.target_pingroup_count)
            for net, record in zip(composed["pingroup"], composed["lineage_nets"]):
                record["added_pingroup_count"] = len(
                    {_group_key(pin) for pin in net} - groups
                )
                nets.append(net)
                groups.update(_group_key(pin) for pin in net)
                lineage.append(record)
            self._register_groups(composed["pingroup"])
            cross_net_report = {
                "enabled": True,
                **composed["report"],
                "trees": composed["trees"],
            }

        for template_index, pattern in enumerate(self._reuse_pattern_schedule()):
            template_nets, template_records = self._build_reuse_template(
                pattern, template_index, len(nets)
            )
            template_groups = {
                _group_key(pin) for net in template_nets for pin in net
            }
            new_groups = template_groups - groups
            if len(groups) + len(new_groups) > self.config.target_pingroup_count:
                break
            for net, record in zip(template_nets, template_records):
                nets.append(net)
                record["added_pingroup_count"] = len(
                    {_group_key(pin) for pin in net} - groups
                )
                groups.update(_group_key(pin) for pin in net)
                lineage.append(record)
            self._register_groups(template_nets)

        for archetype in ARCHETYPES:
            append(archetype)
        next_archetype = 0
        while len(groups) < self.config.target_pingroup_count:
            remaining = self.config.target_pingroup_count - len(groups)
            cycle = self.config.residual_archetype_cycle
            candidates = [
                cycle[(next_archetype + offset) % len(cycle)]
                for offset in range(len(cycle))
            ]
            archetype = next(
                (
                    name
                    for name in candidates
                    if ARCHETYPE_GROUP_COUNTS[name] <= remaining
                    and remaining - ARCHETYPE_GROUP_COUNTS[name] != 1
                ),
                None,
            )
            if archetype is None:
                raise ValueError(f"residual PinGroup target {remaining} is not representable")
            candidate, record = self._build(archetype, len(nets))
            new_groups = {_group_key(pin) for pin in candidate} - groups
            nets.append(candidate)
            groups.update(new_groups)
            record["added_pingroup_count"] = len(new_groups)
            lineage.append(record)
            self._register_groups([candidate])
            next_archetype += 1

        validation = validate_hierarchical_nets(nets, lineage, self.config.target_pingroup_count)
        if not validation["valid"]:
            raise ValueError(f"synthetic hierarchical nets are invalid: {validation['errors'][:5]!r}")
        distribution = Counter(record["net_type"] for record in lineage)
        reuse_report = build_reuse_connectivity_report(nets, lineage)
        report = {
            "synthesizer_version": NET_SYNTHESIZER_VERSION,
            "config": asdict(self.config),
            "phase_seeds": {
                "topology_seed": self.config.topology_seed,
                "width_seed": self.config.width_seed,
                **cross_net_report.get("phase_seeds", {}),
            },
            "net_count": len(nets),
            "pin_count": sum(len(net) for net in nets),
            "pingroup_count": len(groups),
            "net_type_counts": dict(sorted(distribution.items())),
            "source_weight_prior_count": len(self.weights),
            "validation": validation,
            "reuse_connectivity": reuse_report,
            "cross_net_homology": cross_net_report,
        }
        return {
            "pingroup": nets,
            "lineage_nets": lineage,
            "report": report,
            "reuse_connectivity_report": reuse_report,
            "cross_net_homology_report": cross_net_report,
        }


def build_reuse_connectivity_report(
    pingroup: list[list[dict[str, Any]]],
    lineage_nets: list[dict[str, Any]],
) -> dict[str, Any]:
    records = {int(item["derived_net_index"]): item for item in lineage_nets}
    template_patterns: dict[str, str] = {}
    pattern_net_counts: Counter[str] = Counter()
    remap_counts: Counter[str] = Counter()
    singleton_indices: list[int] = []
    group_nets: dict[str, set[int]] = {}
    group_partners: dict[str, set[str]] = {}
    group_widths: dict[str, set[float]] = {}
    full_name_counts: Counter[str] = Counter()
    for net_index, net in enumerate(pingroup):
        record = records.get(net_index, {})
        pattern = record.get("reuse_pattern")
        template_id = record.get("reuse_template_id")
        if pattern and template_id:
            template_patterns[str(template_id)] = str(pattern)
            pattern_net_counts[str(pattern)] += 1
            if pattern == "remapped_counterpart" and record.get("reuse_role") == "remapped":
                remap_counts[str(record.get("remap_mode", "unknown"))] += 1
        if record.get("net_kind") == "dangling_singleton":
            singleton_indices.append(net_index)
        names = [_full_name(pin) for pin in net]
        for pin in net:
            key = _group_key(pin)
            full_name = _full_name(pin)
            full_name_counts[full_name] += 1
            group_nets.setdefault(key, set()).add(net_index)
            group_widths.setdefault(key, set()).add(float(pin.get("width", 0.0)))
            group_partners.setdefault(key, set()).update(name for name in names if name != full_name)
    pattern_templates = Counter(template_patterns.values())
    partner_counts = sorted(
        len(partners)
        for group, partners in group_partners.items()
        if len(group_nets.get(group, set())) > 1
    )
    return {
        "reuse_template_count": len(template_patterns),
        "pattern_template_counts": dict(sorted(pattern_templates.items())),
        "pattern_net_counts": dict(sorted(pattern_net_counts.items())),
        "pattern_template_ratios": {
            pattern: count / len(template_patterns) if template_patterns else 0.0
            for pattern, count in sorted(pattern_templates.items())
        },
        "remapped_endpoint_counts": dict(sorted(remap_counts.items())),
        "expected_singleton_net_count": len(singleton_indices),
        "actual_singleton_net_count": sum(len(net) == 1 for net in pingroup),
        "singleton_net_indices": singleton_indices,
        "homology_group_multi_net_count": sum(len(nets) > 1 for nets in group_nets.values()),
        "same_group_distinct_partner_count": {
            "count": len(partner_counts),
            "min": min(partner_counts) if partner_counts else None,
            "max": max(partner_counts) if partner_counts else None,
        },
        "homology_width_mismatch_count": sum(len(widths) != 1 for widths in group_widths.values()),
        "full_pin_multi_net_violation_count": sum(count != 1 for count in full_name_counts.values()),
    }


def validate_hierarchical_nets(
    pingroup: list[list[dict[str, Any]]],
    lineage_nets: list[dict[str, Any]],
    expected_pingroup_count: int | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    seen_full_names: set[str] = set()
    group_widths: dict[str, set[float]] = {}
    edge_count = 0
    gateway_net_count = 0
    records = {int(item["derived_net_index"]): item for item in lineage_nets}
    singleton_indices: list[int] = []
    group_net_indices: dict[str, set[int]] = {}
    template_connected_groups: dict[str, set[str]] = {}
    for net_index, net in enumerate(pingroup):
        record = records.get(net_index, {})
        full_names = {_full_name(pin) for pin in net}
        if len(full_names) != len(net):
            errors.append({"code": "duplicate_pin_in_net", "detail": str(net_index)})
        overlap = seen_full_names & full_names
        if overlap:
            errors.append({"code": "duplicate_pin_full_name", "detail": min(overlap)})
        seen_full_names.update(full_names)
        adjacency = {name: [] for name in full_names}
        for pin in net:
            group_key = _group_key(pin)
            group_widths.setdefault(group_key, set()).add(float(pin.get("width", 0.0)))
            group_net_indices.setdefault(group_key, set()).add(net_index)
            source = _full_name(pin)
            for successor in pin.get("successors", []):
                if successor not in full_names:
                    errors.append(
                        {"code": "missing_successor", "detail": f"{source}->{successor}"}
                    )
                    continue
                if successor == source:
                    errors.append({"code": "successor_self_loop", "detail": source})
                    continue
                adjacency[source].append(successor)
                edge_count += 1
        state: dict[str, int] = {}

        def visit(node: str) -> bool:
            if state.get(node) == 1:
                return False
            if state.get(node) == 2:
                return True
            state[node] = 1
            if any(not visit(target) for target in adjacency[node]):
                return False
            state[node] = 2
            return True

        if any(not visit(node) for node in adjacency if state.get(node, 0) == 0):
            errors.append({"code": "successor_cycle", "detail": str(net_index)})
        if len(net) == 1:
            singleton_indices.append(net_index)
            if record.get("net_kind") != "dangling_singleton":
                errors.append({"code": "unprovenanced_singleton", "detail": str(net_index)})
            if record.get("reason") != "missing_corresponding_connection":
                errors.append({"code": "invalid_singleton_reason", "detail": str(net_index)})
            if record.get("reuse_pattern") not in {
                "dangling_counterpart",
                "remapped_counterpart",
                "deferred_homology_tree",
            }:
                errors.append({"code": "invalid_singleton_pattern", "detail": str(net_index)})
            if any(adjacency.values()):
                errors.append({"code": "singleton_with_successor", "detail": str(net_index)})
        elif not any(adjacency.values()):
            errors.append({"code": "net_without_successor_edge", "detail": str(net_index)})
        template_id = record.get("reuse_template_id")
        if template_id and len(net) >= 2:
            template_connected_groups.setdefault(str(template_id), set()).update(
                _group_key(pin) for pin in net
            )

    for net_index in singleton_indices:
        record = records.get(net_index, {})
        net = pingroup[net_index]
        if not net:
            errors.append({"code": "empty_singleton", "detail": str(net_index)})
            continue
        group = _group_key(net[0])
        if len(group_net_indices.get(group, set())) < 2:
            errors.append({"code": "orphan_singleton", "detail": group})
        template_id = str(record.get("reuse_template_id", ""))
        if group not in template_connected_groups.get(template_id, set()):
            errors.append({"code": "singleton_without_connected_counterpart", "detail": group})

    for group, widths in group_widths.items():
        if len(widths) != 1:
            errors.append({"code": "homology_width_mismatch", "detail": group})
    type_counts = Counter(record.get("net_type") for record in lineage_nets)
    for required in ARCHETYPES:
        if type_counts.get(required, 0) == 0:
            errors.append({"code": "missing_net_type", "detail": required})
    for record in lineage_nets:
        if record.get("gateway_path"):
            gateway_net_count += 1
    pingroup_count = len(group_widths)
    if expected_pingroup_count is not None and pingroup_count != expected_pingroup_count:
        errors.append(
            {
                "code": "pingroup_count_mismatch",
                "detail": f"{pingroup_count}!={expected_pingroup_count}",
            }
        )
    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "net_count": len(pingroup),
        "pin_count": sum(len(net) for net in pingroup),
        "pingroup_count": pingroup_count,
        "successor_edge_count": edge_count,
        "missing_successor_count": sum(error["code"] == "missing_successor" for error in errors),
        "successor_cycle_count": sum(error["code"] == "successor_cycle" for error in errors),
        "gateway_net_count": gateway_net_count,
        "net_type_counts": dict(sorted(type_counts.items())),
        "singleton_net_count": len(singleton_indices),
        "unprovenanced_singleton_net_count": sum(
            error["code"] == "unprovenanced_singleton" for error in errors
        ),
        "orphan_singleton_count": sum(error["code"] == "orphan_singleton" for error in errors),
    }
