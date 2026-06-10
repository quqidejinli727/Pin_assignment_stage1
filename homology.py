"""Homology groups for reused blocks and pins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

from PlaceDB import Pin, PlaceDB
from geometry_utils import hpwl


@dataclass
class PinHomologyGroup:
    """表示必须保持同一抽象 segment 分配的一组同构 Pin。"""

    name: str
    pins: List[Pin]
    assigned: bool = False
    assigned_segment_id: str | None = None
    score: float = 0.0
    sort_reuse_count: int = 0

    @property
    def reuse_count(self) -> int:
        """返回该同构 Pin 组内实际 Pin 的数量。"""
        return len(self.pins)

    @property
    def max_pin_width(self) -> float:
        """返回组内最大 Pin 宽度，用于 segment 容量检查。"""
        return max((pin.width for pin in self.pins), default=0.0)

    @property
    def module_name(self) -> str:
        """返回该同构组所属的复用 block 类型名。"""
        return self.pins[0].parent_module if self.pins else ""


class HomologyManager:
    """构建同构 Pin 组，并按任务规则计算处理顺序。"""

    def __init__(self, placedb: PlaceDB, use_fanout_reuse_for_sorting: bool = True):
        """根据 PlaceDB 的 Pin 索引初始化同构组和反向查询表。"""
        self.placedb = placedb
        self.use_fanout_reuse_for_sorting = use_fanout_reuse_for_sorting
        self.pin_groups: Dict[str, PinHomologyGroup] = {
            name: PinHomologyGroup(name=name, pins=pins)
            for name, pins in placedb.pins_by_homology.items()
        }
        self.group_by_pin_full_name = {
            pin.full_name: group
            for group in self.pin_groups.values()
            for pin in group.pins
        }
        self.sorted_group_names = self._score_and_sort_groups()
        self.sort_rank = {name: index for index, name in enumerate(self.sorted_group_names)}

    def _score_and_sort_groups(self) -> List[str]:
        """计算同构组优先级分数，并返回排序后的组名列表。"""
        scored_names = []
        for group in self.pin_groups.values():
            group.sort_reuse_count = self._group_sort_reuse_count(group)
            reuse_score = float(group.sort_reuse_count)
            fanout_score = 0.0
            hpwl_score = self._group_hpwl_score(group)
            # Fanout score is intentionally present but weight 0 for v1.
            group.score = reuse_score * 1_000_000.0 + fanout_score * 0.0 + hpwl_score
            scored_names.append(group.name)
        return sorted(
            scored_names,
            key=lambda name: (
                -self.pin_groups[name].sort_reuse_count,
                -self.pin_groups[name].score,
                name,
            ),
        )

    def _group_sort_reuse_count(self, group: PinHomologyGroup) -> int:
        """Return the reuse count used only for homology processing order."""
        pin_successors: Dict[str, Set[str]] = {}
        pin_seen_count: Dict[str, int] = {}
        for pin in group.pins:
            pin_seen_count[pin.full_name] = pin_seen_count.get(pin.full_name, 0) + 1
            pin_successors.setdefault(pin.full_name, set()).update(pin.successors)

        if not self.use_fanout_reuse_for_sorting:
            return len(pin_seen_count)

        total = 0
        for full_name, seen_count in pin_seen_count.items():
            successor_count = len(pin_successors.get(full_name, set()))
            total += max(seen_count, successor_count, 1)
        return total

    def _group_hpwl_score(self, group: PinHomologyGroup) -> float:
        """用相关 Net 的模块质心 HPWL 估算该组的布线影响。"""
        related_nets = self.get_related_nets(group)
        total = 0.0
        for net in related_nets:
            points = [
                self.placedb.get_module(pin.parent_inst).get_centroid()
                for pin in net.pins
            ]
            total += hpwl(points)
        return total

    def get_related_nets(self, group: PinHomologyGroup):
        """返回包含该同构组任一 Pin 的所有 Net，自动去重。"""
        seen = set()
        nets = []
        for pin in group.pins:
            for net in self.placedb.iter_pin_nets(pin):
                if net.net_id not in seen:
                    seen.add(net.net_id)
                    nets.append(net)
        return nets

    def groups_for_pins(self, pins: Iterable[Pin]) -> List[PinHomologyGroup]:
        """根据一批 Pin 找到涉及的同构组，并按全局顺序排序。"""
        groups: Dict[str, PinHomologyGroup] = {}
        for pin in pins:
            group = self.group_by_pin_full_name[pin.full_name]
            groups[group.name] = group
        return sorted(groups.values(), key=lambda group: self.sort_rank[group.name])

    def unassigned_groups(self) -> List[PinHomologyGroup]:
        """返回尚未完成实际分配的同构组。"""
        return [
            self.pin_groups[name]
            for name in self.sorted_group_names
            if not self.pin_groups[name].assigned
        ]

    def fully_contained_groups(
        self,
        groups: Iterable[PinHomologyGroup],
        pin_full_names: Set[str],
    ) -> List[PinHomologyGroup]:
        """筛出组内所有 Pin 都包含在当前 pins_in 中的同构组。"""
        contained = []
        for group in groups:
            if all(pin.full_name in pin_full_names for pin in group.pins):
                contained.append(group)
        return contained

    def mark_assigned(self, group: PinHomologyGroup, segment_id: str) -> None:
        """标记同构组及组内所有 Pin 已分配到指定抽象 segment。"""
        group.assigned = True
        group.assigned_segment_id = segment_id
        for pin in group.pins:
            pin.assigned_segment_id = segment_id
