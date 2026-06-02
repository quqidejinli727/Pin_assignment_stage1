"""Segment abstraction and capacity management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from PlaceDB import Module, Pin, PlaceDB
from geometry_utils import (
    Point,
    align_vertices_to_reference,
    canonical_vertex_mapping,
    segment_length,
    segment_midpoint,
)
from segment_subdivision import EdgeSubdivision, interpolate_edge, subdivision_specs


@dataclass
class SegmentInstance:
    """表示某个模块实例上的真实 segment。"""

    segment_id: str
    module_inst: str
    start: Point
    end: Point
    midpoint: Point
    assigned_pins: List[str] = field(default_factory=list)


@dataclass
class AbstractSegment:
    """表示同一 module_name 复用组共享的抽象母 segment。"""

    segment_id: str
    module_name: str
    index: int
    edge_id: int
    child_index: int
    capacity: float
    used_width: float = 0.0
    assigned_groups: List[str] = field(default_factory=list)
    instances: Dict[str, SegmentInstance] = field(default_factory=dict)

    @property
    def remaining_capacity(self) -> float:
        """返回当前抽象 segment 的剩余容量。"""
        return self.capacity - self.used_width

    def can_fit(self, width: float) -> bool:
        """判断给定宽度是否还能放入该 segment。"""
        return self.remaining_capacity + 1e-9 >= width


class SegmentUsage:
    """MCTS 节点中使用的轻量级 segment 占用快照。"""

    def __init__(self, used_width: Dict[str, float] | None = None):
        """初始化一份 segment_id 到已用宽度的映射。"""
        self.used_width = dict(used_width or {})

    def clone(self) -> "SegmentUsage":
        """复制当前占用快照，供 MCTS 子节点独立修改。"""
        return SegmentUsage(self.used_width)

    def can_assign(self, segment: AbstractSegment, width: float) -> bool:
        """在快照状态下判断某 segment 是否可继续分配。"""
        used = self.used_width.get(segment.segment_id, segment.used_width)
        return used + width <= segment.capacity - 1e-9

    def assign(self, segment: AbstractSegment, width: float) -> None:
        """在快照状态下记录一次 segment 宽度占用。"""
        used = self.used_width.get(segment.segment_id, segment.used_width)
        self.used_width[segment.segment_id] = used + width


class SegmentManager:
    """构建并管理抽象 segment、实例 segment 和容量使用情况。"""

    def __init__(self, placedb: PlaceDB, max_segment_length: float | None = None):
        """根据 PlaceDB 中带 Pin 的模块创建 segment 结构。"""
        self.placedb = placedb
        self.max_segment_length = max_segment_length
        self.abstract_segments: Dict[str, AbstractSegment] = {}
        self.segments_by_module_name: Dict[str, List[AbstractSegment]] = {}
        self.instance_lookup: Dict[Tuple[str, str], SegmentInstance] = {}
        self._build_segments()

    def _build_segments(self) -> None:
        """为所有包含 Pin 的 module_name 复用组创建 segment。"""
        module_names_with_pins = {
            module.module_name
            for module in self.placedb.all_modules_list
            if module.pin_list
        }
        for module_name in sorted(module_names_with_pins):
            modules = [
                module
                for module in self.placedb.modules_by_module_name.get(module_name, [])
                if module.vertex
            ]
            if not modules:
                continue
            self._build_module_group_segments(module_name, modules)

    def _build_module_group_segments(self, module_name: str, modules: List[Module]) -> None:
        """为一个 module_name 复用组创建抽象 segment 和实例映射。"""
        reference = modules[0]
        _, reference_vertices = canonical_vertex_mapping(
            reference.vertex,
            int(reference.direction),
        )
        split_specs = self.split_segments(reference_vertices)

        for index, spec in enumerate(split_specs):
            start, end = self._segment_coordinates(reference_vertices, spec)
            segment_id = f"{module_name}:S{index}"
            abstract = AbstractSegment(
                segment_id=segment_id,
                module_name=module_name,
                index=index,
                edge_id=spec.edge_id,
                child_index=spec.child_index,
                capacity=segment_length(start, end),
            )
            self.abstract_segments[segment_id] = abstract
            self.segments_by_module_name.setdefault(module_name, []).append(abstract)

        for module in modules:
            aligned_vertices = align_vertices_to_reference(
                reference.vertex,
                module.vertex,
                int(module.direction),
                int(reference.direction),
            )
            for index, spec in enumerate(split_specs):
                start, end = self._segment_coordinates(aligned_vertices, spec)
                segment_id = f"{module_name}:S{index}"
                instance = SegmentInstance(
                    segment_id=segment_id,
                    module_inst=module.name,
                    start=start,
                    end=end,
                    midpoint=segment_midpoint(start, end),
                )
                self.abstract_segments[segment_id].instances[module.name] = instance
                self.instance_lookup[(module.name, segment_id)] = instance

    def split_segments(self, reference_vertices: List[Point]) -> List[EdgeSubdivision]:
        """segment 切割预留接口。

        返回每条母边的比例裁剪规格，所有复用实例使用同一份规格。
        """
        return subdivision_specs(reference_vertices, self.max_segment_length)

    def _segment_coordinates(
        self,
        vertices: List[Point],
        spec: EdgeSubdivision,
    ) -> Tuple[Point, Point]:
        """把母边比例裁剪规格应用到某一实例的对应真实边。"""
        edge_start = vertices[spec.edge_id]
        edge_end = vertices[(spec.edge_id + 1) % len(vertices)]
        return (
            interpolate_edge(edge_start, edge_end, spec.t_start),
            interpolate_edge(edge_start, edge_end, spec.t_end),
        )

    def candidates_for_module(self, module_name: str) -> List[AbstractSegment]:
        """返回某个 module_name 可选的抽象 segment 列表。"""
        return self.segments_by_module_name.get(module_name, [])

    def candidate_segment_ids(self, module_name: str) -> List[str]:
        """返回某个 module_name 可选抽象 segment 的 ID 列表。"""
        return [segment.segment_id for segment in self.candidates_for_module(module_name)]

    def get_instance(self, module_inst: str, segment_id: str) -> SegmentInstance:
        """根据模块实例名和抽象 segment ID 获取真实 segment。"""
        return self.instance_lookup[(module_inst, segment_id)]

    def apply_assignment(
        self,
        group_name: str,
        pins: Iterable[Pin],
        segment_id: str,
        allow_overflow: bool = False,
    ) -> None:
        """把一个同构组实际提交到指定 segment，并写回 Pin 坐标。"""
        pins = list(pins)
        segment = self.abstract_segments[segment_id]
        width = max((pin.width for pin in pins), default=0.0)
        if not allow_overflow and not segment.can_fit(width):
            raise ValueError(
                f"Segment {segment_id} capacity exceeded by group {group_name}: "
                f"used={segment.used_width}, width={width}, capacity={segment.capacity}."
            )
        segment.used_width += width
        if group_name not in segment.assigned_groups:
            segment.assigned_groups.append(group_name)
        for pin in pins:
            instance = self.get_instance(pin.parent_inst, segment_id)
            if pin.full_name not in instance.assigned_pins:
                instance.assigned_pins.append(pin.full_name)
            pin.assigned_segment_coord = instance.midpoint
            pin.segment_endpoints = (instance.start, instance.end)
            pin.x, pin.y = instance.midpoint

    def capacity_violations(self) -> List[dict]:
        """返回所有已提交占用超过容量的抽象 segment。"""
        violations = []
        for segment_id, segment in sorted(self.abstract_segments.items()):
            if segment.used_width > segment.capacity + 1e-9:
                violations.append(
                    {
                        "segment_id": segment_id,
                        "used_width": segment.used_width,
                        "capacity": segment.capacity,
                    }
                )
        return violations

    def snapshot_usage(self) -> SegmentUsage:
        """生成当前全局 segment 使用量快照，供 MCTS 搜索使用。"""
        return SegmentUsage(
            {
                segment_id: segment.used_width
                for segment_id, segment in self.abstract_segments.items()
            }
        )

    def to_output_dict(self) -> Dict[str, dict]:
        """按输出 JSON 需要的结构导出全部 segment 分配结果。"""
        output = {}
        for segment_id, segment in sorted(self.abstract_segments.items()):
            output[segment_id] = {
                "module_name": segment.module_name,
                "index": segment.index,
                "edge_id": segment.edge_id,
                "child_index": segment.child_index,
                "capacity": segment.capacity,
                "used_width": segment.used_width,
                "assigned_groups": segment.assigned_groups,
                "segment_instances": {
                    module_inst: {
                        "start": list(instance.start),
                        "end": list(instance.end),
                        "midpoint": list(instance.midpoint),
                        "assigned_pins": instance.assigned_pins,
                    }
                    for module_inst, instance in sorted(segment.instances.items())
                },
            }
        return output
