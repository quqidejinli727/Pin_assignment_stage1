"""Segment 边裁剪工具。

裁剪规则来自用户提供的 ``segment(1).py``：统计全部 block 边长取得
百分位阈值，并把长于阈值的边均匀切分为更短的子 segment。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from geometry_utils import Point, segment_length


@dataclass(frozen=True)
class EdgeSubdivision:
    """一段由原始边按比例切出的子 segment 规格。"""

    edge_id: int
    child_index: int
    t_start: float
    t_end: float


def percentile_edge_length(block_json_path: str | Path, percentile: int) -> float:
    """统计全部 block 边长并返回指定百分位对应长度。"""
    if isinstance(percentile, bool) or not isinstance(percentile, int):
        raise TypeError("segment_length_percentile 必须是 0 到 100 的 int。")
    if not 0 <= percentile <= 100:
        raise ValueError("segment_length_percentile 必须在 0 到 100 之间。")

    root = json.loads(Path(block_json_path).read_text(encoding="utf-8"))
    lengths: List[float] = []

    def visit(node: dict) -> None:
        vertices = node.get("vertex") or []
        if len(vertices) >= 2:
            for index in range(len(vertices)):
                length = segment_length(
                    tuple(vertices[index]),
                    tuple(vertices[(index + 1) % len(vertices)]),
                )
                if length > 0:
                    lengths.append(length)
        for child in node.get("children", []) or []:
            visit(child)

    visit(root)
    if not lengths:
        raise ValueError("block JSON 中没有可用于 segment 裁剪的有效边。")

    lengths.sort()
    position = (percentile / 100.0) * (len(lengths) - 1)
    index = math.floor(position + 0.5)
    return lengths[index]


def subdivision_specs(
    vertices: Sequence[Point],
    max_length: float | None,
) -> List[EdgeSubdivision]:
    """按最大长度为多边形各边生成一致的切分比例规格。"""
    if max_length is not None and max_length <= 0:
        raise ValueError("segment 最大长度阈值必须大于 0。")
    specs = []
    for edge_id in range(len(vertices)):
        start = vertices[edge_id]
        end = vertices[(edge_id + 1) % len(vertices)]
        length = segment_length(start, end)
        if length == 0:
            continue
        pieces = (
            math.floor(length / max_length) + 1
            if max_length is not None and length > max_length
            else 1
        )
        for child_index in range(pieces):
            specs.append(
                EdgeSubdivision(
                    edge_id=edge_id,
                    child_index=child_index,
                    t_start=child_index / pieces,
                    t_end=(child_index + 1) / pieces,
                )
            )
    return specs


def interpolate_edge(start: Point, end: Point, ratio: float) -> Point:
    """根据比例在一条原始边上取得裁剪后的端点。"""
    return (
        start[0] + (end[0] - start[0]) * ratio,
        start[1] + (end[1] - start[1]) * ratio,
    )
