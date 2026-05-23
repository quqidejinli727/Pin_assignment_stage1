"""Geometry helpers for homologous block and segment matching.

The input polygons are already placed in the global layout.  Blocks that share
the same ``module_name`` may be rotated, mirrored, and may start their vertex
list from different corners.  This module normalizes vertices only for matching;
all final segment coordinates are always taken from the original placed shape.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from shapely.geometry import Polygon

Point = Tuple[float, float]

COORD_TOLERANCE = 1e-6


def to_point_list(vertices: Sequence[Sequence[float]]) -> List[Point]:
    """把 JSON 顶点数组转换为统一的浮点坐标元组。"""
    return [(float(x), float(y)) for x, y in vertices]


def polygon_from_vertices(vertices: Sequence[Sequence[float]]) -> Polygon:
    """用顶点创建 shapely 多边形，并尝试修复无效环。"""
    polygon = Polygon(to_point_list(vertices))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def signed_area(vertices: Sequence[Point]) -> float:
    """计算有符号面积，正值表示顶点为逆时针顺序。"""
    area = 0.0
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % len(vertices)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def ensure_counterclockwise(vertices: Sequence[Point]) -> List[Point]:
    """保证顶点按逆时针排列，同时尽量保留原起点。"""
    points = list(vertices)
    if len(points) >= 3 and signed_area(points) < 0:
        first = points[0]
        points = [first] + list(reversed(points[1:]))
    return points


def centroid(vertices: Sequence[Sequence[float]]) -> Point:
    """返回多边形的 shapely 质心坐标。"""
    point = polygon_from_vertices(vertices).centroid
    return (float(point.x), float(point.y))


def segment_midpoint(start: Point, end: Point) -> Point:
    """计算线段中点。"""
    return ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)


def segment_length(start: Point, end: Point) -> float:
    """计算线段欧氏长度。"""
    return math.hypot(end[0] - start[0], end[1] - start[1])


def inverse_direction_point(point: Point, direction: int) -> Point:
    """按 direction 对点做逆旋转/逆镜像变换。

    该变换只用于生成稳定的局部形状签名；后续会平移到局部原点，
    因此全局锚点不会影响 segment 对应关系。
    """
    x, y = point
    if direction == 0:
        return (x, y)
    if direction == 1:
        return (-x, -y)
    if direction == 2:
        return (y, -x)
    if direction == 3:
        return (-y, x)
    if direction == 4:
        return (-x, y)
    if direction == 5:
        return (x, -y)
    if direction == 6:
        # 先沿 X 轴镜像再逆时针旋转 90 度，该组合变换的逆仍为自身。
        return (y, x)
    if direction == 7:
        # 先沿 Y 轴镜像再逆时针旋转 90 度，该组合变换的逆仍为自身。
        return (-y, -x)
    return (x, y)


def normalize_vertices(
    vertices: Sequence[Sequence[float]],
    direction: int = 0,
    precision: int = 6,
) -> List[Point]:
    """归一化已放置顶点，用于匹配复用 block 的形状。

    处理步骤：逆变换、平移到局部原点、恢复逆时针顺序并按容差取整。
    """
    normalized, _ = canonical_vertex_mapping(vertices, direction, precision)
    return normalized


def canonical_vertex_mapping(
    vertices: Sequence[Sequence[float]],
    direction: int = 0,
    precision: int = 6,
) -> Tuple[List[Point], List[Point]]:
    """生成基础坐标顶点及其对应的原始放置坐标。

    reference block 也必须按自身 direction 处理，不能假定第一个实例方向为
    0。反变换后使用基础坐标中字典序最小的顶点作为固定起点，因此即使正方形
    的边签名完全相同，也能稳定确定相同的 segment 编号。
    """
    original = to_point_list(vertices)
    paired = [
        [inverse_direction_point(point, direction), point]
        for point in original
    ]
    transformed = [item[0] for item in paired]
    if len(transformed) >= 3 and signed_area(transformed) < 0:
        paired = list(reversed(paired))
        transformed = [item[0] for item in paired]

    min_x = min(x for x, _ in transformed)
    min_y = min(y for _, y in transformed)
    normalized = [
        (round(point[0] - min_x, precision), round(point[1] - min_y, precision))
        for point in transformed
    ]
    start = min(range(len(normalized)), key=lambda index: normalized[index])
    ordered_normalized = rotate_list(normalized, start)
    ordered_original = rotate_list([item[1] for item in paired], start)
    return ordered_normalized, ordered_original


def rotate_list(values: Sequence[Point], start: int) -> List[Point]:
    """旋转循环列表，使指定下标成为第一个元素。"""
    return list(values[start:]) + list(values[:start])


def align_vertices_to_reference(
    reference_vertices: Sequence[Sequence[float]],
    module_vertices: Sequence[Sequence[float]],
    module_direction: int,
    reference_direction: int = 0,
    tolerance: float = COORD_TOLERANCE,
) -> List[Point]:
    """把模块原始顶点顺序对齐到 reference block 的顶点顺序。

    返回值仍使用原始放置坐标，只调整起点，使同编号边互相对应。
    """
    reference, _ = canonical_vertex_mapping(reference_vertices, reference_direction)
    normalized, original = canonical_vertex_mapping(module_vertices, module_direction)

    if len(reference) != len(normalized):
        raise ValueError("Cannot align polygons with different vertex counts.")

    if not _point_sequences_match(reference, normalized, tolerance):
        raise ValueError("Reused modules do not share the same normalized base shape.")
    return original


def _point_sequences_match(
    reference_vertices: Iterable[Point],
    candidate_vertices: Iterable[Point],
    tolerance: float,
) -> bool:
    """判断基础坐标系下的顶点序列是否逐点一致。"""
    for reference, candidate in zip(reference_vertices, candidate_vertices):
        if abs(reference[0] - candidate[0]) > tolerance:
            return False
        if abs(reference[1] - candidate[1]) > tolerance:
            return False
    return True


def hpwl(points: Sequence[Point]) -> float:
    """根据一组 Pin 坐标计算 HPWL 半周长线长。"""
    if not points:
        return 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))
