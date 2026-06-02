"""验证复用 block 在旋转/镜像后得到的 segment 映射是否正确。

该脚本不依赖 pytest，可直接运行：
    python tests/validate_segment_mapping.py

验证内容：
1. reference 实例允许为任意 direction，而不是固定 direction=0。
2. 实例顶点允许从任意顶点开始记录。
3. 对正方形、矩形、L 形和不规则凸多边形，逐段检查实际映射出的
   segment 起点和终点是否等于基础形状对应边变换后的预期坐标。
4. 覆盖输入定义中的 8 种旋转/镜像方向。
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PlaceDB import PlaceDB  # noqa: E402
from geometry_utils import signed_area  # noqa: E402
from segment import SegmentManager  # noqa: E402
from segment_subdivision import interpolate_edge, subdivision_specs  # noqa: E402

Point = Tuple[float, float]


SHAPES: Dict[str, List[Point]] = {
    "square": [(0, 0), (10, 0), (10, 10), (0, 10)],
    "rectangle": [(0, 0), (18, 0), (18, 7), (0, 7)],
    "l_shape": [(0, 0), (16, 0), (16, 5), (6, 5), (6, 14), (0, 14)],
    "irregular_pentagon": [(0, 0), (13, 0), (17, 6), (8, 15), (1, 9)],
}


def forward_direction_point(point: Point, direction: int) -> Point:
    """按输入 direction 定义，把基础形状点变换到实例方向。"""
    x, y = point
    if direction == 0:  # R0
        return (x, y)
    if direction == 1:  # R180
        return (-x, -y)
    if direction == 2:  # R90
        return (-y, x)
    if direction == 3:  # R270
        return (y, -x)
    if direction == 4:  # MY
        return (-x, y)
    if direction == 5:  # MX
        return (x, -y)
    if direction == 6:  # MX then R90
        return (y, x)
    if direction == 7:  # MY then R90
        return (-y, -x)
    raise ValueError(f"Unknown direction: {direction}")


def translated_instance(
    base_vertices: Sequence[Point],
    direction: int,
    offset: Point,
    cyclic_start: int,
) -> Tuple[List[List[float]], List[Point]]:
    """生成 JSON 实例顶点及按基础 segment 顺序排列的预期坐标。"""
    expected = []
    for point in base_vertices:
        transformed = forward_direction_point(point, direction)
        expected.append((transformed[0] + offset[0], transformed[1] + offset[1]))

    json_order = list(expected)
    if signed_area(json_order) < 0:
        json_order.reverse()
    start = cyclic_start % len(json_order)
    json_order = json_order[start:] + json_order[:start]
    return [[x, y] for x, y in json_order], expected


def build_case(
    shape_name: str,
    base_vertices: Sequence[Point],
    reference_direction: int,
) -> Tuple[dict, dict]:
    """创建一个复用组，reference 和各方向实例均带一个 Pin。"""
    directions = [reference_direction] + [
        direction for direction in range(8) if direction != reference_direction
    ]
    children = []
    pin_net = []
    expected_by_module = {}
    for position, direction in enumerate(directions):
        module_inst = f"TOP.U_{shape_name}_{direction}"
        vertices, expected = translated_instance(
            base_vertices,
            direction,
            offset=(100.0 * position + 23.0, 70.0 * position + 19.0),
            cyclic_start=position + 1,
        )
        children.append(
            {
                "name": module_inst,
                "module_name": shape_name,
                "direction": direction,
                "color": "#000000",
                "vertex": vertices,
                "children": [],
            }
        )
        pin_net.append(
            {
                "parent_inst": module_inst,
                "parent_module": shape_name,
                "pingroup_name": "mapping_probe",
                "scope": [],
                "successors": [],
                "width": 1.0,
            }
        )
        expected_by_module[module_inst] = expected

    block = {
        "name": "TOP",
        "module_name": "TOP",
        "direction": 0,
        "color": "#000000",
        "vertex": [[0, 0], [2000, 0], [2000, 2000], [0, 2000]],
        "children": children,
    }
    return block, [pin_net], expected_by_module


def assert_point_equal(actual: Point, expected: Point, context: str) -> None:
    """在误差范围内比较坐标点，失败时给出具体 segment 上下文。"""
    if not (math.isclose(actual[0], expected[0], abs_tol=1e-6) and
            math.isclose(actual[1], expected[1], abs_tol=1e-6)):
        raise AssertionError(f"{context}: actual={actual}, expected={expected}")


def validate_one_shape(
    shape_name: str,
    base_vertices: Sequence[Point],
    reference_direction: int,
    max_segment_length: float | None = None,
) -> int:
    """构建一组复用 block，逐一验证每个实例所有 segment 端点。"""
    block, pingroup, expected_by_module = build_case(
        shape_name,
        base_vertices,
        reference_direction,
    )
    with tempfile.TemporaryDirectory() as temporary_dir:
        directory = Path(temporary_dir)
        block_path = directory / "block.json"
        pingroup_path = directory / "pingroup.json"
        block_path.write_text(json.dumps(block), encoding="utf-8")
        pingroup_path.write_text(json.dumps(pingroup), encoding="utf-8")

        placedb = PlaceDB(str(block_path), str(pingroup_path))
        segments = SegmentManager(placedb, max_segment_length=max_segment_length)
        specs = subdivision_specs(base_vertices, max_segment_length)

        checked = 0
        for module_inst, expected_vertices in expected_by_module.items():
            for index, spec in enumerate(specs):
                segment = segments.get_instance(module_inst, f"{shape_name}:S{index}")
                edge_start = expected_vertices[spec.edge_id]
                edge_end = expected_vertices[(spec.edge_id + 1) % len(expected_vertices)]
                expected_start = interpolate_edge(edge_start, edge_end, spec.t_start)
                expected_end = interpolate_edge(edge_start, edge_end, spec.t_end)
                context = f"{shape_name}, ref_dir={reference_direction}, {module_inst}, S{index}"
                assert_point_equal(segment.start, expected_start, f"{context} start")
                assert_point_equal(segment.end, expected_end, f"{context} end")
                checked += 1
    return checked


def main() -> None:
    """运行全部图形和 reference 方向组合的 segment 端点验证。"""
    checked = 0
    for shape_name, vertices in SHAPES.items():
        for reference_direction in range(8):
            checked += validate_one_shape(shape_name, vertices, reference_direction)
            checked += validate_one_shape(
                shape_name,
                vertices,
                reference_direction,
                max_segment_length=6.0,
            )
    print(
        "Segment mapping validation passed: "
        f"{len(SHAPES)} shapes, 8 reference directions, original and subdivided edges, "
        f"{checked} segment instances checked."
    )


if __name__ == "__main__":
    main()
