import json
from pathlib import Path

from assignment_solver import AssignmentSolver
from geometry_utils import align_vertices_to_reference
from segment import SegmentManager
from PlaceDB import PlaceDB


def write_case(tmp_path: Path) -> tuple[Path, Path]:
    """生成一个最小可运行测试用例，覆盖复用 block 和同构 Pin。"""
    block = {
        "name": "TOP",
        "module_name": "TOP",
        "direction": 0,
        "color": "#000000",
        "vertex": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "children": [
            {
                "name": "TOP.U_A0",
                "module_name": "A",
                "direction": 0,
                "color": "#aaaaaa",
                "vertex": [[0, 0], [20, 0], [20, 10], [0, 10]],
                "children": [],
            },
            {
                "name": "TOP.U_A1",
                "module_name": "A",
                "direction": 0,
                "color": "#aaaaaa",
                "vertex": [[50, 50], [70, 50], [70, 60], [50, 60]],
                "children": [],
            },
            {
                "name": "TOP.U_B0",
                "module_name": "B",
                "direction": 0,
                "color": "#bbbbbb",
                "vertex": [[30, 0], [40, 0], [40, 10], [30, 10]],
                "children": [],
            },
        ],
    }
    pingroup = [
        [
            {
                "parent_inst": "TOP.U_A0",
                "parent_module": "A",
                "pingroup_name": "p",
                "scope": [],
                "successors": [],
                "width": 1.0,
            },
            {
                "parent_inst": "TOP.U_B0",
                "parent_module": "B",
                "pingroup_name": "q",
                "scope": [],
                "successors": [],
                "width": 1.0,
            },
        ],
        [
            {
                "parent_inst": "TOP.U_A1",
                "parent_module": "A",
                "pingroup_name": "p",
                "scope": [],
                "successors": [],
                "width": 1.0,
            },
            {
                "parent_inst": "TOP.U_B0",
                "parent_module": "B",
                "pingroup_name": "r",
                "scope": [],
                "successors": [],
                "width": 1.0,
            },
        ],
    ]
    block_path = tmp_path / "block.json"
    pingroup_path = tmp_path / "pingroup.json"
    block_path.write_text(json.dumps(block), encoding="utf-8")
    pingroup_path.write_text(json.dumps(pingroup), encoding="utf-8")
    return block_path, pingroup_path


def test_vertex_alignment_handles_cyclic_start():
    """验证顶点起点变化时仍能对齐到 reference 顶点顺序。"""
    reference = [[0, 0], [20, 0], [20, 10], [0, 10]]
    shifted = [[20, 10], [0, 10], [0, 0], [20, 0]]
    aligned = align_vertices_to_reference(reference, shifted, 0)
    assert aligned[0] == (0.0, 0.0)
    assert aligned[1] == (20.0, 0.0)


def test_square_alignment_uses_direction_when_reference_is_rotated():
    """验证正方形 reference 旋转时仍按基础坐标确定对应边。"""
    reference_r180 = [[20, 20], [10, 20], [10, 10], [20, 10]]
    base_instance = [[0, 0], [10, 0], [10, 10], [0, 10]]
    aligned = align_vertices_to_reference(
        reference_r180,
        base_instance,
        module_direction=0,
        reference_direction=1,
    )
    assert aligned[0] == (0.0, 0.0)
    assert aligned[1] == (10.0, 0.0)


def test_placedb_indexes_and_segment_mapping(tmp_path):
    """验证 PlaceDB 索引和 segment 实例映射能正确建立。"""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    assert len(placedb.pins_by_homology["A.p"]) == 2
    assert "A:S0" in segments.abstract_segments
    assert ("TOP.U_A0", "A:S0") in segments.instance_lookup
    assert ("TOP.U_A1", "A:S0") in segments.instance_lookup


def test_solver_assigns_all_pins_and_respects_capacity(tmp_path):
    """验证正常场景下所有 Pin 都完成分配且不超容量。"""
    block_path, pingroup_path = write_case(tmp_path)
    solver = AssignmentSolver(str(block_path), str(pingroup_path), simulations=32)
    result = solver.solve()
    assert result["summary"]["assigned_pin_count"] == result["summary"]["pin_count"]
    assert result["summary"]["unassigned_group_count"] == 0
    assert result["summary"]["capacity_violation_count"] == 0
    for segment in result["segments"].values():
        assert segment["used_width"] <= segment["capacity"] + 1e-9

    for pin in solver.placedb.pin_dict.values():
        assert pin.assigned_segment_id is not None
        assert pin.assigned_segment_coord is not None
        assert pin.segment_endpoints is not None

    assigned_pin_names = []
    for segment in result["segments"].values():
        for instance in segment["segment_instances"].values():
            assigned_pin_names.extend(instance["assigned_pins"])
    assert sorted(assigned_pin_names) == sorted(solver.placedb.pin_dict)


def test_overflow_fallback_completes_assignment_when_pin_is_wider_than_edge(tmp_path):
    """验证 Pin 宽度超过所有边时，overflow fallback 仍能完成分配并报告。"""
    block_path, pingroup_path = write_case(tmp_path)
    pingroup = json.loads(pingroup_path.read_text(encoding="utf-8"))
    pingroup[0][0]["width"] = 50.0
    pingroup[1][0]["width"] = 50.0
    pingroup_path.write_text(json.dumps(pingroup), encoding="utf-8")

    solver = AssignmentSolver(str(block_path), str(pingroup_path), simulations=8)
    result = solver.solve()

    assert result["summary"]["assigned_pin_count"] == result["summary"]["pin_count"]
    assert result["summary"]["unassigned_group_count"] == 0
    assert result["summary"]["capacity_violation_count"] > 0
    assert any(issue["reason"] == "forced_overflow_assignment" for issue in result["assignment_issues"])
