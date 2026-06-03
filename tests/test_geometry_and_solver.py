import json
from pathlib import Path

from assignment_solver import AssignmentSolver
from geometry_utils import align_vertices_to_reference
from homology import HomologyManager
from mcts import MCTSNode, MCTSSolver
from scoring import RewardEvaluator
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


def test_solver_supports_basic_mcts_search_mode(tmp_path):
    """验证 config 可切换到基础版 MCTS 搜索流程。"""
    block_path, pingroup_path = write_case(tmp_path)
    solver = AssignmentSolver(
        str(block_path),
        str(pingroup_path),
        simulations=32,
        mcts_search_mode="basic",
    )
    result = solver.solve()
    assert result["summary"]["assigned_pin_count"] == result["summary"]["pin_count"]
    assert result["summary"]["unassigned_group_count"] == 0


def test_reward_evaluator_normalizes_against_centroid_reference(tmp_path):
    """Verify per-net HPWL reward is normalized against cached centroid references."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    evaluator = RewardEvaluator(
        placedb.nets_list,
        placedb,
        wirelength_weight=1.0,
        feedthrough_weight=0.0,
        enable_feedthrough=False,
    )

    assert evaluator.evaluate({}) == 0.0

    temporary_locations = {
        "TOP.U_A0.p": (35.0, 5.0),
        "TOP.U_A1.p": (35.0, 5.0),
    }
    assert evaluator.evaluate(temporary_locations) == 2.0


def test_mcts_scaled_budget_and_tail_decay(tmp_path):
    """验证新 MCTS 搜索预算公式和长尾衰减。"""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    groups = homology.unassigned_groups()
    mcts = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=1000,
        budget_decay=0.5,
        tail_decay=0.8,
        typical_depth=3,
        space_scale_divisor=1000,
        max_space_factor=10,
        min_layer_simulations=1,
        tail_depth=3,
    )

    usage = segments.snapshot_usage()
    assert mcts._search_space_factor(usage) == 0.064
    assert mcts._total_simulation_budget(usage) == 64
    assert mcts._layer_budget(1000, 1) == 1000
    assert mcts._layer_budget(1000, 2) == 500
    assert mcts._layer_budget(1000, 3) == 250
    assert mcts._layer_budget(1000, 4) == 200


def test_mcts_layer_search_descends_one_child_per_level(tmp_path):
    """验证逐层搜索每层只承诺一个子节点，剩余分配由贪心补全。"""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    groups = homology.unassigned_groups()
    mcts = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=16,
        min_layer_simulations=1,
        typical_depth=1,
        space_scale_divisor=1,
        tail_depth=1,
        enable_tail_early_stop=True,
    )

    calls = {"early_stop": 0}

    def always_stop(_children):
        calls["early_stop"] += 1
        return True

    mcts._has_decisive_ucb_lead = always_stop
    assignment = mcts.search()

    assert calls["early_stop"] == 1
    assert set(assignment) == {group.name for group in groups}


def test_mcts_tail_early_stop_can_be_disabled(tmp_path):
    """验证禁用早停后，即使进入长尾也不会调用领先判断。"""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    groups = homology.unassigned_groups()
    mcts = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=16,
        min_layer_simulations=1,
        typical_depth=1,
        space_scale_divisor=1,
        tail_depth=1,
        enable_tail_early_stop=False,
    )

    def fail_if_called(_children):
        raise AssertionError("early stop should be disabled")

    mcts._has_decisive_ucb_lead = fail_if_called
    assignment = mcts.search()

    assert set(assignment) == {group.name for group in groups}


def test_mcts_ucb_lead_uses_same_layer_std(tmp_path):
    """验证长尾早停口径使用同层子节点 UCB 分数标准差。"""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    groups = homology.unassigned_groups()
    mcts = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        early_stop_std_multiplier=2.0,
    )
    parent = MCTSNode(
        group_index=0,
        usage=segments.snapshot_usage(),
        assignments={},
        visits=100,
    )
    children = []
    for total_reward in [10000.0, 0.0, 0.0]:
        child = MCTSNode(
            group_index=1,
            usage=segments.snapshot_usage(),
            assignments={},
            parent=parent,
            visits=10,
            total_reward=total_reward,
        )
        children.append(child)

    assert mcts._has_decisive_ucb_lead(children)


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
