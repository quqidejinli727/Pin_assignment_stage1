import json
import math
from pathlib import Path

import assignment_solver as assignment_solver_module
from assignment_solver import AssignmentSolver
from geometry_utils import align_vertices_to_reference
from homology import HomologyManager
from mcts import MCTSNode, MCTSSolver
from scoring import RewardEvaluator, final_net_metrics
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


def test_basic_mcts_dynamic_budget_scales_with_total_search_space(tmp_path):
    """Verify Basic mode scales N_base by the capped total search-space factor."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    groups = homology.unassigned_groups()
    usage = segments.snapshot_usage()

    mcts = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=100,
        search_mode="basic",
        basic_space_scale_divisor=10,
        basic_max_space_factor=10,
        basic_min_simulations=1,
    )
    assert abs(mcts._total_search_space_factor(usage) - 6.4) < 1e-9
    assert mcts._basic_simulation_budget(usage) == 640

    doubled = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=200,
        search_mode="basic",
        basic_space_scale_divisor=10,
        basic_max_space_factor=10,
        basic_min_simulations=1,
    )
    assert doubled._basic_simulation_budget(usage) == 1280

    capped = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=100,
        search_mode="basic",
        basic_space_scale_divisor=1,
        basic_max_space_factor=5,
        basic_min_simulations=1,
    )
    assert capped._total_search_space_factor(usage) == 5
    assert capped._basic_simulation_budget(usage) == 500


def test_mcts_search_space_ignores_already_assigned_groups(tmp_path):
    """Already assigned groups should not multiply estimated search space again."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    groups = homology.unassigned_groups()
    groups[0].assigned = True
    usage = segments.snapshot_usage()

    mcts = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=100,
        search_mode="basic",
        basic_space_scale_divisor=10,
        basic_max_space_factor=10,
        basic_min_simulations=1,
        typical_depth=3,
        space_scale_divisor=10,
        max_space_factor=10,
    )

    assert abs(mcts._total_search_space_factor(usage) - 1.6) < 1e-9
    assert mcts._search_space_factor(usage) == 6.4


def test_basic_mcts_dynamic_budget_has_floor_and_can_be_disabled(tmp_path):
    """Small spaces can shrink below N_base but never below the configured minimum."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    groups = homology.unassigned_groups()
    usage = segments.snapshot_usage()

    floored = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=100,
        search_mode="basic",
        basic_space_scale_divisor=1_000_000,
        basic_min_simulations=256,
    )
    assert abs(floored._total_search_space_factor(usage) - 0.000064) < 1e-12
    assert floored._basic_simulation_budget(usage) == 256

    reduced = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=1000,
        search_mode="basic",
        basic_space_scale_divisor=1_000_000,
        basic_min_simulations=10,
    )
    assert reduced._basic_simulation_budget(usage) == 10

    disabled = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=17,
        search_mode="basic",
        basic_dynamic_simulations=False,
    )
    assert disabled._basic_simulation_budget(usage) == 17


def test_basic_mcts_search_loop_uses_dynamic_budget(tmp_path):
    """Basic search should run exactly the dynamic budget number of simulations."""
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
        simulations=3,
        search_mode="basic",
        basic_space_scale_divisor=10,
        basic_max_space_factor=10,
        basic_min_simulations=1,
    )
    expected = mcts._basic_simulation_budget(segments.snapshot_usage())
    calls = {"simulate": 0}

    def fake_simulate(_node):
        calls["simulate"] += 1
        return 0.0

    mcts._simulate = fake_simulate
    mcts.search()

    assert calls["simulate"] == expected


def test_mcts_search_profile_records_depth_and_branch_counts(tmp_path):
    """Hybrid diagnostics should start from a stable per-tree search profile."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    groups = homology.unassigned_groups()
    usage = segments.snapshot_usage()
    mcts = MCTSSolver(placedb, segments, groups, placedb.nets_list)

    profile = mcts._search_profile(usage)

    assert profile.depth == len(groups)
    assert profile.branch_counts == [4, 4, 4]
    assert profile.average_branching == 4
    assert profile.max_branching == 4
    assert profile.log_total_space == math.log(64)


def test_hybrid_routes_small_tree_to_basic_path(tmp_path):
    """Small Hybrid trees should use the Basic fast path."""
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
        search_mode="hybrid",
        hybrid_basic_depth_limit=10,
        hybrid_basic_log_space_limit=math.log(1_000_000),
    )

    called = {"basic": 0}

    def fake_basic():
        called["basic"] += 1
        return {group.name: "A:S0" for group in groups}

    mcts._search_basic = fake_basic
    assignment = mcts.search()

    assert called["basic"] == 1
    assert set(assignment) == {group.name for group in groups}
    assert mcts.last_search_diagnostics["route"] == "basic"


def test_hybrid_beam_route_respects_budget_caps(tmp_path):
    """Deep/large Hybrid trees should use beam-layered search with budget caps."""
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
        simulations=100,
        search_mode="hybrid",
        hybrid_basic_depth_limit=0,
        hybrid_basic_log_space_limit=0.0,
        hybrid_beam_width=2,
        hybrid_min_layer_simulations=1,
        hybrid_max_layer_simulations=2,
        hybrid_max_tree_simulations=4,
        hybrid_enable_layer_early_stop=False,
        enable_candidate_pruning=False,
    )
    calls = {"simulate": 0}

    def fake_simulate(_node):
        calls["simulate"] += 1
        return float(calls["simulate"])

    mcts._simulate = fake_simulate
    assignment = mcts.search()

    assert calls["simulate"] <= 4
    assert all(layer_budget <= 2 for layer_budget in mcts.last_search_diagnostics["layer_budgets"])
    assert mcts.last_search_diagnostics["route"] in {"beam", "tail"}
    assert set(assignment) == {group.name for group in groups}


def test_candidate_pruning_is_safe_and_disableable(tmp_path):
    """Candidate pruning should keep a non-empty subset and be fully disableable."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    group = homology.pin_groups["A.p"]
    usage = segments.snapshot_usage()

    pruned = MCTSSolver(
        placedb,
        segments,
        [group],
        placedb.nets_list,
        candidate_min_count=2,
        candidate_top_k=1,
        candidate_score_tolerance=0.0,
    )
    pruned_candidates = pruned._candidate_segments(group, usage)

    disabled = MCTSSolver(
        placedb,
        segments,
        [group],
        placedb.nets_list,
        enable_candidate_pruning=False,
    )
    all_candidates = disabled._candidate_segments(group, usage)

    assert pruned_candidates
    assert len(pruned_candidates) <= len(all_candidates)
    assert len(all_candidates) == 4


def test_assignment_greedy_selects_reward_best_feasible_segment(tmp_path):
    """Fallback greedy assignment should prefer reward over remaining capacity."""
    block_path, pingroup_path = write_case(tmp_path)
    solver = AssignmentSolver(str(block_path), str(pingroup_path), simulations=1)
    group = solver.homology.pin_groups["A.p"]
    candidates = solver.segment_manager.candidates_for_module(group.module_name)[:2]
    preferred = candidates[1]
    candidates[0].used_width = 0.0
    preferred.used_width = preferred.capacity - 2.0

    class FakeEvaluator:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, temporary_locations):
            return (
                100.0
                if any(point == preferred.instances[pin.parent_inst].midpoint for pin, point in [
                    (group.pins[0], temporary_locations[group.pins[0].full_name])
                ])
                else 0.0
            )

        def close(self):
            pass

    original_evaluator = assignment_solver_module.RewardEvaluator
    assignment_solver_module.RewardEvaluator = FakeEvaluator
    try:
        chosen = solver._best_greedy_segment_by_reward(group, candidates)
    finally:
        assignment_solver_module.RewardEvaluator = original_evaluator

    assert chosen is preferred


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

    scaled_evaluator = RewardEvaluator(
        placedb.nets_list,
        placedb,
        wirelength_weight=1.0,
        feedthrough_weight=0.0,
        enable_feedthrough=False,
        reward_scale=10.0,
    )
    assert scaled_evaluator.evaluate(temporary_locations) == 20.0


class FakeFeedthroughContext:
    """Small test double that records feedthrough calls without starting ftpred."""

    def __init__(self):
        self.calls = []
        self.closed = False

    def run_one_net_at_locations(self, net, locations):
        self.calls.append((net.net_id, tuple(sorted(locations))))
        return 0.0

    def close(self):
        self.closed = True


def test_reward_evaluator_uses_shared_feedthrough_context(tmp_path):
    """Feedthrough references and candidates must call the injected context."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    context = FakeFeedthroughContext()

    evaluator = RewardEvaluator(
        placedb.nets_list,
        placedb,
        feedthrough_weight=1.0,
        enable_feedthrough=True,
        feedthrough_context=context,
    )
    reference_call_count = len(context.calls)
    evaluator.evaluate({"TOP.U_A0.p": (35.0, 5.0)})

    assert reference_call_count == len(placedb.nets_list)
    assert len(context.calls) == reference_call_count + len(placedb.nets_list)
    assert not context.closed


def test_mcts_solvers_share_one_feedthrough_context(tmp_path):
    """Multiple local MCTS solvers can reuse one feedthrough context."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    groups = homology.unassigned_groups()
    context = FakeFeedthroughContext()

    for seed in [1, 2]:
        mcts = MCTSSolver(
            placedb,
            segments,
            groups,
            placedb.nets_list,
            simulations=1,
            random_seed=seed,
            search_mode="basic",
            feedthrough_weight=1.0,
            enable_feedthrough=True,
            feedthrough_context=context,
        )
        mcts.search()

    assert len(context.calls) > 0
    assert not context.closed


def test_assignment_solver_creates_one_feedthrough_context(tmp_path):
    """A full solve should create one shared FT context and close it explicitly."""
    block_path, pingroup_path = write_case(tmp_path)
    created_contexts = []

    class FakeContext(FakeFeedthroughContext):
        def __init__(self, *args, **kwargs):
            super().__init__()
            created_contexts.append(self)

    original_context = assignment_solver_module.FeedthroughContext
    assignment_solver_module.FeedthroughContext = FakeContext
    try:
        solver = AssignmentSolver(
            str(block_path),
            str(pingroup_path),
            simulations=2,
            mcts_search_mode="basic",
            feedthrough_weight=1.0,
            feedthrough_source_dir=tmp_path,
            enable_feedthrough=True,
        )
        result = solver.solve()
        solver.close_feedthrough_context()
    finally:
        assignment_solver_module.FeedthroughContext = original_context

    assert result["summary"]["assigned_pin_count"] == result["summary"]["pin_count"]
    assert len(created_contexts) == 1
    assert created_contexts[0].closed


def test_assignment_solver_skips_context_when_feedthrough_reward_is_disabled(tmp_path):
    """No shared FT context should be opened when feedthrough reward weight is zero."""
    block_path, pingroup_path = write_case(tmp_path)

    class FailingContext:
        def __init__(self, *args, **kwargs):
            raise AssertionError("feedthrough context should not be created")

    original_context = assignment_solver_module.FeedthroughContext
    assignment_solver_module.FeedthroughContext = FailingContext
    try:
        solver = AssignmentSolver(
            str(block_path),
            str(pingroup_path),
            simulations=2,
            feedthrough_weight=0.0,
            feedthrough_source_dir=tmp_path,
            enable_feedthrough=True,
        )
        result = solver.solve()
    finally:
        assignment_solver_module.FeedthroughContext = original_context

    assert result["summary"]["assigned_pin_count"] == result["summary"]["pin_count"]


def test_assignment_solver_closes_context_on_mcts_failure(tmp_path):
    """The shared FT context must be closed when local MCTS search raises."""
    block_path, pingroup_path = write_case(tmp_path)
    created_contexts = []

    class FakeContext(FakeFeedthroughContext):
        def __init__(self, *args, **kwargs):
            super().__init__()
            created_contexts.append(self)

    class FailingMCTS:
        def __init__(self, *args, **kwargs):
            pass

        def search(self):
            raise RuntimeError("forced mcts failure")

    original_context = assignment_solver_module.FeedthroughContext
    original_mcts = assignment_solver_module.MCTSSolver
    assignment_solver_module.FeedthroughContext = FakeContext
    assignment_solver_module.MCTSSolver = FailingMCTS
    try:
        solver = AssignmentSolver(
            str(block_path),
            str(pingroup_path),
            simulations=2,
            feedthrough_weight=1.0,
            feedthrough_source_dir=tmp_path,
            enable_feedthrough=True,
        )
        try:
            solver.solve()
        except RuntimeError as exc:
            assert str(exc) == "forced mcts failure"
        else:
            raise AssertionError("solver should have raised")
    finally:
        assignment_solver_module.FeedthroughContext = original_context
        assignment_solver_module.MCTSSolver = original_mcts

    assert len(created_contexts) == 1
    assert created_contexts[0].closed


def test_final_net_metrics_can_reuse_feedthrough_context(tmp_path):
    """Final metrics should not require a new predictor when context is provided."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    context = FakeFeedthroughContext()

    metrics = final_net_metrics(
        placedb,
        tmp_path,
        enable_feedthrough=True,
        auto_build_feedthrough=False,
        feedthrough_context=context,
    )

    assert len(metrics) == len(placedb.nets_list)
    assert len(context.calls) == len(placedb.nets_list)
    assert not context.closed


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
