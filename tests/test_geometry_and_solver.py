import json
import logging
import math
from io import StringIO
from pathlib import Path

import assignment_solver as assignment_solver_module
from analyze_mcts_tree_batches import analyze_batches
from assignment_solver import AssignmentSolver
from geometry_utils import align_vertices_to_reference
from homology import HomologyManager
from mcts import MCTSNode, MCTSSolver, SKIP_SEGMENT_ID
from scoring import RewardEvaluator, final_net_metrics, net_hpwl, summarize_metrics
from segment import SegmentManager
from PlaceDB import PlaceDB


def write_case(tmp_path: Path) -> tuple[Path, Path]:
    """Test helper."""
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
    """Test helper."""
    reference = [[0, 0], [20, 0], [20, 10], [0, 10]]
    shifted = [[20, 10], [0, 10], [0, 0], [20, 0]]
    aligned = align_vertices_to_reference(reference, shifted, 0)
    assert aligned[0] == (0.0, 0.0)
    assert aligned[1] == (20.0, 0.0)


def test_square_alignment_uses_direction_when_reference_is_rotated():
    """Test helper."""
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
    """Test helper."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    assert len(placedb.pins_by_homology["A.p"]) == 2
    assert "A:S0" in segments.abstract_segments
    assert ("TOP.U_A0", "A:S0") in segments.instance_lookup
    assert ("TOP.U_A1", "A:S0") in segments.instance_lookup


def write_fanout_sort_case(tmp_path: Path) -> tuple[Path, Path]:
    """Create repeated pin records where successors reveal multi-fanout reuse."""
    block = {
        "name": "TOP",
        "module_name": "TOP",
        "direction": 0,
        "color": "#000000",
        "vertex": [[0, 0], [140, 0], [140, 80], [0, 80]],
        "children": [
            {
                "name": "TOP.U_A0",
                "module_name": "A",
                "direction": 0,
                "color": "#aaaaaa",
                "vertex": [[0, 0], [10, 0], [10, 10], [0, 10]],
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
            {
                "name": "TOP.U_B1",
                "module_name": "B",
                "direction": 0,
                "color": "#bbbbbb",
                "vertex": [[50, 0], [60, 0], [60, 10], [50, 10]],
                "children": [],
            },
            {
                "name": "TOP.U_C0",
                "module_name": "C",
                "direction": 0,
                "color": "#cccccc",
                "vertex": [[80, 0], [90, 0], [90, 10], [80, 10]],
                "children": [],
            },
            {
                "name": "TOP.U_C1",
                "module_name": "C",
                "direction": 0,
                "color": "#cccccc",
                "vertex": [[100, 0], [110, 0], [110, 10], [100, 10]],
                "children": [],
            },
        ],
    }
    pingroup = [
        [
            {
                "parent_inst": "TOP.U_A0",
                "parent_module": "A",
                "pingroup_name": "fan",
                "scope": [],
                "successors": ["TOP.U_B0.p"],
                "width": 1.0,
            },
            {
                "parent_inst": "TOP.U_B0",
                "parent_module": "B",
                "pingroup_name": "p",
                "scope": [],
                "successors": [],
                "width": 1.0,
            },
        ],
        [
            {
                "parent_inst": "TOP.U_A0",
                "parent_module": "A",
                "pingroup_name": "fan",
                "scope": [],
                "successors": ["TOP.U_B1.p"],
                "width": 1.0,
            },
            {
                "parent_inst": "TOP.U_B1",
                "parent_module": "B",
                "pingroup_name": "p",
                "scope": [],
                "successors": [],
                "width": 1.0,
            },
        ],
        [
            {
                "parent_inst": "TOP.U_C0",
                "parent_module": "C",
                "pingroup_name": "p",
                "scope": [],
                "successors": [],
                "width": 1.0,
            },
            {
                "parent_inst": "TOP.U_C1",
                "parent_module": "C",
                "pingroup_name": "p",
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


def test_fanout_reuse_sorting_can_be_disabled(tmp_path):
    """Multi-fanout pins can either boost reuse sorting or be treated as ordinary pins."""
    block_path, pingroup_path = write_fanout_sort_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))

    fanout_homology = HomologyManager(placedb, use_fanout_reuse_for_sorting=True)
    ordinary_homology = HomologyManager(placedb, use_fanout_reuse_for_sorting=False)

    assert fanout_homology.pin_groups["A.fan"].sort_reuse_count == 2
    assert ordinary_homology.pin_groups["A.fan"].sort_reuse_count == 1
    assert fanout_homology.unassigned_groups()[0].name == "A.fan"
    assert ordinary_homology.unassigned_groups()[0].name == "B.p"


def test_reward_true_skip_removes_pin_from_hpwl(tmp_path):
    """Skipped pins are removed from reward metrics instead of using virtual locations."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    net = placedb.nets_list[0]
    skipped = {"TOP.U_B0.q"}

    assigned_b_location = {"TOP.U_B0.q": (1000.0, 1000.0)}
    assert net_hpwl(net, placedb, assigned_b_location) > 0.0
    assert net_hpwl(net, placedb, assigned_b_location, skipped) == 0.0

    evaluator = RewardEvaluator(
        [net],
        placedb,
        wirelength_weight=1.0,
        feedthrough_weight=0.0,
        skipped_pin_names=skipped,
    )
    try:
        assert evaluator.metric_nets == []
        assert evaluator.evaluate(assigned_b_location) == 0.0
    finally:
        evaluator.close()


def test_mcts_skip_groups_do_not_inflate_search_profile(tmp_path):
    """Skip-only groups should not count as effective MCTS depth or emit locations."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    groups = homology.unassigned_groups()
    skipped_names = {groups[0].name}

    mcts = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=4,
        search_mode="basic",
        feedthrough_weight=0.0,
        skipped_group_names=skipped_names,
    )
    profile = mcts._search_profile(segments.snapshot_usage())
    assert profile.depth == len(groups) - 1

    assignments = {groups[0].name: "__SKIP_UNCOVERED_GROUP__"}
    locations = mcts._temporary_locations(assignments)
    assert all(pin.full_name not in locations for pin in groups[0].pins)


def test_solver_true_skip_excludes_groups_from_mcts_tree(tmp_path):
    """True-skip groups should not be passed into the current MCTS tree."""
    block = {
        "name": "TOP",
        "module_name": "TOP",
        "direction": 0,
        "color": "#000000",
        "vertex": [[0, 0], [200, 0], [200, 80], [0, 80]],
        "children": [
            {
                "name": f"TOP.U_S{index}",
                "module_name": "S",
                "direction": 0,
                "color": "#aaaaaa",
                "vertex": [[index * 20, 0], [index * 20 + 10, 0], [index * 20 + 10, 10], [index * 20, 10]],
                "children": [],
            }
            for index in range(3)
        ]
        + [
            {
                "name": f"TOP.U_T{index}",
                "module_name": "T",
                "direction": 0,
                "color": "#bbbbbb",
                "vertex": [[80 + index * 20, 0], [90 + index * 20, 0], [90 + index * 20, 10], [80 + index * 20, 10]],
                "children": [],
            }
            for index in range(2)
        ],
    }
    pingroup = [
        [
            {
                "parent_inst": f"TOP.U_S{index}",
                "parent_module": "S",
                "pingroup_name": "seed",
                "scope": [],
                "successors": [],
                "width": 1.0,
            }
            for index in range(3)
        ]
        + [
            {
                "parent_inst": "TOP.U_T0",
                "parent_module": "T",
                "pingroup_name": "partial",
                "scope": [],
                "successors": [],
                "width": 1.0,
            }
        ],
        [
            {
                "parent_inst": "TOP.U_T1",
                "parent_module": "T",
                "pingroup_name": "partial",
                "scope": [],
                "successors": [],
                "width": 1.0,
            }
        ],
    ]
    block_path = tmp_path / "block.json"
    pingroup_path = tmp_path / "pingroup.json"
    block_path.write_text(json.dumps(block), encoding="utf-8")
    pingroup_path.write_text(json.dumps(pingroup), encoding="utf-8")

    calls = []
    original_mcts = assignment_solver_module.MCTSSolver

    class FakeMCTS:
        def __init__(self, *args, **kwargs):
            self.groups = kwargs["groups"]
            self.skipped_group_names = set(kwargs.get("skipped_group_names", set()))
            self.skipped_pin_names = set(kwargs.get("skipped_pin_names", set()))
            self.last_simulation_count = 0
            self.timing_profile = {}
            self.reward_evaluator = type("FakeReward", (), {"timing_profile": {}})()
            calls.append(self)

        def search(self):
            return {}

    assignment_solver_module.MCTSSolver = FakeMCTS
    try:
        solver = AssignmentSolver(
            str(block_path),
            str(pingroup_path),
            simulations=1,
            homology_skip_uncovered_groups=True,
            homology_skip_coverage_threshold=1.0,
            homology_group_commit_coverage_threshold=1.0,
            feedthrough_weight=0.0,
        )
        solver.solve()
    finally:
        assignment_solver_module.MCTSSolver = original_mcts

    first_call = calls[0]
    assert [group.name for group in first_call.groups] == ["S.seed"]
    assert first_call.skipped_group_names == {"T.partial"}
    assert first_call.skipped_pin_names == {"TOP.U_T0.partial", "TOP.U_T1.partial"}


def test_assignment_progress_no_longer_emits_periodic_log(tmp_path):
    """Assignment progress counters should not emit the old periodic log."""
    block_path, pingroup_path = write_case(tmp_path)
    solver = AssignmentSolver(str(block_path), str(pingroup_path), simulations=2)
    group = solver.homology.unassigned_groups()[0]
    solver.total_mcts_simulations = 1234
    output = StringIO()
    handler = logging.StreamHandler(output)
    logger = logging.getLogger("assignment_solver")
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        for _ in range(99):
            solver._print_assignment_progress(group)
        assert output.getvalue() == ""
        solver._print_assignment_progress(group)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    assert output.getvalue() == ""


def test_mcts_records_actual_simulation_count(tmp_path):
    """MCTS exposes the actual number of rollout simulations used by one search."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    mcts = MCTSSolver(
        placedb,
        segments,
        homology.unassigned_groups(),
        placedb.nets_list,
        simulations=5,
        search_mode="basic",
        basic_dynamic_simulations=False,
        feedthrough_weight=0.0,
    )

    mcts.search()

    assert mcts.last_simulation_count == 5


def test_skip_assignment_is_not_committed_or_greedy_fallback(tmp_path):
    """A true-skip MCTS action should not be converted into a real segment assignment."""
    block_path, pingroup_path = write_case(tmp_path)
    solver = AssignmentSolver(str(block_path), str(pingroup_path), simulations=2)
    group = solver.homology.unassigned_groups()[0]
    fallback_calls = []

    def fake_greedy(unused_group, unused_reason):
        fallback_calls.append((unused_group.name, unused_reason))
        return True

    solver._assign_group_greedily = fake_greedy
    committed = solver._commit_contained_groups(
        [group],
        list(group.pins),
        {group.name: SKIP_SEGMENT_ID},
    )

    assert committed == 0
    assert fallback_calls == []
    assert not group.assigned
    assert all(pin.assigned_segment_id is None for pin in group.pins)


def test_zero_coverage_threshold_disables_uncovered_group_skip(tmp_path):
    """Coverage threshold 0 keeps every group placeable even when skip mode is enabled."""
    block_path, pingroup_path = write_low_committable_case(tmp_path)
    solver = AssignmentSolver(
        str(block_path),
        str(pingroup_path),
        simulations=2,
        enable_segment_subdivision=False,
        homology_skip_uncovered_groups=True,
        homology_skip_coverage_threshold=0.0,
        homology_group_commit_coverage_threshold=1.0,
    )
    groups = solver.homology.unassigned_groups()
    assert solver._uncovered_group_names(groups, set(), solver.homology_skip_coverage_threshold) == set()


def test_solver_assigns_all_pins_and_respects_capacity(tmp_path):
    """Test helper."""
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
    """Test helper."""
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


def test_partial_homology_group_can_commit_by_coverage_threshold(tmp_path):
    """A partially visible homology group can be committed as a whole above threshold."""
    block = {
        "name": "TOP",
        "module_name": "TOP",
        "direction": 0,
        "color": "#000000",
        "vertex": [[0, 0], [120, 0], [120, 120], [0, 120]],
        "children": [
            {
                "name": "TOP.U_A0",
                "module_name": "A",
                "direction": 0,
                "color": "#aaaaaa",
                "vertex": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "children": [],
            },
            {
                "name": "TOP.U_A1",
                "module_name": "A",
                "direction": 0,
                "color": "#aaaaaa",
                "vertex": [[20, 0], [30, 0], [30, 10], [20, 10]],
                "children": [],
            },
            {
                "name": "TOP.U_A2",
                "module_name": "A",
                "direction": 0,
                "color": "#aaaaaa",
                "vertex": [[40, 0], [50, 0], [50, 10], [40, 10]],
                "children": [],
            },
            {
                "name": "TOP.U_B0",
                "module_name": "B",
                "direction": 0,
                "color": "#bbbbbb",
                "vertex": [[80, 0], [90, 0], [90, 10], [80, 10]],
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
            }
        ],
        [
            {
                "parent_inst": "TOP.U_A2",
                "parent_module": "A",
                "pingroup_name": "p",
                "scope": [],
                "successors": [],
                "width": 1.0,
            }
        ],
    ]
    block_path = tmp_path / "block.json"
    pingroup_path = tmp_path / "pingroup.json"
    block_path.write_text(json.dumps(block), encoding="utf-8")
    pingroup_path.write_text(json.dumps(pingroup), encoding="utf-8")

    solver = AssignmentSolver(
        str(block_path),
        str(pingroup_path),
        simulations=1,
        enable_segment_subdivision=False,
        homology_group_commit_coverage_threshold=1 / 3,
    )
    group = solver.homology.pin_groups["A.p"]
    pins_in = [solver.placedb.pin_dict["TOP.U_A0.p"], solver.placedb.pin_dict["TOP.U_B0.q"]]

    committed = solver._commit_contained_groups(
        [group],
        pins_in,
        {"A.p": "A:S0"},
    )

    assert committed == 1
    assert group.assigned
    assert all(pin.assigned_segment_id == "A:S0" for pin in group.pins)


def write_low_committable_case(tmp_path: Path) -> tuple[Path, Path]:
    """Build a case where one net touches many partially covered homology groups."""
    children = []
    pingroup_net = []
    singleton_nets = []
    for index, module_name in enumerate(["A", "B", "C", "D"]):
        for inst_index in range(2):
            inst_name = f"TOP.U_{module_name}{inst_index}"
            x0 = index * 30 + inst_index * 10
            children.append(
                {
                    "name": inst_name,
                    "module_name": module_name,
                    "direction": 0,
                    "color": "#aaaaaa",
                    "vertex": [[x0, 0], [x0 + 8, 0], [x0 + 8, 8], [x0, 8]],
                    "children": [],
                }
            )
            pin = {
                "parent_inst": inst_name,
                "parent_module": module_name,
                "pingroup_name": "p",
                "scope": [],
                "successors": [],
                "width": 1.0,
            }
            if inst_index == 0:
                pingroup_net.append(pin)
            else:
                singleton_nets.append([pin])
    block = {
        "name": "TOP",
        "module_name": "TOP",
        "direction": 0,
        "color": "#000000",
        "vertex": [[0, 0], [160, 0], [160, 80], [0, 80]],
        "children": children,
    }
    pingroup = [pingroup_net] + singleton_nets
    block_path = tmp_path / "block.json"
    pingroup_path = tmp_path / "pingroup.json"
    block_path.write_text(json.dumps(block), encoding="utf-8")
    pingroup_path.write_text(json.dumps(pingroup), encoding="utf-8")
    return block_path, pingroup_path


def test_low_committable_ratio_still_builds_mcts_tree(tmp_path):
    """Low committable/search group ratio should no longer skip local MCTS trees."""
    block_path, pingroup_path = write_low_committable_case(tmp_path)
    calls = {"mcts": 0}

    class FakeMCTS:
        def __init__(self, *args, **kwargs):
            calls["mcts"] += 1

        def search(self):
            return {"A.p": "A:S0", "B.p": "B:S0", "C.p": "C:S0", "D.p": "D:S0"}

    original_mcts = assignment_solver_module.MCTSSolver
    assignment_solver_module.MCTSSolver = FakeMCTS
    try:
        solver = AssignmentSolver(
            str(block_path),
            str(pingroup_path),
            simulations=2,
            enable_segment_subdivision=False,
            homology_group_commit_coverage_threshold=1.0,
        )
        result = solver.solve()
    finally:
        assignment_solver_module.MCTSSolver = original_mcts

    assert calls["mcts"] >= 1
    assert "skipped_mcts_tree_count" not in result["summary"]
    assert "skipped_mcts_trees" not in result
    assert result["summary"]["assigned_pin_count"] == result["summary"]["pin_count"]


def test_high_committable_ratio_builds_mcts_tree(tmp_path):
    """High committable/search group ratio should still invoke MCTS and commit groups."""
    block_path, pingroup_path = write_case(tmp_path)
    calls = {"mcts": 0}

    class FakeMCTS:
        def __init__(self, *args, **kwargs):
            calls["mcts"] += 1

        def search(self):
            return {"A.p": "A:S0", "B.q": "B:S0", "B.r": "B:S0"}

    original_mcts = assignment_solver_module.MCTSSolver
    assignment_solver_module.MCTSSolver = FakeMCTS
    try:
        solver = AssignmentSolver(
            str(block_path),
            str(pingroup_path),
            simulations=2,
            enable_segment_subdivision=False,
            homology_group_commit_coverage_threshold=1.0,
        )
        result = solver.solve()
    finally:
        assignment_solver_module.MCTSSolver = original_mcts

    assert calls["mcts"] == 1
    assert "skipped_mcts_tree_count" not in result["summary"]
    assert result["summary"]["assigned_pin_count"] == result["summary"]["pin_count"]


def test_batch_analysis_does_not_skip_low_committable_trees(tmp_path):
    """The analysis script should report low-yield trees instead of skipping them."""
    block_path, pingroup_path = write_low_committable_case(tmp_path)

    report = analyze_batches(
        block_path,
        pingroup_path,
        coverage_threshold=1.0,
    )

    assert report["summary"]["mcts_tree_count"] == 4
    assert "skipped_mcts_tree_count" not in report["summary"]
    assert "skipped_tree_reports" not in report
    assert report["tree_reports"][0]["committable_group_ratio_of_search_groups"] == 0.25


def test_batch_analysis_true_skip_uses_effective_mcts_depth(tmp_path):
    """True-skip groups should be raw candidates but not effective MCTS depth."""
    block_path, pingroup_path = write_low_committable_case(tmp_path)

    report = analyze_batches(
        block_path,
        pingroup_path,
        coverage_threshold=0.0,
        skip_uncovered_groups=True,
        skip_coverage_threshold=1.0,
    )

    first_tree = report["tree_reports"][0]
    assert first_tree["raw_mcts_depth"] == 4
    assert first_tree["mcts_depth"] == 1
    assert first_tree["raw_committable_group_count"] == 4
    assert first_tree["committable_group_count"] == 1
    assert first_tree["true_skipped_group_count"] == 3
    assert first_tree["raw_search_pin_count"] == 8
    assert first_tree["search_pin_count"] == 2
    assert report["summary"]["true_skipped_group_visits"] == 6
    assert report["summary"]["total_mcts_search_group_visits"] == 4


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
        basic_depth1_simulations=0,
        basic_depth2_simulations=0,
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
        basic_depth1_simulations=0,
        basic_depth2_simulations=0,
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
        basic_depth1_simulations=0,
        basic_depth2_simulations=0,
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
        basic_depth1_simulations=0,
        basic_depth2_simulations=0,
    )

    assert abs(mcts._total_search_space_factor(usage) - 1.6) < 1e-9


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
        basic_depth1_simulations=0,
        basic_depth2_simulations=0,
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
        basic_depth1_simulations=0,
        basic_depth2_simulations=0,
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
        basic_depth1_simulations=0,
        basic_depth2_simulations=0,
    )
    assert disabled._basic_simulation_budget(usage) == 17


def test_basic_mcts_depth_overrides_and_depth1_bypasses_pruning(tmp_path):
    """Depth-1/2 basic trees use manual budgets and depth-1 keeps all candidates."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    groups = homology.unassigned_groups()
    usage = segments.snapshot_usage()

    depth1 = MCTSSolver(
        placedb,
        segments,
        groups[:1],
        placedb.nets_list,
        simulations=4096,
        search_mode="basic",
        basic_depth1_simulations=32,
        basic_depth2_simulations=256,
        basic_disable_pruning_depth_limit=1,
        candidate_min_count=1,
        candidate_top_k=1,
    )
    depth1_profile = depth1._search_profile(usage)
    assert depth1_profile.depth == 1
    assert depth1._basic_simulation_budget(usage, depth1_profile) == 32

    depth1._active_basic_depth = 1
    raw_candidates = depth1._raw_feasible_segments(groups[0], usage)
    pruned_candidates = depth1._candidate_segments(groups[0], usage)
    assert len(raw_candidates) > 1
    assert len(pruned_candidates) == len(raw_candidates)

    depth2 = MCTSSolver(
        placedb,
        segments,
        groups[:2],
        placedb.nets_list,
        simulations=4096,
        search_mode="basic",
        basic_depth1_simulations=32,
        basic_depth2_simulations=256,
    )
    depth2_profile = depth2._search_profile(usage)
    assert depth2_profile.depth == 2
    assert depth2._basic_simulation_budget(usage, depth2_profile) == 256


def test_basic_depth1_search_uses_reward_greedy_without_simulation(tmp_path):
    """Depth-1 basic search should score all candidates directly with the reward evaluator."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    group = homology.pin_groups["A.p"]
    candidates = segments.candidates_for_module(group.module_name)
    preferred = candidates[-1]
    preferred_points = {
        preferred.instances[pin.parent_inst].midpoint
        for pin in group.pins
    }

    class FakeRewardEvaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, temporary_locations):
            self.calls += 1
            points = set(temporary_locations.values())
            return 10.0 if preferred_points <= points else 0.0

        def close(self):
            pass

    mcts = MCTSSolver(
        placedb,
        segments,
        [group],
        placedb.nets_list,
        simulations=4096,
        search_mode="basic",
        enable_candidate_pruning=True,
        candidate_min_count=1,
        candidate_top_k=1,
    )
    fake_evaluator = FakeRewardEvaluator()
    mcts.reward_evaluator = fake_evaluator
    mcts._simulate = lambda _node: (_ for _ in ()).throw(
        AssertionError("depth-1 search should not run simulations")
    )

    assignment = mcts.search()

    assert assignment == {group.name: preferred.segment_id}
    assert fake_evaluator.calls == len(candidates)


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
        basic_depth1_simulations=0,
        basic_depth2_simulations=0,
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


def test_hybrid_ultradeep_limits_expanded_depth_and_fast_completes(tmp_path):
    """Ultra-deep Hybrid trees should search only a bounded prefix before completion."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    base_groups = homology.unassigned_groups()
    groups = base_groups * 40
    mcts = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=100,
        search_mode="hybrid",
        hybrid_basic_depth_limit=0,
        hybrid_ultradeep_depth=20,
        hybrid_enable_ultradeep_profile=True,
        hybrid_max_expanded_depth=5,
        hybrid_ultradeep_beam_width=1,
        hybrid_ultradeep_min_layer_simulations=1,
        hybrid_ultradeep_max_layer_simulations=2,
        hybrid_max_tree_simulations=20,
        hybrid_enable_layer_early_stop=False,
        enable_candidate_pruning=False,
    )

    calls = {"simulate": 0, "fast": 0}

    def fake_simulate(_node):
        calls["simulate"] += 1
        return float(calls["simulate"])

    def fake_fast(assignments, _usage):
        calls["fast"] += 1
        completed = dict(assignments)
        for group in groups:
            completed.setdefault(group.name, "A:S0" if group.module_name == "A" else "B:S0")
        return completed

    mcts._simulate = fake_simulate
    mcts._complete_fast_by_heuristic = fake_fast
    assignment = mcts.search()

    assert mcts.last_search_diagnostics["route"] == "ultradeep"
    assert mcts.last_search_diagnostics["expanded_depth"] == 5
    assert calls["simulate"] <= 10
    assert calls["fast"] == 1
    assert set(assignment) == {group.name for group in base_groups}


def test_hybrid_can_disable_ultradeep_profile(tmp_path):
    """Disabling the ultra-deep profile should fall back to the prior hybrid route."""
    block_path, pingroup_path = write_case(tmp_path)
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    segments = SegmentManager(placedb)
    homology = HomologyManager(placedb)
    groups = homology.unassigned_groups() * 8
    mcts = MCTSSolver(
        placedb,
        segments,
        groups,
        placedb.nets_list,
        simulations=10,
        search_mode="hybrid",
        hybrid_basic_depth_limit=0,
        hybrid_enable_ultradeep_profile=False,
        hybrid_ultradeep_depth=5,
        hybrid_min_layer_simulations=1,
        hybrid_max_layer_simulations=1,
        hybrid_max_tree_simulations=3,
        hybrid_enable_layer_early_stop=False,
        enable_candidate_pruning=False,
    )

    mcts._simulate = lambda _node: 1.0
    mcts.search()

    assert mcts.last_search_diagnostics["route"] != "ultradeep"
    assert mcts.last_search_diagnostics["expanded_depth"] == len(groups)


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


def test_assignment_solver_selects_feedthrough_reward_source(tmp_path):
    """Reward FT context can use predict or evaluate source according to config."""
    block_path, pingroup_path = write_case(tmp_path)
    created = []

    class FakeContext(FakeFeedthroughContext):
        def __init__(self, _placedb, source_dir, **kwargs):
            super().__init__()
            created.append((Path(source_dir), kwargs.get("role")))

    original_context = assignment_solver_module.FeedthroughContext
    assignment_solver_module.FeedthroughContext = FakeContext
    try:
        predict_dir = tmp_path / "predict"
        evaluate_dir = tmp_path / "evaluate"
        solver = AssignmentSolver(
            str(block_path),
            str(pingroup_path),
            simulations=2,
            feedthrough_weight=1.0,
            feedthrough_predict_source_dir=predict_dir,
            feedthrough_evaluate_source_dir=evaluate_dir,
            feedthrough_reward_source="evaluate",
            enable_feedthrough=True,
        )
        solver.solve()
        solver.close_feedthrough_context()

        default_solver = AssignmentSolver(
            str(block_path),
            str(pingroup_path),
            simulations=2,
            feedthrough_weight=1.0,
            feedthrough_predict_source_dir=predict_dir,
            feedthrough_evaluate_source_dir=evaluate_dir,
            enable_feedthrough=True,
        )
        default_solver.solve()
        default_solver.close_feedthrough_context()
    finally:
        assignment_solver_module.FeedthroughContext = original_context

    assert created[0] == (evaluate_dir, "evaluate")
    assert created[1] == (evaluate_dir, "evaluate")


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


def test_single_pin_nets_are_skipped_for_metric_feedthrough(tmp_path):
    """Single-pin nets should remain in records but skip HPWL/FT evaluation work."""
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
                "vertex": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "children": [],
            },
            {
                "name": "TOP.U_A1",
                "module_name": "A",
                "direction": 0,
                "color": "#aaaaaa",
                "vertex": [[60, 0], [70, 0], [70, 10], [60, 10]],
                "children": [],
            },
        ],
    }
    pingroup = [
        [
            {
                "parent_inst": "TOP.U_A0",
                "parent_module": "A",
                "pingroup_name": "single",
                "scope": [],
                "successors": [],
                "width": 1.0,
            }
        ],
        [
            {
                "parent_inst": "TOP.U_A0",
                "parent_module": "A",
                "pingroup_name": "pair",
                "scope": [],
                "successors": [],
                "width": 1.0,
            },
            {
                "parent_inst": "TOP.U_A1",
                "parent_module": "A",
                "pingroup_name": "pair",
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

    placedb = PlaceDB(str(block_path), str(pingroup_path))
    context = FakeFeedthroughContext()
    metrics = final_net_metrics(
        placedb,
        tmp_path,
        enable_feedthrough=True,
        auto_build_feedthrough=False,
        feedthrough_context=context,
    )
    summary = summarize_metrics(metrics)

    assert len(metrics) == 2
    assert [metric.pin_count for metric in metrics] == [1, 2]
    assert len(context.calls) == 1
    assert summary["net_count"] == 2
    assert summary["metric_net_count"] == 1
    assert summary["skipped_single_pin_net_count"] == 1


def test_overflow_fallback_completes_assignment_when_pin_is_wider_than_edge(tmp_path):
    """Test helper."""
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
