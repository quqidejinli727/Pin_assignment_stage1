from pathlib import Path

from stage1_parallel_waves import (
    _apply_pin_state,
    _pin_state,
    _pin_state_delta,
    _solver_kwargs,
    run_parallel_waves,
    schedule_waves,
)
from stage1_tree_parallel_experiment import TreeDescription, TreeConflictGraph

from assignment_solver import AssignmentSolver
from tests.test_geometry_and_solver import write_case


def test_wave_scheduler_is_stable_and_respects_worker_width():
    trees = [
        TreeDescription(i, f"G{i}", (), (f"G{i}",), (f"G{i}",), (), 1, 1, 0, 1.0)
        for i in range(6)
    ]
    graph = TreeConflictGraph(tuple(trees), {})
    waves = schedule_waves(trees, graph, workers=2, trees_per_worker=2, max_estimated_work=20)
    assert [index for wave in waves for index in wave.tree_indices] == list(range(6))
    assert all(len(wave.tree_indices) <= 4 for wave in waves)
    assert all(wave.hard_conflicts == 0 for wave in waves)


def test_budget_profile_follows_requested_simulations():
    options = _solver_kwargs("block.json", "pingroup.json", simulations=2, seed=7)
    assert options["mcts_basic_min_simulations"] == 2
    assert options["mcts_basic_depth1_simulations"] == 2
    assert options["mcts_basic_depth2_simulations"] == 2
    assert options["mcts_hybrid_min_layer_simulations"] == 2
    assert options["mcts_hybrid_max_layer_simulations"] == 16
    assert options["mcts_hybrid_max_tree_simulations"] == 32


def test_wave_sync_contains_groups_segments_and_pin_state(tmp_path: Path):
    block, pingroup = write_case(tmp_path)
    solver = AssignmentSolver(
        str(block),
        str(pingroup),
        simulations=2,
        enable_feedthrough=False,
    )
    state = _pin_state(solver)
    assert set(state) == {"assigned_groups", "segment_used_width", "pins"}
    assert state["segment_used_width"]
    assert state["pins"]
    assert all("scope" in item for item in state["pins"].values())


def test_dynamic_delta_sync_replays_monotonic_commit(tmp_path: Path):
    block, pingroup = write_case(tmp_path)
    source = AssignmentSolver(str(block), str(pingroup), simulations=2)
    target = AssignmentSolver(str(block), str(pingroup), simulations=2)
    group = next(iter(source.homology.pin_groups.values()))
    segment = source.segment_manager.candidates_for_module(group.module_name)[0]
    source._commit_group_assignment(group, segment.segment_id)

    state = _pin_state_delta(
        source,
        [group.name],
        from_version=0,
        to_version=1,
    )
    _apply_pin_state(target.homology, target.segment_manager, target.placedb, state)

    replayed = target.homology.pin_groups[group.name]
    assert replayed.assigned_segment_id == segment.segment_id
    assert target.segment_manager.abstract_segments[segment.segment_id].used_width == (
        source.segment_manager.abstract_segments[segment.segment_id].used_width
    )
    assert {
        pin.full_name: (pin.x, pin.y, pin.assigned_segment_id)
        for pin in replayed.pins
    } == {
        pin.full_name: (pin.x, pin.y, pin.assigned_segment_id)
        for pin in group.pins
    }


def test_real_wave_adapter_reuses_solver_and_reports_completion(tmp_path: Path):
    block, pingroup = write_case(tmp_path)
    report = run_parallel_waves(
        block,
        pingroup,
        workers=1,
        simulations=2,
        trees_per_worker=2,
        max_estimated_work=8,
        final_greedy_completion=True,
    )
    assert report["parallel_waves"]["proposal_count"] > 0
    assert report["parallel_waves"]["total_mcts_simulations"] == report["baseline_assignment_solver"]["total_mcts_simulations"]
    assert sum(item["simulations"] for item in report["parallel_waves"]["tree_simulations"]) == report["parallel_waves"]["total_mcts_simulations"]
    baseline_tree_simulations = report["baseline_assignment_solver"]["tree_simulations"]
    if baseline_tree_simulations:
        assert sum(item["simulations"] for item in baseline_tree_simulations) == report["baseline_assignment_solver"]["total_mcts_simulations"]
    assert report["performance"]["simulation_budget_equal"]
    assert report["parallel_waves"]["final_greedy_added_groups"] >= 0
    assert report["correctness"]["parallel_complete"]
