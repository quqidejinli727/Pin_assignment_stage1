from pathlib import Path

from config import DEFAULT_CONFIG
from stage1_dynamic_queue_benchmark import HYBRID_DEFAULT_FIELDS, solver_kwargs
from stage1_dynamic_queue_experiment import run_dynamic_queue
from tests.test_geometry_and_solver import write_case


def test_hybrid_benchmark_keeps_repository_simulation_schedule(tmp_path: Path):
    block, pingroup = write_case(tmp_path)
    options = solver_kwargs(
        block,
        pingroup,
        simulations=1024,
        seed=7,
        search_mode="hybrid",
        feedthrough_weight=0.1,
        feedthrough_source=tmp_path,
    )

    assert options["simulations"] == 1024
    assert all(
        options[name] == getattr(DEFAULT_CONFIG, name)
        for name in HYBRID_DEFAULT_FIELDS
    )


def test_dynamic_queue_two_workers_completes_without_capacity_violation(tmp_path: Path):
    block, pingroup = write_case(tmp_path)
    report = run_dynamic_queue(
        block,
        pingroup,
        workers=2,
        simulations=2,
        random_seed=7,
        solver_overrides={
            "mcts_search_mode": "hybrid",
            "enable_feedthrough": False,
            "feedthrough_weight": 0.0,
        },
    )

    assert report["result"]["summary"]["unassigned_group_count"] == 0
    assert report["result"]["assignment_issues"] == []
    assert report["result"]["capacity_violations"] == []
    assert report["scheduler"]["dispatched_tree_count"] > 0
    assert report["scheduler"]["sync_mode"] == "monotonic_delta"
    assert report["commit"]["counts"]["direct_commit"] > 0
