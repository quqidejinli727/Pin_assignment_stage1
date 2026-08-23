from pathlib import Path

from homology import HomologyManager
from PlaceDB import PlaceDB
from segment import SegmentManager
from stage1_tree_parallel_experiment import (
    TreeDescription,
    build_tree_conflict_graph,
    extract_tree_descriptions,
    run_tree_experiment,
    tree_seed,
    weighted_greedy_batches,
)

from tests.test_geometry_and_solver import write_case


def test_extract_reuses_seed_local_tree_description(tmp_path: Path):
    block, pingroup = write_case(tmp_path)
    trees, analysis = extract_tree_descriptions(block, pingroup)
    assert [tree.tree_index for tree in trees] == list(range(len(trees)))
    assert len(trees) == analysis["summary"]["mcts_tree_count"]
    assert trees[0].seed_group in trees[0].search_group_names
    assert trees[0].committable_group_names
    assert tree_seed(7, 3) == tree_seed(7, 3)


def test_tree_conflict_reasons_are_not_global_module_components(tmp_path: Path):
    block, pingroup = write_case(tmp_path)
    placedb = PlaceDB(str(block), str(pingroup))
    homology = HomologyManager(placedb, use_fanout_reuse_for_sorting=True)
    segments = SegmentManager(placedb)
    base = TreeDescription(
        tree_index=0,
        seed_group="A.p",
        related_net_ids=(0,),
        search_group_names=("A.p",),
        committable_group_names=("A.p",),
        deferred_group_names=(),
        depth=1,
        search_pin_count=2,
        related_net_count=1,
        estimated_work=1.0,
    )
    other = TreeDescription(
        tree_index=1,
        seed_group="A.p",
        related_net_ids=(0,),
        search_group_names=("A.p",),
        committable_group_names=("A.p",),
        deferred_group_names=(),
        depth=1,
        search_pin_count=2,
        related_net_count=1,
        estimated_work=1.0,
    )
    graph = build_tree_conflict_graph([base, other], placedb, homology, segments)
    reasons = set(graph.edge_reasons[(0, 1)])
    assert {"group_overlap", "net_topology", "module_type", "candidate_abstract_segment"} <= reasons
    assert len(graph.trees) == 2
    batches = weighted_greedy_batches(
        graph,
        max_trees_per_batch=2,
        max_estimated_work=8,
    )
    assert len(batches) == 1
    assert batches[0].tree_indices == (0, 1)


def test_tree_experiment_hashes_without_fallback(tmp_path: Path):
    block, pingroup = write_case(tmp_path)
    report = run_tree_experiment(
        block,
        pingroup,
        workers=1,
        simulations=2,
        max_trees_per_batch=2,
        max_estimated_work=8,
        repetitions=1,
    )
    assert report["correctness"]["naive"]["assignment_hash_equal_to_serial"]
    assert report["correctness"]["conflict_aware"]["assignment_hash_equal_to_serial"]
    assert report["correctness"]["naive"]["no_fallback_used"]
    assert report["correctness"]["conflict_aware"]["no_fallback_used"]
