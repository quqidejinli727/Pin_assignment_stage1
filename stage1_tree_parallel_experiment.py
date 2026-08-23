"""Tree-level Stage1 MCTS parallelism experiment.

The existing solver constructs a local tree for each seed pingroup, commits
the covered groups, and then advances to the next stable seed.  This module
replays that tree *description* with :func:`analyze_mcts_tree_batches.analyze_batches`
and schedules those same trees; it does not collapse all groups into one
global MCTS tree.  Worker processes search trees independently and return
proposals.  The parent validates and commits proposals in tree/group order.

The experiment is deliberately shadow-only.  ``AssignmentSolver`` defaults
are unchanged, and a rejected or stale proposal is recorded rather than
repaired by greedy or overflow fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import socket
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from analyze_mcts_tree_batches import analyze_batches
from homology import HomologyManager
from mcts import MCTSSolver, SKIP_SEGMENT_ID
from PlaceDB import PlaceDB
from segment import SegmentManager


HARD_CONFLICTS = {
    "group_overlap",
    "candidate_abstract_segment",
    "segment_instance",
    "capacity_write_conflict",
}
CONFLICT_WEIGHTS = {
    "group_overlap": 1000.0,
    "capacity_write_conflict": 900.0,
    "segment_instance": 500.0,
    "candidate_abstract_segment": 100.0,
    "net_topology": 8.0,
    "module_type": 2.0,
}


@dataclass(frozen=True)
class TreeDescription:
    """Stable tree-local payload extracted from the existing coverage flow."""

    tree_index: int
    seed_group: str
    related_net_ids: Tuple[int, ...]
    search_group_names: Tuple[str, ...]
    committable_group_names: Tuple[str, ...]
    deferred_group_names: Tuple[str, ...]
    depth: int
    search_pin_count: int
    related_net_count: int
    estimated_work: float

    @property
    def skipped_group_names(self) -> Tuple[str, ...]:
        """Compatibility alias for older experiment reports."""
        return self.deferred_group_names


@dataclass
class TreeConflictGraph:
    trees: Tuple[TreeDescription, ...]
    edge_reasons: Dict[Tuple[int, int], Tuple[str, ...]]

    def reason_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for reasons in self.edge_reasons.values():
            for reason in reasons:
                counts[reason] += 1
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class TreeBatch:
    batch_index: int
    tree_indices: Tuple[int, ...]
    estimated_work: float


def _edge_key(left: int, right: int) -> Tuple[int, int]:
    return (left, right) if left < right else (right, left)


def tree_seed(random_seed: int, tree_index: int) -> int:
    """Bind randomness to stable tree identity, never to its batch."""

    return int(random_seed) + int(tree_index)


def extract_tree_descriptions(
    block_json: str | Path,
    pingroup_json: str | Path,
    *,
    coverage_threshold: float = 1.0,
    skip_uncovered_groups: bool = True,
    skip_coverage_threshold: float = 1.0,
    use_fanout_reuse_for_sorting: bool = False,
) -> Tuple[List[TreeDescription], dict]:
    """Replay the existing analyzer's stable seed-local tree construction.

    The analyzer performs the same related-net, coverage, true-skip and
    committable-group simulation as ``AssignmentSolver``.  We retain only the
    compact payload needed by workers and keep its full report for auditing.
    """

    analyzer_options = {
        "coverage_threshold": coverage_threshold,
        "use_fanout_reuse_for_sorting": use_fanout_reuse_for_sorting,
        "skip_uncovered_groups": skip_uncovered_groups,
        "skip_coverage_threshold": skip_coverage_threshold,
    }
    try:
        # New analyzer versions gate the compact group details behind this
        # flag; older branch snapshots always include those fields.
        analysis = analyze_batches(
            block_json,
            pingroup_json,
            include_tree_details=True,
            **analyzer_options,
        )
    except TypeError as error:
        if "include_tree_details" not in str(error):
            raise
        analysis = analyze_batches(block_json, pingroup_json, **analyzer_options)
    trees: List[TreeDescription] = []
    for report in analysis["tree_reports"]:
        # Use tree_index (rather than candidate_tree_index) because it is the
        # stable order used by the analyzer and by our deterministic commit.
        index = int(report["tree_index"])
        trees.append(
            TreeDescription(
                tree_index=index,
                seed_group=str(report["seed_group"]),
                related_net_ids=tuple(int(item) for item in report["related_net_ids"]),
                search_group_names=tuple(
                    item["name"] for item in report["search_groups"]
                ),
                committable_group_names=tuple(
                    item["name"] for item in report["committable_groups"]
                ),
                deferred_group_names=tuple(
                    item["name"]
                    for item in report.get(
                        "deferred_from_search_groups",
                        report["true_skipped_groups"],
                    )
                ),
                depth=int(report["mcts_depth"]),
                search_pin_count=int(report["search_pin_count"]),
                related_net_count=int(report["related_net_count"]),
                # Depth is a useful stable proxy for MCTS work.  The actual
                # worker timings remain the authoritative measurement.
                estimated_work=float(max(1, report["mcts_depth"])),
            )
        )
    trees.sort(key=lambda tree: tree.tree_index)
    return trees, analysis


def build_tree_conflict_graph(
    trees: Sequence[TreeDescription],
    placedb: PlaceDB,
    homology: HomologyManager,
    segment_manager: SegmentManager,
) -> TreeConflictGraph:
    """Build weighted tree-level conflicts without global module components."""

    groups_by_tree = {tree.tree_index: set(tree.search_group_names) for tree in trees}
    nets_by_tree = {tree.tree_index: set(tree.related_net_ids) for tree in trees}
    modules_by_tree: Dict[int, Set[str]] = {}
    candidates_by_tree: Dict[int, Set[str]] = {}
    instances_by_tree: Dict[int, Set[Tuple[str, str]]] = {}
    demands_by_tree: Dict[int, Dict[str, float]] = {}
    for tree in trees:
        modules: Set[str] = set()
        candidates: Set[str] = set()
        instances: Set[Tuple[str, str]] = set()
        demands: Dict[str, float] = defaultdict(float)
        for group_name in tree.search_group_names:
            group = homology.pin_groups[group_name]
            modules.add(group.module_name)
            group_candidates = segment_manager.candidates_for_module(group.module_name)
            for segment in group_candidates:
                candidates.add(segment.segment_id)
                demands[segment.segment_id] += group.max_pin_width
                for pin in group.pins:
                    if (pin.parent_inst, segment.segment_id) in segment_manager.instance_lookup:
                        instances.add((pin.parent_inst, segment.segment_id))
        modules_by_tree[tree.tree_index] = modules
        candidates_by_tree[tree.tree_index] = candidates
        instances_by_tree[tree.tree_index] = instances
        demands_by_tree[tree.tree_index] = dict(demands)

    reasons: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
    for left_pos, left in enumerate(trees):
        for right in trees[left_pos + 1 :]:
            left_index, right_index = left.tree_index, right.tree_index
            if groups_by_tree[left_index] & groups_by_tree[right_index]:
                reasons[_edge_key(left_index, right_index)].add("group_overlap")
            if nets_by_tree[left_index] & nets_by_tree[right_index]:
                reasons[_edge_key(left_index, right_index)].add("net_topology")
            if modules_by_tree[left_index] & modules_by_tree[right_index]:
                reasons[_edge_key(left_index, right_index)].add("module_type")
            if candidates_by_tree[left_index] & candidates_by_tree[right_index]:
                reasons[_edge_key(left_index, right_index)].add(
                    "candidate_abstract_segment"
                )
            if instances_by_tree[left_index] & instances_by_tree[right_index]:
                reasons[_edge_key(left_index, right_index)].add("segment_instance")
            shared_segments = (
                demands_by_tree[left_index].keys()
                & demands_by_tree[right_index].keys()
            )
            if any(
                demands_by_tree[left_index][segment_id]
                + demands_by_tree[right_index][segment_id]
                > segment_manager.abstract_segments[segment_id].capacity + 1e-9
                for segment_id in shared_segments
            ):
                reasons[_edge_key(left_index, right_index)].add(
                    "capacity_write_conflict"
                )
    return TreeConflictGraph(
        trees=tuple(trees),
        edge_reasons={key: tuple(sorted(value)) for key, value in sorted(reasons.items())},
    )


def conflict_cost(reasons: Iterable[str]) -> float:
    return sum(CONFLICT_WEIGHTS.get(reason, 1.0) for reason in reasons)


def _batch_conflict_cost(
    batch_indices: Sequence[int],
    tree_index: int,
    graph: TreeConflictGraph,
) -> Tuple[float, Counter]:
    total = 0.0
    counter: Counter = Counter()
    for existing in batch_indices:
        reasons = graph.edge_reasons.get(_edge_key(existing, tree_index), ())
        total += conflict_cost(reasons)
        counter.update(reasons)
    return total, counter


def naive_consecutive_batches(
    trees: Sequence[TreeDescription],
    *,
    max_trees_per_batch: int = 8,
    max_estimated_work: float = 80.0,
) -> List[TreeBatch]:
    """Pack stable tree order consecutively under count/work limits."""

    batches: List[TreeBatch] = []
    current: List[int] = []
    work = 0.0
    for tree in trees:
        if current and (
            len(current) >= max_trees_per_batch
            or work + tree.estimated_work > max_estimated_work
        ):
            batches.append(TreeBatch(len(batches), tuple(current), work))
            current, work = [], 0.0
        current.append(tree.tree_index)
        work += tree.estimated_work
    if current:
        batches.append(TreeBatch(len(batches), tuple(current), work))
    return batches


def weighted_greedy_batches(
    graph: TreeConflictGraph,
    *,
    max_trees_per_batch: int = 8,
    max_estimated_work: float = 80.0,
) -> List[TreeBatch]:
    """Greedy weighted bin packing minimizing incremental conflict cost.

    Hard conflicts are not silently ignored or converted into global
    components.  They contribute a large score and are later validated as
    stale/rejected submissions in the parent.  This keeps enough batch width
    to amortize IPC while making the trade-off explicit.
    """

    tree_by_index = {tree.tree_index: tree for tree in graph.trees}
    ordered = sorted(
        graph.trees,
        key=lambda tree: (-tree.estimated_work, tree.tree_index),
    )
    batch_indices: List[List[int]] = []
    batch_work: List[float] = []
    for tree in ordered:
        choices = []
        for batch_index, indices in enumerate(batch_indices):
            if len(indices) >= max_trees_per_batch:
                continue
            if batch_work[batch_index] + tree.estimated_work > max_estimated_work:
                continue
            incremental, reasons = _batch_conflict_cost(indices, tree.tree_index, graph)
            # Load term is deliberately tiny relative to hard/soft conflict
            # weights; it only breaks ties toward balanced bins.
            projected = batch_work[batch_index] + tree.estimated_work
            score = incremental + projected / max(1.0, max_estimated_work)
            choices.append((score, projected, batch_index, reasons))
        if choices:
            _, projected, batch_index, _ = min(
                choices,
                key=lambda item: (item[0], item[1], item[2]),
            )
            batch_indices[batch_index].append(tree.tree_index)
            batch_work[batch_index] = projected
        else:
            batch_indices.append([tree.tree_index])
            batch_work.append(tree.estimated_work)
    # Commit order is always stable tree order inside every task; batch index
    # only controls scheduling, not proposal application order.
    return [
        TreeBatch(index, tuple(sorted(indices)), batch_work[index])
        for index, indices in enumerate(batch_indices)
    ]


def summarize_batches(
    batches: Sequence[TreeBatch],
    graph: TreeConflictGraph,
) -> dict:
    reason_counts: Counter = Counter()
    hard_edges = soft_edges = 0
    conflict_cost_total = 0.0
    records = []
    for batch in batches:
        local_reasons: Counter = Counter()
        local_cost = 0.0
        for position, left in enumerate(batch.tree_indices):
            for right in batch.tree_indices[position + 1 :]:
                reasons = graph.edge_reasons.get(_edge_key(left, right), ())
                if reasons:
                    reason_counts.update(reasons)
                    local_reasons.update(reasons)
                    local_cost += conflict_cost(reasons)
                    conflict_cost_total += conflict_cost(reasons)
                    if any(reason in HARD_CONFLICTS for reason in reasons):
                        hard_edges += 1
                    else:
                        soft_edges += 1
        records.append(
            {
                "batch_index": batch.batch_index,
                "tree_indices": list(batch.tree_indices),
                "tree_count": len(batch.tree_indices),
                "estimated_work": batch.estimated_work,
                "conflict_cost": local_cost,
                "conflict_reasons": dict(sorted(local_reasons.items())),
            }
        )
    loads = [batch.estimated_work for batch in batches]
    return {
        "batch_count": len(batches),
        "tree_count": sum(len(batch.tree_indices) for batch in batches),
        "batch_tree_counts": [len(batch.tree_indices) for batch in batches],
        "estimated_work_loads": loads,
        "max_estimated_work": max(loads, default=0.0),
        "min_estimated_work": min(loads, default=0.0),
        "average_estimated_work": sum(loads) / len(loads) if loads else 0.0,
        "conflict_cost": conflict_cost_total,
        "intra_batch_hard_edge_count": hard_edges,
        "intra_batch_soft_edge_count": soft_edges,
        "intra_batch_conflict_reason_counts": dict(sorted(reason_counts.items())),
        "records": records,
    }


_TREE_WORKER: Dict[str, object] = {}


def _tree_worker_init(
    block_json: str,
    pingroup_json: str,
    mcts_options: Mapping[str, object],
) -> None:
    started = time.perf_counter()
    placedb = PlaceDB(block_json, pingroup_json)
    homology = HomologyManager(placedb, use_fanout_reuse_for_sorting=False)
    segment_manager = SegmentManager(placedb)
    _TREE_WORKER.clear()
    _TREE_WORKER.update(
        placedb=placedb,
        homology=homology,
        segment_manager=segment_manager,
        mcts_options=dict(mcts_options),
        init_seconds=time.perf_counter() - started,
        init_reported=False,
    )


def _tree_worker_task(
    task: Tuple[
        int,
        Tuple[int, ...],
        int,
        Tuple[Tuple[str, ...], ...],
        Tuple[Tuple[int, ...], ...],
        Tuple[Tuple[str, ...], ...],
    ]
) -> dict:
    batch_index, tree_indices, random_seed, group_names, net_ids, deferred_names = task
    placedb: PlaceDB = _TREE_WORKER["placedb"]  # type: ignore[assignment]
    homology: HomologyManager = _TREE_WORKER["homology"]  # type: ignore[assignment]
    segment_manager: SegmentManager = _TREE_WORKER["segment_manager"]  # type: ignore[assignment]
    options = dict(_TREE_WORKER["mcts_options"])  # type: ignore[arg-type]
    budget = int(options.pop("simulations", 16))
    started = time.perf_counter()
    tree_results = []
    for position, tree_index in enumerate(tree_indices):
        groups = [homology.pin_groups[name] for name in group_names[position]]
        nets = [placedb.nets_list[net_id] for net_id in net_ids[position]]
        search_started = time.perf_counter()
        mcts = MCTSSolver(
            placedb=placedb,
            segment_manager=segment_manager,
            groups=groups,
            nets=nets,
            simulations=budget,
            random_seed=tree_seed(random_seed, tree_index),
            enable_feedthrough=False,
            feedthrough_weight=0.0,
            deferred_group_names=deferred_names[position],
            excluded_pin_names=set(),
            **options,
        )
        proposal = mcts.search()
        search_seconds = time.perf_counter() - search_started
        tree_results.append(
            {
                "tree_index": tree_index,
                "proposal": dict(proposal),
                "search_seconds": search_seconds,
                "worker_simulations": getattr(mcts, "last_simulation_count", 0),
            }
        )
    init_seconds = 0.0
    if not bool(_TREE_WORKER["init_reported"]):
        init_seconds = float(_TREE_WORKER["init_seconds"])
        _TREE_WORKER["init_reported"] = True
    return {
        "batch_index": batch_index,
        "tree_indices": list(tree_indices),
        "trees": tree_results,
        "worker_seconds": time.perf_counter() - started,
        "initializer_seconds": init_seconds,
    }


def _make_tasks(
    batches: Sequence[TreeBatch],
    trees: Sequence[TreeDescription],
    random_seed: int,
) -> List[tuple]:
    tree_by_index = {tree.tree_index: tree for tree in trees}
    return [
        (
            batch.batch_index,
            batch.tree_indices,
            random_seed,
            tuple(tuple(tree_by_index[index].search_group_names) for index in batch.tree_indices),
            tuple(tuple(tree_by_index[index].related_net_ids) for index in batch.tree_indices),
            tuple(tuple(tree_by_index[index].deferred_group_names) for index in batch.tree_indices),
        )
        for batch in batches
    ]


def _normalized_commit(
    block_json: str,
    pingroup_json: str,
    trees: Sequence[TreeDescription],
    proposals: Sequence[dict],
) -> dict:
    """Commit only valid, non-stale proposals; never fallback."""

    placedb = PlaceDB(block_json, pingroup_json)
    homology = HomologyManager(placedb, use_fanout_reuse_for_sorting=False)
    segment_manager = SegmentManager(placedb)
    tree_by_index = {tree.tree_index: tree for tree in trees}
    proposal_by_tree: Dict[int, dict] = {}
    for task in proposals:
        for tree_result in task["trees"]:
            proposal_by_tree[int(tree_result["tree_index"])] = tree_result["proposal"]
    rejected: Counter = Counter()
    stale_groups: List[str] = []
    accepted_groups: List[str] = []
    accepted_tree_count = 0
    rejected_tree_count = 0
    started = time.perf_counter()
    for tree in sorted(trees, key=lambda item: item.tree_index):
        proposal = proposal_by_tree.get(tree.tree_index)
        tree_accepted = 0
        if proposal is None:
            rejected["missing_tree_proposal"] += 1
            rejected_tree_count += 1
            continue
        for group_name in tree.committable_group_names:
            group = homology.pin_groups[group_name]
            if group.assigned:
                stale_groups.append(group_name)
                rejected["stale_group_overlap"] += 1
                continue
            segment_id = proposal.get(group_name)
            if segment_id is None:
                rejected["missing_group_proposal"] += 1
                continue
            if segment_id == SKIP_SEGMENT_ID:
                rejected["skip_proposal"] += 1
                continue
            segment = segment_manager.abstract_segments.get(segment_id)
            if segment is None:
                rejected["invalid_segment"] += 1
                continue
            if segment.module_name != group.module_name:
                rejected["module_type_mismatch"] += 1
                continue
            if not segment.can_fit(group.max_pin_width):
                rejected["capacity_exceeded"] += 1
                continue
            if any(
                (pin.parent_inst, segment_id) not in segment_manager.instance_lookup
                for pin in group.pins
            ):
                rejected["missing_segment_instance"] += 1
                continue
            segment_manager.apply_assignment(group_name, group.pins, segment_id)
            homology.mark_assigned(group, segment_id)
            accepted_groups.append(group_name)
            tree_accepted += 1
        if tree_accepted:
            accepted_tree_count += 1
        else:
            rejected_tree_count += 1
    commit_seconds = time.perf_counter() - started
    assignment = {
        name: homology.pin_groups[name].assigned_segment_id
        for name in sorted(homology.pin_groups)
        if homology.pin_groups[name].assigned_segment_id is not None
    }
    assignment_text = json.dumps(assignment, sort_keys=True, separators=(",", ":"))
    assignment_hash = hashlib.sha256(assignment_text.encode("utf-8")).hexdigest()
    capacity_violations = segment_manager.capacity_violations()
    return {
        "accepted_groups": len(accepted_groups),
        "accepted_tree_count": accepted_tree_count,
        "rejected_tree_count": rejected_tree_count,
        "rejected_proposals": sum(rejected.values()),
        "rejection_reasons": dict(sorted(rejected.items())),
        "stale_group_count": len(stale_groups),
        "stale_groups_sample": stale_groups[:20],
        "unassigned_group_count": len(homology.unassigned_groups()),
        "capacity_violation_count": len(capacity_violations),
        "capacity_safe": not capacity_violations,
        "correct": not homology.unassigned_groups() and not capacity_violations,
        "assignment_hash": assignment_hash,
        "commit_seconds": commit_seconds,
        "assignment_sample": dict(list(assignment.items())[:20]),
    }


def _result_batch_timing(results: Sequence[dict]) -> List[dict]:
    """Compact actual timing/load records for each packed task."""

    return [
        {
            "batch_index": item["batch_index"],
            "tree_count": len(item["trees"]),
            "worker_seconds": item["worker_seconds"],
            "search_seconds_sum": sum(
                tree["search_seconds"] for tree in item["trees"]
            ),
        }
        for item in sorted(results, key=lambda item: item["batch_index"])
    ]


def _run_serial(
    block_json: str,
    pingroup_json: str,
    trees: Sequence[TreeDescription],
    batches: Sequence[TreeBatch],
    options: Mapping[str, object],
    random_seed: int,
) -> dict:
    _tree_worker_init(block_json, pingroup_json, options)
    tasks = _make_tasks(batches, trees, random_seed)
    started = time.perf_counter()
    results = [_tree_worker_task(task) for task in tasks]
    wall = time.perf_counter() - started
    committed = _normalized_commit(block_json, pingroup_json, trees, results)
    return {
        "wall_seconds": wall,
        "worker_seconds_sum": sum(item["worker_seconds"] for item in results),
        "search_seconds_sum": sum(
            tree["search_seconds"]
            for item in results
            for tree in item["trees"]
        ),
        "commit": committed,
        "batch_timing": _result_batch_timing(results),
        "results": results,
    }


def _run_pool(
    block_json: str,
    pingroup_json: str,
    trees: Sequence[TreeDescription],
    batches: Sequence[TreeBatch],
    options: Mapping[str, object],
    random_seed: int,
    workers: int,
    repetitions: int,
) -> dict:
    tasks = _make_tasks(batches, trees, random_seed)
    pool_started = time.perf_counter()
    pool = mp.Pool(
        processes=max(1, int(workers)),
        initializer=_tree_worker_init,
        initargs=(block_json, pingroup_json, options),
    )
    pool_create_seconds = time.perf_counter() - pool_started
    runs = []
    try:
        for repetition in range(max(1, int(repetitions))):
            started = time.perf_counter()
            results = pool.map(_tree_worker_task, tasks)
            wall = time.perf_counter() - started
            runs.append(
                {
                    "repetition": repetition,
                    "wall_seconds": wall,
                    "worker_seconds_sum": sum(item["worker_seconds"] for item in results),
                    "search_seconds_sum": sum(
                        tree["search_seconds"]
                        for item in results
                        for tree in item["trees"]
                    ),
                    "initializer_seconds_observed": sum(
                        item.get("initializer_seconds", 0.0) for item in results
                    ),
                    "results": results,
                }
            )
    finally:
        pool.close()
        pool.join()
    final = runs[-1]
    committed = _normalized_commit(block_json, pingroup_json, trees, final["results"])
    return {
        "pool_create_seconds": pool_create_seconds,
        "cold_map_seconds": runs[0]["wall_seconds"],
        "cold_total_seconds": pool_create_seconds + runs[0]["wall_seconds"],
        "initializer_seconds_observed": runs[0]["initializer_seconds_observed"],
        "steady_state_map_seconds": [run["wall_seconds"] for run in runs[1:]],
        "steady_state_map_average_seconds": (
            sum(run["wall_seconds"] for run in runs[1:]) / (len(runs) - 1)
            if len(runs) > 1 else None
        ),
        "runs": [
            {key: value for key, value in run.items() if key != "results"}
            for run in runs
        ],
        "commit": committed,
        "batch_timing_final": _result_batch_timing(final["results"]),
        "results": final["results"],
    }


def _schedule_summary(
    batches: Sequence[TreeBatch],
    graph: TreeConflictGraph,
) -> dict:
    return summarize_batches(batches, graph)


def run_tree_experiment(
    block_json: str | Path,
    pingroup_json: str | Path,
    *,
    workers: int = 2,
    simulations: int = 16,
    max_trees_per_batch: int = 8,
    max_estimated_work: float = 80.0,
    repetitions: int = 2,
    random_seed: int = 7,
    use_fanout_reuse_for_sorting: bool = False,
) -> dict:
    block_json, pingroup_json = str(block_json), str(pingroup_json)
    trees, analysis = extract_tree_descriptions(
        block_json,
        pingroup_json,
        use_fanout_reuse_for_sorting=use_fanout_reuse_for_sorting,
    )
    placedb = PlaceDB(block_json, pingroup_json)
    homology = HomologyManager(
        placedb,
        use_fanout_reuse_for_sorting=use_fanout_reuse_for_sorting,
    )
    segment_manager = SegmentManager(placedb)
    graph = build_tree_conflict_graph(trees, placedb, homology, segment_manager)
    naive = naive_consecutive_batches(
        trees,
        max_trees_per_batch=max_trees_per_batch,
        max_estimated_work=max_estimated_work,
    )
    conflict_aware = weighted_greedy_batches(
        graph,
        max_trees_per_batch=max_trees_per_batch,
        max_estimated_work=max_estimated_work,
    )
    options = {
        "simulations": simulations,
        "search_mode": "basic",
        "basic_dynamic_simulations": False,
        "basic_min_simulations": max(1, simulations),
        "basic_depth1_simulations": max(1, simulations),
        "basic_depth2_simulations": max(1, simulations),
        "enable_depth1_greedy": False,
        "enable_candidate_pruning": True,
        "candidate_min_count": 16,
        "candidate_top_k": 12,
        "candidate_tail_top_k": 8,
        "candidate_score_tolerance": 0.03,
        "wirelength_weight": 0.9,
        "reward_normalization_floor": 1.0,
        "reward_scale": 100.0,
    }
    # Serial execution is the reference tree sequence.  It uses the same
    # conflict-aware schedule's stable tree order only for task construction;
    # commit is always tree-index order.
    serial = _run_serial(
        block_json,
        pingroup_json,
        trees,
        [TreeBatch(index, (tree.tree_index,), tree.estimated_work) for index, tree in enumerate(trees)],
        options,
        random_seed,
    )
    naive_run = _run_pool(
        block_json, pingroup_json, trees, naive, options, random_seed, workers, repetitions
    )
    aware_run = _run_pool(
        block_json,
        pingroup_json,
        trees,
        conflict_aware,
        options,
        random_seed,
        workers,
        repetitions,
    )

    def correctness(run: dict) -> dict:
        return {
            "assignment_hash_equal_to_serial": run["commit"]["assignment_hash"]
            == serial["commit"]["assignment_hash"],
            "accepted_group_count_equal_to_serial": run["commit"]["accepted_groups"]
            == serial["commit"]["accepted_groups"],
            "no_fallback_used": "fallback" not in run["commit"]["rejection_reasons"],
        }

    tree_depths = [tree.depth for tree in trees]
    serial_wall = serial["wall_seconds"]
    naive_steady = naive_run.get("steady_state_map_average_seconds")
    aware_steady = aware_run.get("steady_state_map_average_seconds")
    return {
        "input": {"block": block_json, "pingroup": pingroup_json},
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "parameters": {
            "workers": workers,
            "simulations": simulations,
            "max_trees_per_batch": max_trees_per_batch,
            "max_estimated_work": max_estimated_work,
            "repetitions": repetitions,
            "random_seed": random_seed,
            "use_fanout_reuse_for_sorting": use_fanout_reuse_for_sorting,
        },
        "tree_summary": {
            **analysis["summary"],
            "tree_count": len(trees),
            "depth_histogram": analysis["depth_histogram"],
            "tree_index_is_stable": [tree.tree_index for tree in trees]
            == list(range(len(trees))),
            "min_depth": min(tree_depths, default=0),
            "max_depth": max(tree_depths, default=0),
            "average_depth": sum(tree_depths) / len(tree_depths) if tree_depths else 0.0,
        },
        "tree_conflicts": {
            "tree_count": len(trees),
            "edge_count": len(graph.edge_reasons),
            "reason_edge_counts": graph.reason_counts(),
            "hard_reason_types": sorted(HARD_CONFLICTS),
            "soft_reason_types": sorted(set(CONFLICT_WEIGHTS) - HARD_CONFLICTS),
        },
        "schedules": {
            "naive_consecutive": _schedule_summary(naive, graph),
            "conflict_aware_weighted_greedy": _schedule_summary(conflict_aware, graph),
        },
        "serial": {
            key: value for key, value in serial.items() if key not in {"results"}
        },
        "naive_consecutive": {
            key: value for key, value in naive_run.items() if key not in {"results"}
        },
        "conflict_aware": {
            key: value for key, value in aware_run.items() if key not in {"results"}
        },
        "correctness": {
            "naive": correctness(naive_run),
            "conflict_aware": correctness(aware_run),
            "serial_assignment_hash": serial["commit"]["assignment_hash"],
            "naive_assignment_hash": naive_run["commit"]["assignment_hash"],
            "conflict_aware_assignment_hash": aware_run["commit"]["assignment_hash"],
        },
        "performance": {
            "serial_wall_seconds": serial_wall,
            "naive_cold_total_seconds": naive_run["cold_total_seconds"],
            "conflict_aware_cold_total_seconds": aware_run["cold_total_seconds"],
            "naive_steady_speedup_vs_serial": (
                serial_wall / naive_steady if naive_steady else None
            ),
            "conflict_aware_steady_speedup_vs_serial": (
                serial_wall / aware_steady if aware_steady else None
            ),
            "naive_batch_duration_max_min_ratio": (
                max((item["worker_seconds"] for item in naive_run["batch_timing_final"]), default=0.0)
                / max(1e-12, min((item["worker_seconds"] for item in naive_run["batch_timing_final"]), default=0.0))
            ),
            "conflict_aware_batch_duration_max_min_ratio": (
                max((item["worker_seconds"] for item in aware_run["batch_timing_final"]), default=0.0)
                / max(1e-12, min((item["worker_seconds"] for item in aware_run["batch_timing_final"]), default=0.0))
            ),
        },
        "interpretation": {
            "tree_unit": "Existing seed-local MCTS tree; workers never share a tree.",
            "duplicate_policy": "Fixed tree-index commit order; later duplicate groups are stale and rejected.",
            "hard_conflict_policy": "Hard conflicts are scored heavily but not merged into global components; invalid/stale submissions are counted and rejected.",
            "soft_conflict_policy": "Net topology and module type add weighted batching cost and may co-reside.",
            "serial_reference_scope": (
                "Serial search generates every tree proposal from the same initial shadow state "
                "and commits afterward; it is not the online per-tree commit behavior of AssignmentSolver."
            ),
            "conflict_metric_scope": (
                "Conflict cost covers trees merged into the same worker task. All tasks are still "
                "dispatched in one pool map, so cross-task state synchronization and conflict recovery "
                "remain future work; this experiment does not claim fewer rejected proposals."
            ),
            "not_default_behavior": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True)
    parser.add_argument("--pingroup", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--max-trees-per-batch", type=int, default=8)
    parser.add_argument("--max-estimated-work", type=float, default=80.0)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_tree_experiment(
        args.block,
        args.pingroup,
        workers=args.workers,
        simulations=args.simulations,
        max_trees_per_batch=args.max_trees_per_batch,
        max_estimated_work=args.max_estimated_work,
        repetitions=args.repetitions,
        random_seed=args.seed,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote tree-level Stage1 experiment: {args.output}")
    print(json.dumps({key: report[key] for key in ("tree_summary", "tree_conflicts", "schedules", "serial", "naive_consecutive", "conflict_aware", "correctness")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
