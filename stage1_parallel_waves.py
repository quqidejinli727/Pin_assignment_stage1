"""Real seed-local MCTS wave experiment with explicit worker state sync.

Unlike the earlier offline reports, this adapter owns a real
``AssignmentSolver`` in the parent process and commits accepted proposals to
its real ``HomologyManager``/``SegmentManager``.  A resident process per worker
receives an explicit state snapshot before every wave; workers never share a
tree or mutate the parent.  The default solver path is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import platform
import queue
import socket
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from assignment_solver import AssignmentSolver
from scoring import FeedthroughContext, net_hpwl
from stage1_tree_parallel_experiment import (
    HARD_CONFLICTS,
    CONFLICT_WEIGHTS,
    TreeDescription,
    build_tree_conflict_graph,
    extract_tree_descriptions,
    conflict_cost,
    _edge_key,
)
from PlaceDB import PlaceDB
from homology import HomologyManager
from segment import SegmentManager
from mcts import MCTSSolver, SKIP_SEGMENT_ID
from config import DEFAULT_CONFIG


@dataclass(frozen=True)
class Wave:
    wave_index: int
    tree_indices: Tuple[int, ...]
    estimated_work: float
    hard_conflicts: int
    soft_conflicts: int
    conflict_cost: float


def _tree_seed(seed: int, tree_index: int) -> int:
    return int(seed) + int(tree_index)


def _tree_conflicts(indices: Sequence[int], graph) -> Tuple[int, int, float]:
    hard = soft = 0
    cost = 0.0
    for position, left in enumerate(indices):
        for right in indices[position + 1 :]:
            reasons = graph.edge_reasons.get(_edge_key(left, right), ())
            if not reasons:
                continue
            cost += conflict_cost(reasons)
            if any(reason in HARD_CONFLICTS for reason in reasons):
                hard += 1
            else:
                soft += 1
    return hard, soft, cost


def schedule_waves(
    trees: Sequence[TreeDescription],
    graph,
    *,
    workers: int,
    trees_per_worker: int = 4,
    max_estimated_work: float = 80.0,
) -> List[Wave]:
    """Select waves by minimizing conflicts against all already-selected trees."""

    max_trees = max(1, int(workers)) * max(1, int(trees_per_worker))
    remaining = {tree.tree_index for tree in trees}
    by_index = {tree.tree_index: tree for tree in trees}
    waves: List[Wave] = []
    while remaining:
        selected: List[int] = []
        work = 0.0
        while len(selected) < max_trees:
            candidates = [by_index[index] for index in sorted(remaining)]
            candidates = [
                tree
                for tree in candidates
                if not selected
                or work + tree.estimated_work <= max_estimated_work
            ]
            if not candidates:
                break
            scored = []
            for tree in candidates:
                hard, soft, cost = _tree_conflicts(selected + [tree.tree_index], graph)
                # Prefer zero-hard candidates, then least weighted conflict,
                # then stable index.  Work is a final balancing tie-breaker.
                scored.append(
                    (
                        hard,
                        cost,
                        soft,
                        work + tree.estimated_work,
                        tree.tree_index,
                        tree,
                    )
                )
            _, _, _, projected, _, chosen = min(scored, key=lambda item: item[:-1])
            selected.append(chosen.tree_index)
            remaining.remove(chosen.tree_index)
            work = projected
        if not selected:
            chosen = by_index[min(remaining)]
            selected = [chosen.tree_index]
            remaining.remove(chosen.tree_index)
            work = chosen.estimated_work
        hard, soft, cost = _tree_conflicts(selected, graph)
        waves.append(Wave(len(waves), tuple(sorted(selected)), work, hard, soft, cost))
    return waves


def _estimate_tree_ft_work(tree: TreeDescription, placedb: PlaceDB | None) -> float:
    """Estimate predictor work for deterministic wave task routing."""

    if placedb is None:
        return float(max(1.0, tree.estimated_work))
    cost = 0.0
    for net_id in tree.related_net_ids:
        if 0 <= net_id < len(placedb.nets_list):
            pin_count = len(getattr(placedb.nets_list[net_id], "pins", ()))
            # Predictor geometry work grows super-linearly for very large
            # nets.  The quadratic term is only a routing estimate; it does
            # not alter MCTS inputs or scores.
            pins = max(1, pin_count)
            cost += 1.0 + 0.25 * pins + 0.01 * pins * pins
        else:
            cost += 1.0
    return max(1.0, cost)


def _partition_wave(
    wave: Wave,
    trees: Mapping[int, TreeDescription],
    workers: int,
    trees_per_worker: int,
    placedb: PlaceDB | None = None,
) -> List[Tuple[int, ...]]:
    """Deterministically pack a wave by estimated FT work with light affinity.

    ``TreeDescription.estimated_work`` is depth-oriented and is useful for
    wave formation, but it is a weak predictor of FT reward cost.  When a
    PlaceDB is available, use net/pin volume as the dispatch cost.  A small
    affinity bonus keeps overlapping net sets on the same resident worker
    when the projected loads are close, preserving local FT cache reuse.  It
    only changes task routing: tree seeds, wave membership, and commit order
    remain stable.
    """

    packs: List[List[int]] = [[] for _ in range(max(1, workers))]
    loads = [0.0] * len(packs)
    net_sets: List[set[int]] = [set() for _ in packs]
    costs = {
        index: _estimate_tree_ft_work(trees[index], placedb)
        for index in wave.tree_indices
    }
    for index in sorted(wave.tree_indices, key=lambda item: (-costs[item], item)):
        choices = [
            worker
            for worker, pack in enumerate(packs)
            if len(pack) < max(1, trees_per_worker)
        ]
        if not choices:
            # This should be unreachable because the scheduler caps the wave.
            choices = list(range(len(packs)))
        tree_nets = set(trees[index].related_net_ids)
        # Keep affinity deliberately small: it cannot outweigh a material
        # load imbalance, but wins deterministic near-ties.
        def score(worker: int) -> tuple[float, float, int, int]:
            overlap = len(tree_nets & net_sets[worker])
            projected = loads[worker] + costs[index]
            affinity_adjusted = projected - 0.25 * overlap
            return (affinity_adjusted, projected, -overlap, worker)

        worker = min(choices, key=score)
        packs[worker].append(index)
        loads[worker] += costs[index]
        net_sets[worker].update(tree_nets)
    return [tuple(pack) for pack in packs]


def _pin_state(solver: AssignmentSolver) -> dict:
    pins = {}
    for pin in solver.placedb.pin_dict.values():
        pins[pin.full_name] = {
            "x": pin.x,
            "y": pin.y,
            "assigned_segment_id": pin.assigned_segment_id,
            "assigned_segment_coord": pin.assigned_segment_coord,
            "segment_endpoints": pin.segment_endpoints,
            "scope": pin.scope,
        }
    return {
        "assigned_groups": {
            name: group.assigned_segment_id
            for name, group in solver.homology.pin_groups.items()
            if group.assigned and group.assigned_segment_id is not None
        },
        "segment_used_width": {
            segment_id: segment.used_width
            for segment_id, segment in solver.segment_manager.abstract_segments.items()
        },
        "pins": pins,
    }


def _pin_state_delta(
    solver: AssignmentSolver,
    group_names: Sequence[str],
    *,
    from_version: int,
    to_version: int,
) -> dict:
    """Serialize only monotonic assignments added since a worker snapshot.

    Dynamic-queue assignments never roll back.  Sending the complete pin state
    before every small Prime tree would otherwise turn 11,489 trees and 33,000
    pins into hundreds of millions of redundant pin records.
    """

    names = tuple(dict.fromkeys(str(name) for name in group_names))
    assigned = {
        name: solver.homology.pin_groups[name].assigned_segment_id
        for name in names
        if solver.homology.pin_groups[name].assigned
        and solver.homology.pin_groups[name].assigned_segment_id is not None
    }
    segment_ids = tuple(dict.fromkeys(assigned.values()))
    pins = {}
    for name in assigned:
        for pin in solver.homology.pin_groups[name].pins:
            pins[pin.full_name] = {
                "x": pin.x,
                "y": pin.y,
                "assigned_segment_id": pin.assigned_segment_id,
                "assigned_segment_coord": pin.assigned_segment_coord,
                "segment_endpoints": pin.segment_endpoints,
                "scope": pin.scope,
            }
    return {
        "mode": "delta",
        "from_version": int(from_version),
        "to_version": int(to_version),
        "assigned_groups": assigned,
        "segment_used_width": {
            segment_id: solver.segment_manager.abstract_segments[segment_id].used_width
            for segment_id in segment_ids
        },
        "segment_assigned_groups": {
            segment_id: list(
                solver.segment_manager.abstract_segments[segment_id].assigned_groups
            )
            for segment_id in segment_ids
        },
        "pins": pins,
    }


def _apply_pin_state(
    homology: HomologyManager,
    segment_manager: SegmentManager,
    placedb: PlaceDB,
    state: dict,
) -> None:
    if state.get("mode") == "delta":
        assigned = state["assigned_groups"]
        for name, segment_id in assigned.items():
            group = homology.pin_groups[name]
            group.assigned = True
            group.assigned_segment_id = segment_id
        for segment_id, used_width in state["segment_used_width"].items():
            segment = segment_manager.abstract_segments[segment_id]
            segment.used_width = float(used_width)
            segment.assigned_groups = list(
                state["segment_assigned_groups"].get(segment_id, ())
            )
        for pin_name, item in state["pins"].items():
            pin = placedb.pin_dict[pin_name]
            pin.x = item.get("x", 0.0)
            pin.y = item.get("y", 0.0)
            pin.assigned_segment_id = item.get("assigned_segment_id")
            pin.assigned_segment_coord = item.get("assigned_segment_coord")
            pin.segment_endpoints = item.get("segment_endpoints")
            if "scope" in item:
                pin.scope = item["scope"]
        return

    assigned = state["assigned_groups"]
    for group in homology.pin_groups.values():
        group.assigned = group.name in assigned
        group.assigned_segment_id = assigned.get(group.name)
    for segment_id, segment in segment_manager.abstract_segments.items():
        segment.used_width = float(state["segment_used_width"].get(segment_id, 0.0))
        segment.assigned_groups = [
            name for name, assigned_segment in assigned.items() if assigned_segment == segment_id
        ]
    for pin in placedb.pin_dict.values():
        item = state["pins"].get(pin.full_name, {})
        pin.x = item.get("x", 0.0)
        pin.y = item.get("y", 0.0)
        pin.assigned_segment_id = item.get("assigned_segment_id")
        pin.assigned_segment_coord = item.get("assigned_segment_coord")
        pin.segment_endpoints = item.get("segment_endpoints")
        if "scope" in item:
            pin.scope = item["scope"]


def _make_worker_feedthrough_context(placedb: PlaceDB, solver_options: Mapping[str, object]):
    """Create one FT session inside this worker process, never in the parent."""

    enabled = bool(solver_options.get("enable_feedthrough", False))
    weight = float(solver_options.get("feedthrough_weight", 0.0))
    if not enabled or weight == 0.0:
        return None
    source = solver_options.get("feedthrough_evaluate_source_dir") or solver_options.get(
        "feedthrough_source_dir"
    )
    if not source:
        raise ValueError("feedthrough_evaluate_source_dir is required for worker FT reward")
    role = str(solver_options.get("feedthrough_reward_source", "evaluate")).lower()
    return FeedthroughContext(
        placedb,
        Path(source),
        auto_build_feedthrough=bool(solver_options.get("auto_build_feedthrough", False)),
        cmake_generator=solver_options.get("cmake_generator"),
        role=role,
        trace_enabled=bool(solver_options.get("trace_feedthrough", False)),
    )


def _wave_worker_main(
    worker_index: int,
    block_json: str,
    pingroup_json: str,
    solver_options: Mapping[str, object],
    command_queue,
    result_queue,
) -> None:
    """Resident worker command loop with explicit sync acknowledgements."""

    started = time.perf_counter()
    placedb = PlaceDB(block_json, pingroup_json)
    homology = HomologyManager(
        placedb,
        use_fanout_reuse_for_sorting=bool(solver_options["homology_use_fanout_reuse_for_sorting"]),
    )
    segment_manager = SegmentManager(
        placedb,
        max_segment_length=solver_options["max_segment_length"],
    )
    mcts_options = dict(solver_options["mcts_options"])
    feedthrough_context = _make_worker_feedthrough_context(placedb, solver_options)
    _wave_worker_loop(
        worker_index,
        solver_options,
        placedb,
        homology,
        segment_manager,
        mcts_options,
        feedthrough_context,
        command_queue,
        result_queue,
        started,
    )


def _wave_worker_loop(
    worker_index: int,
    solver_options: Mapping[str, object],
    placedb: PlaceDB,
    homology: HomologyManager,
    segment_manager: SegmentManager,
    mcts_options: Mapping[str, object],
    feedthrough_context,
    command_queue,
    result_queue,
    worker_started: float,
) -> None:
    """Run the resident command loop and close the process-local FT session."""

    parallel_trace_reports = []
    resident_shutdown_seconds = 0.0
    try:
        result_queue.put(
            {
                "kind": "ready",
                "worker": worker_index,
                "init_seconds": time.perf_counter() - worker_started,
                "feedthrough_context_created": feedthrough_context is not None,
                "feedthrough_trace": (
                    feedthrough_context.trace_summary()
                    if feedthrough_context is not None
                    and bool(getattr(feedthrough_context, "trace_enabled", False))
                    else None
                ),
            }
        )
        _wave_worker_commands(
            worker_index,
            solver_options,
            placedb,
            homology,
            segment_manager,
            mcts_options,
            feedthrough_context,
            command_queue,
            result_queue,
        )
    finally:
        shutdown_started = time.perf_counter()
        if feedthrough_context is not None:
            feedthrough_context.close()
        shutdown_seconds = time.perf_counter() - shutdown_started
        result_queue.put(
            {
                "kind": "stopped",
                "worker": worker_index,
                "feedthrough_context_closed": True,
                "shutdown_seconds": shutdown_seconds,
                "feedthrough_trace": (
                    feedthrough_context.trace_summary()
                    if feedthrough_context is not None
                    and bool(getattr(feedthrough_context, "trace_enabled", False))
                    else None
                ),
            }
        )


def _wave_worker_commands(
    worker_index: int,
    solver_options: Mapping[str, object],
    placedb: PlaceDB,
    homology: HomologyManager,
    segment_manager: SegmentManager,
    mcts_options: Mapping[str, object],
    feedthrough_context,
    command_queue,
    result_queue,
) -> None:
    while True:
        command, payload = command_queue.get()
        if command == "stop":
            result_queue.put({"kind": "stop_requested", "worker": worker_index})
            return
        if command == "sync":
            _apply_pin_state(homology, segment_manager, placedb, payload)
            result_queue.put({"kind": "synced", "worker": worker_index})
            continue
        if command != "search":
            result_queue.put({"kind": "error", "worker": worker_index, "error": f"unknown command {command}"})
            continue
        trees_payload = payload
        started = time.perf_counter()
        tree_results = []
        for tree in trees_payload:
            group_names = tuple(tree["group_names"])
            groups = [homology.pin_groups[name] for name in group_names]
            nets = [placedb.nets_list[net_id] for net_id in tree["net_ids"]]
            deferred_names = tuple(
                tree.get("deferred_group_names", tree.get("skipped_group_names", ()))
            )
            search_started = time.perf_counter()
            mcts = MCTSSolver(
                placedb=placedb,
                segment_manager=segment_manager,
                groups=groups,
                nets=nets,
                simulations=int(solver_options["simulations"]),
                random_seed=_tree_seed(int(solver_options["random_seed"]), tree["tree_index"]),
                deferred_group_names=deferred_names,
                excluded_pin_names=set(),
                feedthrough_context=feedthrough_context,
                **mcts_options,
            )
            proposal = mcts.search()
            reward_profile = getattr(getattr(mcts, "reward_evaluator", None), "timing_profile", {})
            tree_results.append(
                {
                    "tree_index": tree["tree_index"],
                    "proposal": dict(proposal),
                    "search_seconds": time.perf_counter() - search_started,
                    "simulations": getattr(mcts, "last_simulation_count", 0),
                    "reward_feedthrough_seconds": reward_profile.get("reward_feedthrough", 0.0),
                    "reward_feedthrough_eval_seconds": reward_profile.get(
                        "reward_feedthrough_eval", 0.0
                    ),
                    "reward_feedthrough_location_seconds": reward_profile.get(
                        "reward_feedthrough_location", 0.0
                    ),
                }
            )
        result_queue.put(
            {
                "kind": "searched",
                "worker": worker_index,
                "trees": tree_results,
                "worker_seconds": time.perf_counter() - started,
                "feedthrough_cache_hits": getattr(feedthrough_context, "cache_hits", 0),
                "feedthrough_cache_misses": getattr(feedthrough_context, "cache_misses", 0),
                "feedthrough_batch_calls": getattr(feedthrough_context, "batch_calls", 0),
                "feedthrough_batch_requests": getattr(
                    feedthrough_context, "batch_request_count", 0
                ),
                "feedthrough_batch_misses": getattr(
                    feedthrough_context, "batch_miss_count", 0
                ),
            }
        )


class ResidentWaveWorkers:
    """Explicit per-worker Queue/Pipe resident process manager."""

    def __init__(self, block_json: str, pingroup_json: str, options: Mapping[str, object], worker_count: int):
        self.queues = []
        self.result_queue = mp.Queue()
        self.processes = []
        self.ready_reports = []
        self.stopped_reports = []
        started = time.perf_counter()
        for worker_index in range(max(1, worker_count)):
            command_queue = mp.Queue()
            process = mp.Process(
                target=_wave_worker_main,
                args=(worker_index, block_json, pingroup_json, options, command_queue, self.result_queue),
            )
            process.start()
            self.queues.append(command_queue)
            self.processes.append(process)
        for _ in self.processes:
            self.ready_reports.append(self._get_kind("ready"))
        self.cold_seconds = time.perf_counter() - started

    def _get_kind(self, kind: str) -> dict:
        while True:
            result = self.result_queue.get(timeout=120)
            if result.get("kind") == kind:
                return result
            if result.get("kind") == "error":
                raise RuntimeError(result["error"])

    def sync(self, state: dict) -> float:
        started = time.perf_counter()
        for command_queue in self.queues:
            command_queue.put(("sync", state))
        for _ in self.queues:
            self._get_kind("synced")
        return time.perf_counter() - started

    def search(self, packs: Sequence[Sequence[dict]]) -> Tuple[List[dict], float]:
        started = time.perf_counter()
        for worker_index, command_queue in enumerate(self.queues):
            command_queue.put(("search", list(packs[worker_index])))
        results = [self._get_kind("searched") for _ in self.queues]
        return results, time.perf_counter() - started

    def close(self) -> None:
        for command_queue in self.queues:
            command_queue.put(("stop", None))
        for _ in self.processes:
            self.stopped_reports.append(self._get_kind("stopped"))
        for process in self.processes:
            process.join(timeout=30)


def _solver_kwargs(block_json: str, pingroup_json: str, simulations: int, seed: int) -> dict:
    """Match ``main.py --simulations N --no-feedthrough`` configuration.

    Config dataclass names intentionally mirror AssignmentSolver arguments;
    reflection here prevents the experiment from silently using constructor
    defaults where ``main.py`` supplies a different config default.
    """

    values = {}
    for name in inspect.signature(AssignmentSolver.__init__).parameters:
        if name in {"self", "block_json_path", "pingroup_json_path"}:
            continue
        if hasattr(DEFAULT_CONFIG, name):
            values[name] = getattr(DEFAULT_CONFIG, name)
    budget = max(1, int(simulations))
    values.update(
        {
            "block_json_path": block_json,
            "pingroup_json_path": pingroup_json,
            "simulations": simulations,
            "random_seed": seed,
            "enable_feedthrough": False,
            "feedthrough_weight": DEFAULT_CONFIG.feedthrough_weight,
            # Keep the budget profile derived from the requested simulations so
            # small probes do not silently run a larger MCTS budget.
            "mcts_basic_dynamic_simulations": False,
            "mcts_basic_min_simulations": budget,
            "mcts_basic_depth1_simulations": budget,
            "mcts_basic_depth2_simulations": budget,
            "mcts_hybrid_min_layer_simulations": budget,
            "mcts_hybrid_max_layer_simulations": budget * 8,
            "mcts_hybrid_max_tree_simulations": budget * 16,
        }
    )
    return values


def _make_worker_options(
    solver: AssignmentSolver,
    *,
    trace_feedthrough: bool = False,
) -> dict:
    return {
        "simulations": solver.simulations,
        "random_seed": solver.random_seed,
        "max_segment_length": solver.max_segment_length,
        "homology_use_fanout_reuse_for_sorting": solver.homology.use_fanout_reuse_for_sorting,
        "mcts_options": dict(solver.mcts_options),
        "enable_feedthrough": solver.enable_feedthrough,
        "feedthrough_weight": solver.feedthrough_weight,
        "feedthrough_source_dir": str(solver.feedthrough_source_dir)
        if solver.feedthrough_source_dir is not None
        else None,
        "feedthrough_predict_source_dir": str(solver.feedthrough_predict_source_dir)
        if solver.feedthrough_predict_source_dir is not None
        else None,
        "feedthrough_evaluate_source_dir": str(solver.feedthrough_evaluate_source_dir)
        if solver.feedthrough_evaluate_source_dir is not None
        else None,
        "feedthrough_reward_source": solver.feedthrough_reward_source,
        "auto_build_feedthrough": solver.auto_build_feedthrough,
        "cmake_generator": solver.cmake_generator,
        "trace_feedthrough": bool(trace_feedthrough),
    }


def _tree_payload(tree: TreeDescription, solver: AssignmentSolver) -> dict | None:
    active_search = tuple(
        name for name in tree.search_group_names
        if not solver.homology.pin_groups[name].assigned
    )
    active_committable = tuple(
        name for name in tree.committable_group_names
        if not solver.homology.pin_groups[name].assigned
    )
    if not active_search:
        return None
    return {
        "tree_index": tree.tree_index,
        "group_names": active_search,
        "committable_group_names": active_committable,
        "net_ids": tree.related_net_ids,
        "deferred_group_names": tuple(
            name for name in tree.deferred_group_names
            if not solver.homology.pin_groups[name].assigned
        ),
    }


def _hashes(solver: AssignmentSolver) -> dict:
    assignment = {
        name: group.assigned_segment_id
        for name, group in sorted(solver.homology.pin_groups.items())
        if group.assigned and group.assigned_segment_id is not None
    }
    assignment_text = json.dumps(assignment, sort_keys=True, separators=(",", ":"))
    segment_text = json.dumps(
        solver.segment_manager.to_output_dict(), sort_keys=True, separators=(",", ":")
    )
    return {
        "assignment_hash": hashlib.sha256(assignment_text.encode()).hexdigest(),
        "segment_output_hash": hashlib.sha256(segment_text.encode()).hexdigest(),
    }


def _hpwl(solver: AssignmentSolver) -> float:
    return sum(net_hpwl(net, solver.placedb, {}) for net in solver.placedb.nets_list)


def _trace_concurrency(trace_reports: Sequence[Mapping[str, object]]) -> dict:
    """Summarize overlap of worker-local FT batch intervals in memory."""

    by_worker = []
    events = []
    for report in trace_reports:
        intervals = sorted(
            (float(item[0]), float(item[1]))
            for item in (report.get("intervals") or [])
            if len(item) >= 2 and float(item[1]) >= float(item[0])
        )
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        by_worker.append(merged)
        for start, end in merged:
            events.append((start, 1))
            events.append((end, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    max_concurrent = 0
    overlap_seconds = 0.0
    prior = None
    for timestamp, delta in events:
        if prior is not None and active >= 2:
            overlap_seconds += max(0.0, timestamp - prior)
        active += delta
        max_concurrent = max(max_concurrent, active)
        prior = timestamp
    total_active = sum(end - start for merged in by_worker for start, end in merged)
    return {
        "worker_count": len(by_worker),
        "max_concurrent_evaluators": max_concurrent,
        "overlap_seconds": overlap_seconds,
        "total_merged_active_seconds": total_active,
        "overlap_ratio": overlap_seconds / total_active if total_active else 0.0,
        "worker_intervals": [
            {"interval_count": len(merged), "first": merged[0][0] if merged else None,
             "last": merged[-1][1] if merged else None}
            for merged in by_worker
        ],
    }


def _final_feedthrough_summary(solver: AssignmentSolver) -> dict:
    """Evaluate final FT metrics in the parent without sharing worker sessions."""

    if not solver.enable_feedthrough or solver.feedthrough_weight == 0.0:
        return {"enabled": False, "total": 0.0, "average": 0.0, "seconds": 0.0}
    source = solver.feedthrough_evaluate_source_dir or solver.feedthrough_source_dir
    if source is None:
        raise ValueError("feedthrough_evaluate_source_dir is required for final metrics")
    started = time.perf_counter()
    metrics = solver.final_net_metrics(
        feedthrough_source_dir=source,
        enable_feedthrough=True,
        auto_build_feedthrough=solver.auto_build_feedthrough,
        cmake_generator=solver.cmake_generator,
    )
    total = sum(float(metric.feedthrough) for metric in metrics)
    metric_count = sum(1 for metric in metrics if metric.pin_count > 1)
    return {
        "enabled": True,
        "total": total,
        "average": total / metric_count if metric_count else 0.0,
        "metric_count": metric_count,
        "seconds": time.perf_counter() - started,
    }


def _strict_commit_wave_tree(solver: AssignmentSolver, tree: TreeDescription, proposal: dict) -> Counter:
    reasons: Counter = Counter()
    for name in tree.committable_group_names:
        group = solver.homology.pin_groups[name]
        if group.assigned:
            reasons["stale_group_overlap"] += 1
            continue
        segment_id = proposal.get(name)
        if segment_id is None:
            reasons["missing_group_proposal"] += 1
            continue
        if segment_id == SKIP_SEGMENT_ID:
            reasons["skip_proposal"] += 1
            continue
        invalid = solver._validate_segment_assignment(group, segment_id)
        if invalid is not None:
            reasons[invalid] += 1
            continue
        solver._commit_group_assignment(group, segment_id)
    return reasons


def run_parallel_waves(
    block_json: str | Path,
    pingroup_json: str | Path,
    *,
    workers: int = 2,
    simulations: int = 32,
    trees_per_worker: int = 4,
    max_estimated_work: float = 80.0,
    random_seed: int = 7,
    final_greedy_completion: bool = True,
    trace_feedthrough: bool = False,
) -> dict:
    block_json, pingroup_json = str(block_json), str(pingroup_json)
    kwargs = _solver_kwargs(block_json, pingroup_json, simulations, random_seed)
    baseline_solver = AssignmentSolver(**kwargs)
    baseline_solver.trace_feedthrough_enabled = bool(trace_feedthrough)
    baseline_context = None
    baseline_started = time.perf_counter()
    baseline_output = baseline_solver.solve()
    baseline_feedthrough = _final_feedthrough_summary(baseline_solver)
    baseline_context = baseline_solver.feedthrough_context
    baseline_trace = None
    baseline_wall = time.perf_counter() - baseline_started
    baseline_hashes = _hashes(baseline_solver)
    baseline_shutdown_started = time.perf_counter()
    baseline_solver.close_feedthrough_context()
    baseline_shutdown_seconds = time.perf_counter() - baseline_shutdown_started
    if baseline_context is not None and bool(
        getattr(baseline_context, "trace_enabled", False)
    ):
        baseline_trace = baseline_context.trace_summary()

    solver = AssignmentSolver(**kwargs)
    trees, tree_analysis = extract_tree_descriptions(
        block_json,
        pingroup_json,
        coverage_threshold=solver.homology_group_commit_coverage_threshold,
        skip_uncovered_groups=solver.homology_skip_uncovered_groups,
        skip_coverage_threshold=solver.homology_skip_coverage_threshold,
        use_fanout_reuse_for_sorting=solver.homology.use_fanout_reuse_for_sorting,
    )
    graph = build_tree_conflict_graph(trees, solver.placedb, solver.homology, solver.segment_manager)
    waves = schedule_waves(
        trees,
        graph,
        workers=workers,
        trees_per_worker=trees_per_worker,
        max_estimated_work=max_estimated_work,
    )
    trees_by_index = {tree.tree_index: tree for tree in trees}
    worker_options = _make_worker_options(
        solver,
        trace_feedthrough=trace_feedthrough,
    )
    parallel_started = time.perf_counter()
    resident = ResidentWaveWorkers(block_json, pingroup_json, worker_options, workers)
    wave_records = []
    total_search = 0.0
    total_worker_active_wall = 0.0
    total_worker_idle = 0.0
    total_search_barrier = 0.0
    total_sync = 0.0
    total_commit = 0.0
    rejection_counts: Counter = Counter()
    total_proposals = 0
    total_parallel_simulations = 0
    parallel_tree_simulations: Dict[int, int] = {}
    worker_feedthrough_counters: Dict[int, Tuple[int, int]] = {}
    worker_feedthrough_batches: Dict[int, Tuple[int, int, int]] = {}
    total_reward_feedthrough_seconds = 0.0
    total_reward_feedthrough_eval_seconds = 0.0
    total_reward_feedthrough_location_seconds = 0.0
    try:
        for wave in waves:
            active_payloads = []
            for index in wave.tree_indices:
                payload = _tree_payload(trees_by_index[index], solver)
                if payload is not None:
                    active_payloads.append(payload)
            state = _pin_state(solver)
            sync_seconds = resident.sync(state)
            packs_indices = _partition_wave(
                wave,
                trees_by_index,
                workers,
                trees_per_worker,
                placedb=solver.placedb,
            )
            packs = [
                [
                    payload
                    for payload in active_payloads
                    if payload["tree_index"] in pack
                ]
                for pack in packs_indices
            ]
            results, search_seconds = resident.search(packs)
            commit_started = time.perf_counter()
            tree_proposals = [
                tree_result
                for result in results
                for tree_result in result["trees"]
            ]
            for result in results:
                worker_feedthrough_counters[result["worker"]] = (
                    int(result.get("feedthrough_cache_hits", 0)),
                    int(result.get("feedthrough_cache_misses", 0)),
                )
                worker_feedthrough_batches[result["worker"]] = (
                    int(result.get("feedthrough_batch_calls", 0)),
                    int(result.get("feedthrough_batch_requests", 0)),
                    int(result.get("feedthrough_batch_misses", 0)),
                )
            for tree_result in sorted(tree_proposals, key=lambda item: item["tree_index"]):
                parallel_tree_simulations[tree_result["tree_index"]] = int(
                    tree_result.get("simulations", 0)
                )
                total_parallel_simulations += int(tree_result.get("simulations", 0))
                total_reward_feedthrough_seconds += float(
                    tree_result.get("reward_feedthrough_seconds", 0.0)
                )
                total_reward_feedthrough_eval_seconds += float(
                    tree_result.get("reward_feedthrough_eval_seconds", 0.0)
                )
                total_reward_feedthrough_location_seconds += float(
                    tree_result.get("reward_feedthrough_location_seconds", 0.0)
                )
                rejection_counts.update(
                    _strict_commit_wave_tree(
                        solver,
                        trees_by_index[tree_result["tree_index"]],
                        tree_result["proposal"],
                    )
                )
            commit_seconds = time.perf_counter() - commit_started
            total_proposals += len(tree_proposals)
            total_search += search_seconds
            worker_active = [float(result.get("worker_seconds", 0.0)) for result in results]
            active_critical = max(worker_active, default=0.0)
            barrier_seconds = max(0.0, search_seconds - active_critical)
            idle_seconds = sum(max(0.0, active_critical - value) for value in worker_active)
            total_worker_active_wall += active_critical
            total_worker_idle += idle_seconds
            total_search_barrier += barrier_seconds
            total_sync += sync_seconds
            total_commit += commit_seconds
            wave_records.append(
                {
                    "wave_index": wave.wave_index,
                    "tree_indices": list(wave.tree_indices),
                    "active_tree_count": len(active_payloads),
                    "worker_packs": [list(pack) for pack in packs_indices],
                    "estimated_work": wave.estimated_work,
                    "hard_conflict_count": wave.hard_conflicts,
                    "soft_conflict_count": wave.soft_conflicts,
                    "conflict_cost": wave.conflict_cost,
                    "sync_seconds": sync_seconds,
                    "search_seconds": search_seconds,
                    "commit_seconds": commit_seconds,
                    "proposal_count": len(tree_proposals),
                    "simulations": sum(
                        int(tree_result.get("simulations", 0))
                        for tree_result in tree_proposals
                    ),
                    "worker_seconds_sum": sum(result["worker_seconds"] for result in results),
                    "worker_active_wall_seconds": active_critical,
                    "worker_idle_seconds": idle_seconds,
                    "search_barrier_seconds": barrier_seconds,
                    "worker_profiles": [
                        {
                            "worker": result.get("worker"),
                            "tree_count": len(result.get("trees", [])),
                            "simulations": sum(
                                int(item.get("simulations", 0))
                                for item in result.get("trees", [])
                            ),
                            "active_wall_seconds": float(
                                result.get("worker_seconds", 0.0)
                            ),
                            "reward_feedthrough_seconds": sum(
                                float(item.get("reward_feedthrough_seconds", 0.0))
                                for item in result.get("trees", [])
                            ),
                            "reward_feedthrough_eval_seconds": sum(
                                float(item.get("reward_feedthrough_eval_seconds", 0.0))
                                for item in result.get("trees", [])
                            ),
                            "feedthrough_cache_misses": int(
                                result.get("feedthrough_cache_misses", 0)
                            ),
                            "feedthrough_batch_calls": int(
                                result.get("feedthrough_batch_calls", 0)
                            ),
                            "feedthrough_batch_misses": int(
                                result.get("feedthrough_batch_misses", 0)
                            ),
                        }
                        for result in results
                    ],
                    "dispatch_estimated_ft_work": [
                        {
                            "worker": worker_index,
                            "tree_indices": list(pack),
                            "estimated_ft_work": sum(
                                _estimate_tree_ft_work(
                                    trees_by_index[index], solver.placedb
                                )
                                for index in pack
                            ),
                            "net_count": len({
                                net_id
                                for index in pack
                                for net_id in trees_by_index[index].related_net_ids
                            }),
                        }
                        for worker_index, pack in enumerate(packs_indices)
                    ],
                    "reward_feedthrough_seconds": sum(
                        float(tree_result.get("reward_feedthrough_seconds", 0.0))
                        for tree_result in tree_proposals
                    ),
                    "reward_feedthrough_eval_seconds": sum(
                        float(tree_result.get("reward_feedthrough_eval_seconds", 0.0))
                        for tree_result in tree_proposals
                    ),
                    "feedthrough_cache_hits": sum(
                        int(result.get("feedthrough_cache_hits", 0)) for result in results
                    ),
                    "feedthrough_cache_misses": sum(
                        int(result.get("feedthrough_cache_misses", 0)) for result in results
                    ),
                    "feedthrough_batch_calls": sum(
                        int(result.get("feedthrough_batch_calls", 0)) for result in results
                    ),
                    "feedthrough_batch_requests": sum(
                        int(result.get("feedthrough_batch_requests", 0))
                        for result in results
                    ),
                    "feedthrough_batch_misses": sum(
                        int(result.get("feedthrough_batch_misses", 0)) for result in results
                    ),
                }
            )
    finally:
        resident_shutdown_started = time.perf_counter()
        resident.close()
        resident_shutdown_seconds = time.perf_counter() - resident_shutdown_started
        parallel_trace_reports = [
            report.get("feedthrough_trace")
            for report in resident.stopped_reports
            if report.get("feedthrough_trace")
        ]
    assigned_before_completion = sum(
        1 for group in solver.homology.pin_groups.values() if group.assigned
    )
    capacity_violations_before_completion = len(solver.segment_manager.capacity_violations())
    completion_started = time.perf_counter()
    if final_greedy_completion:
        solver._finalize_unassigned_groups()
    completion_seconds = time.perf_counter() - completion_started
    assigned_after_completion = sum(
        1 for group in solver.homology.pin_groups.values() if group.assigned
    )
    solver.assignment_rounds = len(wave_records)
    solver.total_mcts_simulations = total_parallel_simulations
    output = solver.build_output()
    hashes = _hashes(solver)
    parallel_feedthrough = _final_feedthrough_summary(solver)
    parallel_wall = time.perf_counter() - parallel_started
    baseline_simulations = int(baseline_solver.total_mcts_simulations)
    performance = {
        "baseline_wall_seconds": baseline_wall,
        "baseline_full_lifecycle_wall_seconds": baseline_wall
        + baseline_shutdown_seconds,
        "parallel_wall_seconds": parallel_wall,
        "parallel_algorithm_wall_seconds": max(
            0.0, parallel_wall - resident_shutdown_seconds
        ),
        "parallel_shutdown_seconds": resident_shutdown_seconds,
        "wall_speedup": (baseline_wall / parallel_wall) if parallel_wall else None,
        "algorithm_wall_speedup": (
            baseline_wall / max(1e-12, parallel_wall - resident_shutdown_seconds)
        ),
        "baseline_total_mcts_simulations": baseline_simulations,
        "parallel_total_mcts_simulations": total_parallel_simulations,
        "simulation_budget_equal": baseline_simulations == total_parallel_simulations,
        "parallel_worker_search_seconds_sum": sum(
            record["worker_seconds_sum"] for record in wave_records
        ),
        "parallel_worker_active_wall_seconds": total_worker_active_wall,
        "parallel_worker_idle_seconds": total_worker_idle,
        "parallel_search_barrier_seconds": total_search_barrier,
        "parallel_search_wall_seconds": total_search,
        "parallel_ipc_sync_seconds": total_sync,
        "parallel_parent_commit_seconds": total_commit,
        "parallel_cold_start_seconds": resident.cold_seconds,
        "parallel_final_completion_seconds": completion_seconds,
        "baseline_final_feedthrough_seconds": baseline_feedthrough["seconds"],
        "parallel_final_feedthrough_seconds": parallel_feedthrough["seconds"],
        "baseline_core_adjusted_wall_seconds": max(
            0.0, baseline_wall - baseline_feedthrough["seconds"]
        ),
        "parallel_core_adjusted_wall_seconds": max(
            0.0,
            parallel_wall
            - resident_shutdown_seconds
            - parallel_feedthrough["seconds"],
        ),
        "baseline_reward_feedthrough_seconds": getattr(
            baseline_solver, "mcts_reward_feedthrough_seconds", 0.0
        ),
        "baseline_reward_feedthrough_eval_seconds": getattr(
            baseline_solver, "mcts_reward_feedthrough_eval_seconds", 0.0
        ),
        "baseline_reward_feedthrough_location_seconds": getattr(
            baseline_solver, "mcts_reward_feedthrough_location_seconds", 0.0
        ),
        "baseline_feedthrough_cache_hits": getattr(
            baseline_solver, "mcts_feedthrough_cache_hits", 0
        ),
        "baseline_feedthrough_cache_misses": getattr(
            baseline_solver, "mcts_feedthrough_cache_misses", 0
        ),
        "baseline_shutdown_seconds": baseline_shutdown_seconds,
        "parallel_reward_feedthrough_seconds": total_reward_feedthrough_seconds,
        "parallel_reward_feedthrough_eval_seconds": total_reward_feedthrough_eval_seconds,
        "parallel_reward_feedthrough_location_seconds": total_reward_feedthrough_location_seconds,
        "parallel_feedthrough_batch_calls": sum(
            values[0] for values in worker_feedthrough_batches.values()
        ),
        "parallel_feedthrough_batch_requests": sum(
            values[1] for values in worker_feedthrough_batches.values()
        ),
        "parallel_feedthrough_batch_misses": sum(
            values[2] for values in worker_feedthrough_batches.values()
        ),
        "parallel_ft_trace_concurrency": _trace_concurrency(parallel_trace_reports)
        if parallel_trace_reports
        else {"enabled": False},
    }
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
            "trees_per_worker": trees_per_worker,
            "max_estimated_work": max_estimated_work,
            "random_seed": random_seed,
            "final_greedy_completion": final_greedy_completion,
            "trace_feedthrough": bool(trace_feedthrough),
            "search_mode": solver.mcts_options["search_mode"],
            "segment_subdivision": solver.max_segment_length is not None,
            "effective_budget_profile": {
                "solver_simulations": solver.simulations,
                "basic_dynamic_simulations": solver.mcts_options["basic_dynamic_simulations"],
                "basic_min_simulations": solver.mcts_options["basic_min_simulations"],
                "basic_depth1_simulations": solver.mcts_options["basic_depth1_simulations"],
                "basic_depth2_simulations": solver.mcts_options["basic_depth2_simulations"],
                "hybrid_min_layer_simulations": solver.mcts_options["hybrid_min_layer_simulations"],
                "hybrid_max_layer_simulations": solver.mcts_options["hybrid_max_layer_simulations"],
                "hybrid_max_tree_simulations": solver.mcts_options["hybrid_max_tree_simulations"],
            },
        },
        "baseline_assignment_solver": {
            "wall_seconds": baseline_wall,
            "full_lifecycle_wall_seconds": baseline_wall + baseline_shutdown_seconds,
            "shutdown_seconds": baseline_shutdown_seconds,
            "total_mcts_simulations": baseline_solver.total_mcts_simulations,
            "tree_simulations": list(getattr(baseline_solver, "mcts_tree_simulations", [])),
            "assigned_groups": baseline_output["summary"]["homology_group_count"]
            - baseline_output["summary"]["unassigned_group_count"],
            "unassigned_groups": baseline_output["summary"]["unassigned_group_count"],
            "capacity_violations": len(baseline_output["capacity_violations"]),
            "assignment_hashes": baseline_hashes,
            "hpwl": _hpwl(baseline_solver),
            "feedthrough": baseline_feedthrough,
            "reward_feedthrough_seconds": getattr(
                baseline_solver, "mcts_reward_feedthrough_seconds", 0.0
            ),
            "reward_feedthrough_eval_seconds": getattr(
                baseline_solver, "mcts_reward_feedthrough_eval_seconds", 0.0
            ),
            "reward_feedthrough_location_seconds": getattr(
                baseline_solver, "mcts_reward_feedthrough_location_seconds", 0.0
            ),
            "feedthrough_cache_hits": getattr(
                baseline_solver, "mcts_feedthrough_cache_hits", 0
            ),
            "feedthrough_cache_misses": getattr(
                baseline_solver, "mcts_feedthrough_cache_misses", 0
            ),
            "feedthrough_batch_calls": getattr(
                baseline_solver, "mcts_feedthrough_batch_calls", 0
            ),
            "feedthrough_batch_requests": getattr(
                baseline_solver, "mcts_feedthrough_batch_requests", 0
            ),
            "feedthrough_batch_misses": getattr(
                baseline_solver, "mcts_feedthrough_batch_misses", 0
            ),
            "feedthrough_trace": baseline_trace,
            "summary": baseline_output["summary"],
        },
        "tree_analysis": {
            "tree_count": len(trees),
            "summary": tree_analysis["summary"],
            "depth_histogram": tree_analysis["depth_histogram"],
        },
        "tree_conflicts": {
            "edge_count": len(graph.edge_reasons),
            "reason_edge_counts": graph.reason_counts(),
            "hard_conflict_types": sorted(HARD_CONFLICTS),
            "soft_conflict_types": sorted(set(CONFLICT_WEIGHTS) - HARD_CONFLICTS),
        },
        "waves": {
            "wave_count": len(waves),
            "tree_count": len(trees),
            "hard_conflict_count": sum(wave.hard_conflicts for wave in waves),
            "soft_conflict_count": sum(wave.soft_conflicts for wave in waves),
            "conflict_cost": sum(wave.conflict_cost for wave in waves),
            "records": wave_records,
        },
        "parallel_waves": {
            "cold_worker_seconds": resident.cold_seconds,
            "worker_ready_reports": resident.ready_reports,
            "worker_feedthrough_context_count": sum(
                bool(report.get("feedthrough_context_created"))
                for report in resident.ready_reports
            ),
            "all_workers_have_feedthrough_context": all(
                bool(report.get("feedthrough_context_created"))
                for report in resident.ready_reports
            ),
            "worker_stop_reports": resident.stopped_reports,
            "feedthrough_trace_reports": parallel_trace_reports,
            "feedthrough_trace_concurrency": _trace_concurrency(parallel_trace_reports)
            if parallel_trace_reports
            else {"enabled": False},
            "all_workers_closed_feedthrough_context": all(
                bool(report.get("feedthrough_context_closed"))
                for report in resident.stopped_reports
            ),
            "feedthrough_cache_hits": sum(item[0] for item in worker_feedthrough_counters.values()),
            "feedthrough_cache_misses": sum(item[1] for item in worker_feedthrough_counters.values()),
            "feedthrough_batch_calls": sum(
                values[0] for values in worker_feedthrough_batches.values()
            ),
            "feedthrough_batch_requests": sum(
                values[1] for values in worker_feedthrough_batches.values()
            ),
            "feedthrough_batch_misses": sum(
                values[2] for values in worker_feedthrough_batches.values()
            ),
            "reward_feedthrough_seconds": total_reward_feedthrough_seconds,
            "reward_feedthrough_eval_seconds": total_reward_feedthrough_eval_seconds,
            "reward_feedthrough_location_seconds": total_reward_feedthrough_location_seconds,
            "wall_seconds": parallel_wall,
            "shutdown_seconds": resident_shutdown_seconds,
            "algorithm_wall_seconds": max(0.0, parallel_wall - resident_shutdown_seconds),
            "sync_seconds": total_sync,
            "search_wall_seconds": total_search,
            "worker_active_wall_seconds": total_worker_active_wall,
            "worker_idle_seconds": total_worker_idle,
            "search_barrier_seconds": total_search_barrier,
            "commit_seconds": total_commit,
            "completion_seconds": completion_seconds,
            "total_mcts_simulations": total_parallel_simulations,
            "tree_simulations": [
                {
                    "tree_index": index,
                    "seed": _tree_seed(random_seed, index),
                    "simulations": parallel_tree_simulations[index],
                }
                for index in sorted(parallel_tree_simulations)
            ],
            "proposal_count": total_proposals,
            "accepted_groups_before_completion": assigned_before_completion,
            "accepted_groups_after_completion": assigned_after_completion,
            "final_greedy_added_groups": assigned_after_completion - assigned_before_completion,
            "capacity_violations_before_completion": capacity_violations_before_completion,
            "capacity_violations_after_completion": len(output["capacity_violations"]),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "assignment_hashes": hashes,
            "hpwl": _hpwl(solver),
            "feedthrough": parallel_feedthrough,
            "output_summary": output["summary"],
        },
        "performance": performance,
        "correctness": {
            "parallel_assignment_hash_equal_to_baseline": hashes["assignment_hash"]
            == baseline_hashes["assignment_hash"],
            "parallel_segment_output_hash_equal_to_baseline": hashes["segment_output_hash"]
            == baseline_hashes["segment_output_hash"],
            "parallel_capacity_safe": not output["capacity_violations"],
            "baseline_capacity_safe": not baseline_output["capacity_violations"],
            "baseline_complete": baseline_output["summary"]["unassigned_group_count"] == 0,
            "parallel_complete": output["summary"]["unassigned_group_count"] == 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True)
    parser.add_argument("--pingroup", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--trees-per-worker", type=int, default=4)
    parser.add_argument("--max-estimated-work", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-final-greedy-completion", action="store_true")
    parser.add_argument(
        "--trace-feedthrough",
        action="store_true",
        help="Collect in-memory per-worker FT batch intervals and process metadata.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_parallel_waves(
        args.block,
        args.pingroup,
        workers=args.workers,
        simulations=args.simulations,
        trees_per_worker=args.trees_per_worker,
        max_estimated_work=args.max_estimated_work,
        random_seed=args.seed,
        final_greedy_completion=not args.no_final_greedy_completion,
        trace_feedthrough=args.trace_feedthrough,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote real wave experiment: {args.output}")
    print(json.dumps({key: report[key] for key in ("baseline_assignment_solver", "tree_analysis", "tree_conflicts", "waves", "parallel_waves", "correctness")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
