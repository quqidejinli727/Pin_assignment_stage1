"""Asynchronous tree-queue experiment for the Stage1 MCTS flow.

Workers search one immutable tree description at a time.  The parent owns the
real assignment state, validates proposals as they arrive, and is the only
process allowed to commit or greedily repair a capacity conflict.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Mapping, Sequence

from assignment_solver import AssignmentSolver
from scoring import net_hpwl
from stage1_parallel_waves import (
    _final_feedthrough_summary,
    _hashes,
    _hpwl,
    _make_worker_options,
    _pin_state_delta,
    _solver_kwargs,
    _trace_concurrency,
    _tree_payload,
    _wave_worker_main,
)
from stage1_tree_parallel_experiment import TreeDescription, extract_tree_descriptions
from mcts import SKIP_SEGMENT_ID


class ResidentDynamicWorkers:
    """Resident workers driven independently through one queue per process."""

    def __init__(
        self,
        block_json: str,
        pingroup_json: str,
        options: Mapping[str, object],
        worker_count: int,
    ) -> None:
        self.queues = []
        self.result_queue = mp.Queue()
        self.processes = []
        self.ready_reports = []
        self.stopped_reports = []
        started = time.perf_counter()
        for worker_index in range(max(1, int(worker_count))):
            command_queue = mp.Queue()
            process = mp.Process(
                target=_wave_worker_main,
                args=(
                    worker_index,
                    block_json,
                    pingroup_json,
                    options,
                    command_queue,
                    self.result_queue,
                ),
            )
            process.start()
            self.queues.append(command_queue)
            self.processes.append(process)
        while len(self.ready_reports) < len(self.processes):
            result = self.get(timeout=120.0)
            if result.get("kind") == "ready":
                self.ready_reports.append(result)
            elif result.get("kind") == "error":
                raise RuntimeError(str(result.get("error")))
        self.cold_seconds = time.perf_counter() - started

    def get(self, timeout: float = 120.0) -> dict:
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty as error:
            exitcodes = [process.exitcode for process in self.processes]
            raise RuntimeError(
                f"dynamic worker result timeout; exitcodes={exitcodes}"
            ) from error

    def dispatch(self, worker_index: int, state: dict, payload: dict) -> None:
        command_queue = self.queues[worker_index]
        # Commands from one producer to one worker are FIFO.  The worker
        # applies the latest parent snapshot before starting this tree.
        command_queue.put(("sync", state))
        command_queue.put(("search", [payload]))

    def close(self) -> float:
        started = time.perf_counter()
        for command_queue in self.queues:
            command_queue.put(("stop", None))
        while len(self.stopped_reports) < len(self.processes):
            result = self.get(timeout=120.0)
            if result.get("kind") == "stopped":
                self.stopped_reports.append(result)
            elif result.get("kind") == "error":
                raise RuntimeError(str(result.get("error")))
        for process in self.processes:
            process.join(timeout=30.0)
            if process.exitcode not in (0, None):
                raise RuntimeError(
                    f"dynamic worker {process.pid} exited with {process.exitcode}"
                )
        return time.perf_counter() - started


def _ensure_parent_feedthrough(solver: AssignmentSolver) -> float:
    """Open the parent FT context lazily for reward-aware greedy repair."""

    if solver.feedthrough_context is not None:
        return 0.0
    started = time.perf_counter()
    solver._open_feedthrough_context_if_needed()
    return time.perf_counter() - started


def _affected_metrics(solver: AssignmentSolver, group) -> dict:
    """Measure HPWL/FT only on nets touched by one repaired pingroup."""

    nets = solver.homology.get_related_nets(group)
    hpwl = sum(net_hpwl(net, solver.placedb, {}) for net in nets)
    feedthrough = 0.0
    if solver.enable_feedthrough and solver.feedthrough_weight != 0.0:
        _ensure_parent_feedthrough(solver)
        context = solver.feedthrough_context
        if context is not None:
            for net in nets:
                if len(net.pins) <= 1:
                    continue
                locations = {
                    pin.full_name: solver.placedb.get_pin_location_estimate(pin)
                    for pin in net.pins
                }
                feedthrough += float(
                    context.run_one_net_at_locations(net, locations)
                )
    return {"hpwl": hpwl, "feedthrough": feedthrough, "net_count": len(nets)}


def _commit_dynamic_tree(
    solver: AssignmentSolver,
    tree: TreeDescription,
    tree_result: Mapping[str, object],
    *,
    worker_index: int,
    dispatch_version: int,
    current_version: int,
) -> tuple[Counter, list[dict], int, list[str]]:
    """Validate one returned proposal and greedily repair capacity conflicts."""

    proposal = dict(tree_result.get("proposal") or {})
    counts: Counter = Counter()
    greedy_records = []
    committed_names = []
    version = current_version
    for group_name in tree.committable_group_names:
        counts["proposal_groups_considered"] += 1
        group = solver.homology.pin_groups[group_name]
        if group.assigned:
            counts["already_assigned"] += 1
            continue
        segment_id = proposal.get(group_name)
        if segment_id is None:
            counts["missing_group_proposal"] += 1
            continue
        if segment_id == SKIP_SEGMENT_ID:
            counts["skip_proposal"] += 1
            continue
        invalid = solver._validate_segment_assignment(group, segment_id)
        if invalid is None:
            solver._commit_group_assignment(group, segment_id)
            counts["direct_commit"] += 1
            committed_names.append(group.name)
            version += 1
            continue
        if invalid != "capacity_exceeded":
            counts[invalid] += 1
            continue

        counts["capacity_conflict"] += 1
        counts["greedy_attempt"] += 1
        segment = solver.segment_manager.abstract_segments.get(segment_id)
        conflict_state = {
            "used_width": float(segment.used_width) if segment is not None else None,
            "capacity": float(segment.capacity) if segment is not None else None,
            "required_width": float(group.max_pin_width),
        }
        parent_context_cold = _ensure_parent_feedthrough(solver)
        before = _affected_metrics(solver, group)
        succeeded = bool(
            solver._assign_group_greedily(group, "dynamic_capacity_conflict")
        )
        after = _affected_metrics(solver, group)
        if succeeded:
            counts["greedy_success"] += 1
            committed_names.append(group.name)
            version += 1
        else:
            counts["greedy_failed"] += 1
        greedy_records.append(
            {
                "tree_id": tree.tree_index,
                "worker_id": worker_index,
                "pingroup_name": group_name,
                "proposal_segment": segment_id,
                "conflict_reason": invalid,
                "conflict_state": conflict_state,
                "greedy_segment": group.assigned_segment_id,
                "success": succeeded,
                "overflow": any(
                    item.get("segment_id") == group.assigned_segment_id
                    for item in solver.segment_manager.capacity_violations()
                ),
                "snapshot_version": dispatch_version,
                "commit_version": current_version,
                "staleness": current_version - dispatch_version,
                "parent_ft_cold_seconds": parent_context_cold,
                "affected_net_count": before["net_count"],
                "hpwl_before": before["hpwl"],
                "hpwl_after": after["hpwl"],
                "hpwl_delta": after["hpwl"] - before["hpwl"],
                "feedthrough_before": before["feedthrough"],
                "feedthrough_after": after["feedthrough"],
                "feedthrough_delta": after["feedthrough"]
                - before["feedthrough"],
            }
        )
    return counts, greedy_records, version, committed_names


def run_dynamic_queue(
    block_json: str | Path,
    pingroup_json: str | Path,
    *,
    workers: int = 2,
    simulations: int = 1024,
    random_seed: int = 7,
    trace_feedthrough: bool = False,
    solver_overrides: Mapping[str, object] | None = None,
) -> dict:
    """Run all static tree descriptions through an asynchronous work queue."""

    block_json = str(block_json)
    pingroup_json = str(pingroup_json)
    kwargs = _solver_kwargs(block_json, pingroup_json, simulations, random_seed)
    if solver_overrides:
        kwargs.update(dict(solver_overrides))
    # Capacity-conflict repair must never silently use the overflow fallback.
    kwargs["allow_overflow_fallback"] = False
    solver = AssignmentSolver(**kwargs)
    trees, tree_analysis = extract_tree_descriptions(
        block_json,
        pingroup_json,
        coverage_threshold=solver.homology_group_commit_coverage_threshold,
        skip_uncovered_groups=solver.homology_skip_uncovered_groups,
        skip_coverage_threshold=solver.homology_skip_coverage_threshold,
        use_fanout_reuse_for_sorting=solver.homology.use_fanout_reuse_for_sorting,
    )
    trees_by_index = {tree.tree_index: tree for tree in trees}
    pending = deque(sorted(trees, key=lambda tree: tree.tree_index))
    options = _make_worker_options(solver, trace_feedthrough=trace_feedthrough)

    started = time.perf_counter()
    resident = ResidentDynamicWorkers(
        block_json, pingroup_json, options, max(1, workers)
    )
    queue_started = time.perf_counter()
    busy: Dict[int, dict] = {}
    worker_active_seconds: Counter = Counter()
    worker_tree_counts: Counter = Counter()
    worker_feedthrough_counters: Dict[int, tuple[int, int]] = {}
    worker_feedthrough_batches: Dict[int, tuple[int, int, int]] = {}
    tree_records = []
    greedy_records = []
    commit_counts: Counter = Counter()
    segment_conflicts: Counter = Counter()
    total_simulations = 0
    total_reward_ft = 0.0
    total_reward_ft_eval = 0.0
    total_reward_ft_location = 0.0
    sync_ack_count = 0
    commit_version = 0
    predispatch_skips = 0
    completion_order = 0
    commit_history: list[str] = []
    worker_sync_versions = {
        worker_index: 0 for worker_index in range(len(resident.queues))
    }
    sync_delta_groups = 0
    sync_delta_pins = 0

    def dispatch_next(worker_index: int) -> bool:
        nonlocal predispatch_skips, sync_delta_groups, sync_delta_pins
        while pending:
            tree = pending.popleft()
            payload = _tree_payload(tree, solver)
            if payload is None:
                predispatch_skips += 1
                continue
            busy[worker_index] = {
                "tree": tree,
                "dispatch_version": commit_version,
                "dispatched_at": time.perf_counter(),
                "payload_group_count": len(payload["group_names"]),
            }
            from_version = worker_sync_versions[worker_index]
            changed_groups = commit_history[from_version:commit_version]
            state = _pin_state_delta(
                solver,
                changed_groups,
                from_version=from_version,
                to_version=commit_version,
            )
            sync_delta_groups += len(state["assigned_groups"])
            sync_delta_pins += len(state["pins"])
            resident.dispatch(worker_index, state, payload)
            worker_sync_versions[worker_index] = commit_version
            return True
        return False

    for worker_index in range(len(resident.queues)):
        dispatch_next(worker_index)

    try:
        while busy:
            result = resident.get(timeout=300.0)
            kind = result.get("kind")
            if kind == "synced":
                sync_ack_count += 1
                continue
            if kind == "error":
                raise RuntimeError(str(result.get("error")))
            if kind != "searched":
                continue
            worker_index = int(result["worker"])
            task = busy.pop(worker_index)
            tree = task["tree"]
            returned_trees = list(result.get("trees") or [])
            if len(returned_trees) != 1:
                raise RuntimeError(
                    f"worker {worker_index} returned {len(returned_trees)} trees"
                )
            tree_result = returned_trees[0]
            completion_order += 1
            arrival_version = commit_version
            counts, repairs, commit_version, committed_names = _commit_dynamic_tree(
                solver,
                tree,
                tree_result,
                worker_index=worker_index,
                dispatch_version=int(task["dispatch_version"]),
                current_version=commit_version,
            )
            commit_history.extend(committed_names)
            if len(commit_history) != commit_version:
                raise RuntimeError(
                    "dynamic commit history/version mismatch: "
                    f"{len(commit_history)} != {commit_version}"
                )
            commit_counts.update(counts)
            greedy_records.extend(repairs)
            for repair in repairs:
                segment_conflicts[str(repair["proposal_segment"])] += 1
            simulations_done = int(tree_result.get("simulations", 0))
            total_simulations += simulations_done
            total_reward_ft += float(
                tree_result.get("reward_feedthrough_seconds", 0.0)
            )
            total_reward_ft_eval += float(
                tree_result.get("reward_feedthrough_eval_seconds", 0.0)
            )
            total_reward_ft_location += float(
                tree_result.get("reward_feedthrough_location_seconds", 0.0)
            )
            active_seconds = float(result.get("worker_seconds", 0.0))
            worker_active_seconds[worker_index] += active_seconds
            worker_tree_counts[worker_index] += 1
            worker_feedthrough_counters[worker_index] = (
                int(result.get("feedthrough_cache_hits", 0)),
                int(result.get("feedthrough_cache_misses", 0)),
            )
            worker_feedthrough_batches[worker_index] = (
                int(result.get("feedthrough_batch_calls", 0)),
                int(result.get("feedthrough_batch_requests", 0)),
                int(result.get("feedthrough_batch_misses", 0)),
            )
            tree_records.append(
                {
                    "tree_index": tree.tree_index,
                    "worker": worker_index,
                    "completion_order": completion_order,
                    "dispatch_version": task["dispatch_version"],
                    "commit_version_before": arrival_version,
                    "staleness": arrival_version - int(task["dispatch_version"]),
                    "payload_group_count": task["payload_group_count"],
                    "simulations": simulations_done,
                    "search_seconds": float(
                        tree_result.get("search_seconds", active_seconds)
                    ),
                    "elapsed_since_dispatch": time.perf_counter()
                    - float(task["dispatched_at"]),
                    "commit_counts": dict(counts),
                }
            )
            dispatch_next(worker_index)
    finally:
        queue_seconds = time.perf_counter() - queue_started
        shutdown_seconds = resident.close()

    solver.assignment_rounds = len(tree_records)
    solver.total_mcts_simulations = total_simulations
    output = solver.build_output()
    assignment_hashes = _hashes(solver)
    hpwl = _hpwl(solver)
    final_feedthrough = _final_feedthrough_summary(solver)
    parent_shutdown_started = time.perf_counter()
    solver.close_feedthrough_context()
    parent_shutdown_seconds = time.perf_counter() - parent_shutdown_started
    total_wall = time.perf_counter() - started

    trace_reports = [
        report.get("feedthrough_trace")
        for report in resident.stopped_reports
        if report.get("feedthrough_trace")
    ]
    active_sum = sum(worker_active_seconds.values())
    worker_count = len(resident.queues)
    non_search_seconds = max(0.0, worker_count * queue_seconds - active_sum)
    greedy_count = int(commit_counts["greedy_attempt"])
    considered = int(commit_counts["proposal_groups_considered"])
    core_adjusted = max(
        0.0,
        total_wall
        - resident.cold_seconds
        - shutdown_seconds
        - parent_shutdown_seconds
        - float(final_feedthrough["seconds"]),
    )
    return {
        "input": {"block": block_json, "pingroup": pingroup_json},
        "parameters": {
            "workers": worker_count,
            "simulations": simulations,
            "random_seed": random_seed,
            "search_mode": solver.mcts_options["search_mode"],
            "feedthrough_enabled": solver.enable_feedthrough,
            "feedthrough_weight": solver.feedthrough_weight,
            "feedthrough_reward_source": solver.feedthrough_reward_source,
            "trace_feedthrough": bool(trace_feedthrough),
            "commit_policy": "completion_order",
            "overflow_fallback": False,
        },
        "tree_analysis": {
            "tree_count": len(trees),
            "summary": tree_analysis.get("summary", {}),
        },
        "performance": {
            "wall_seconds": total_wall,
            "cold_start_seconds": resident.cold_seconds,
            "queue_search_commit_seconds": queue_seconds,
            "worker_shutdown_seconds": shutdown_seconds,
            "parent_context_shutdown_seconds": parent_shutdown_seconds,
            "final_feedthrough_seconds": final_feedthrough["seconds"],
            "core_adjusted_seconds": core_adjusted,
            "worker_active_seconds_sum": active_sum,
            "worker_non_search_or_idle_seconds": non_search_seconds,
            "reward_feedthrough_seconds": total_reward_ft,
            "reward_feedthrough_eval_seconds": total_reward_ft_eval,
            "reward_feedthrough_location_seconds": total_reward_ft_location,
        },
        "scheduler": {
            "dispatched_tree_count": len(tree_records),
            "predispatch_skipped_tree_count": predispatch_skips,
            "sync_ack_count": sync_ack_count,
            "sync_mode": "monotonic_delta",
            "sync_delta_group_records": sync_delta_groups,
            "sync_delta_pin_records": sync_delta_pins,
            "stale_tree_count": sum(
                int(record["staleness"] > 0) for record in tree_records
            ),
            "average_staleness": (
                sum(int(record["staleness"]) for record in tree_records)
                / len(tree_records)
                if tree_records
                else 0.0
            ),
            "max_staleness": max(
                (int(record["staleness"]) for record in tree_records),
                default=0,
            ),
            "completion_tree_order": [
                record["tree_index"] for record in tree_records
            ],
            "worker_tree_counts": dict(sorted(worker_tree_counts.items())),
            "worker_active_seconds": dict(sorted(worker_active_seconds.items())),
            "tree_records": tree_records,
        },
        "commit": {
            "counts": dict(sorted(commit_counts.items())),
            "greedy_ratio": greedy_count / considered if considered else 0.0,
            "greedy_records": greedy_records,
            "segment_conflict_counts": dict(sorted(segment_conflicts.items())),
        },
        "feedthrough_workers": {
            "context_count": sum(
                bool(report.get("feedthrough_context_created"))
                for report in resident.ready_reports
            ),
            "all_contexts_closed": all(
                bool(report.get("feedthrough_context_closed"))
                for report in resident.stopped_reports
            ),
            "cache_hits": sum(item[0] for item in worker_feedthrough_counters.values()),
            "cache_misses": sum(item[1] for item in worker_feedthrough_counters.values()),
            "batch_calls": sum(item[0] for item in worker_feedthrough_batches.values()),
            "batch_requests": sum(item[1] for item in worker_feedthrough_batches.values()),
            "batch_misses": sum(item[2] for item in worker_feedthrough_batches.values()),
            "trace_reports": trace_reports,
            "trace_concurrency": _trace_concurrency(trace_reports)
            if trace_reports
            else {"enabled": False},
        },
        "result": {
            "summary": output["summary"],
            "capacity_violations": output["capacity_violations"],
            "assignment_issues": output["assignment_issues"],
            "assignment_hashes": assignment_hashes,
            "hpwl": hpwl,
            "feedthrough": final_feedthrough,
        },
    }
