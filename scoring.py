"""Reward、HPWL 和 feedthrough 最终评估函数。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import hashlib
import importlib.util
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Iterable, List, Set, Tuple

from PlaceDB import Net, PlaceDB
from geometry_utils import Point


_MISSING_POINT = object()


def feedthrough_reward(_: Iterable[Net]) -> float:
    """feedthrough 指标预留接口，当前原型权重为 0。"""
    return 0.0


def net_hpwl(
    net: Net,
    placedb: PlaceDB,
    temporary_locations: Dict[str, Point],
    excluded_pin_names: Set[str] | None = None,
    *,
    skipped_pin_names: Set[str] | None = None,
) -> float:
    """Compute HPWL while removing only explicitly invalid design pins.

    ``skipped_pin_names`` remains a compatibility alias for callers that used
    the old name for permanent exclusion.  Deferred MCTS groups must not be
    supplied through either argument.
    """
    excluded = set(excluded_pin_names or ())
    excluded.update(skipped_pin_names or ())
    locations_get = temporary_locations.get
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    active_count = 0
    for pin in net.pins:
        pin_name = pin.full_name
        if pin_name in excluded:
            continue
        point = locations_get(pin_name, _MISSING_POINT)
        if point is _MISSING_POINT:
            point = placedb.get_pin_location_estimate(pin)
        x, y = point
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
        active_count += 1
    if active_count <= 1:
        return 0.0
    return (max_x - min_x) + (max_y - min_y)

@dataclass
class NetReferenceMetrics:
    """Reference metrics computed from each pin's parent-block centroid."""

    hpwl: float
    feedthrough: float = 0.0


class FeedthroughContext:
    """Own one long-lived feedthrough predictor session for a full Stage 1 run."""

    def __init__(
        self,
        placedb: PlaceDB,
        feedthrough_source_dir: Path,
        *,
        auto_build_feedthrough: bool = False,
        cmake_generator: str | None = None,
        role: str = "feedthrough",
        trace_enabled: bool = False,
    ):
        init_started = time.perf_counter()
        self.placedb = placedb
        executable_started = time.perf_counter()
        executable = ensure_ftpred_executable(
            feedthrough_source_dir,
            auto_build=auto_build_feedthrough,
            cmake_generator=cmake_generator,
        )
        self.role = role
        self.trace_enabled = bool(trace_enabled)
        self.trace_context_id = f"{os.getpid()}:{id(self)}"
        self.trace_created_epoch = time.time()
        self.trace_closed_epoch = None
        self.trace_intervals = []
        self.trace_dropped_intervals = 0
        self.trace_batch_count = 0
        self.trace_batch_net_count = 0
        self.trace_batch_miss_count = 0
        self.trace_batch_wall_seconds = 0.0
        self.trace_batch_eval_seconds = 0.0
        self.executable = executable
        self.trace_executable_seconds = time.perf_counter() - executable_started
        loader_started = time.perf_counter()
        self.ftpred_loader = load_ftpred_loader(feedthrough_source_dir, role)
        self.trace_loader_seconds = time.perf_counter() - loader_started
        modules_started = time.perf_counter()
        modules_text = self.ftpred_loader.build_modules_text(placedb)
        self.trace_modules_seconds = time.perf_counter() - modules_started
        session_started = time.perf_counter()
        self.session = self.ftpred_loader.FtpredBinSession(str(executable), modules_text)
        self.trace_session_seconds = time.perf_counter() - session_started
        self.trace_init_seconds = time.perf_counter() - init_started
        self.trace_child_pid = getattr(getattr(self.session, "_p", None), "pid", None)
        self.feedthrough_cache: Dict[Tuple[object, ...], float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_calls = 0
        self.batch_request_count = 0
        self.batch_miss_count = 0

    def close(self) -> None:
        """Close the long-lived predictor session."""
        if self.session is not None:
            self.session.close()
            self.session = None
        if self.trace_enabled and self.trace_closed_epoch is None:
            self.trace_closed_epoch = time.time()

    def trace_summary(self) -> dict:
        """Return in-memory FT trace data; no filesystem I/O is performed."""
        session = self.session
        process = getattr(session, "_p", None) if session is not None else None
        durations = sorted(item[2] for item in self.trace_intervals)
        p95_index = int(0.95 * (len(durations) - 1)) if durations else 0
        return {
            "enabled": self.trace_enabled,
            "context_id": self.trace_context_id,
            "worker_pid": os.getpid(),
            "child_pid": self.trace_child_pid or getattr(process, "pid", None),
            "executable": str(self.executable),
            "init_seconds": self.trace_init_seconds,
            "executable_seconds": self.trace_executable_seconds,
            "loader_seconds": self.trace_loader_seconds,
            "modules_seconds": self.trace_modules_seconds,
            "session_seconds": self.trace_session_seconds,
            "created_epoch": self.trace_created_epoch,
            "closed_epoch": self.trace_closed_epoch,
            "batch_count": self.trace_batch_count,
            "batch_net_count": self.trace_batch_net_count,
            "batch_miss_count": self.trace_batch_miss_count,
            "batch_wall_seconds": self.trace_batch_wall_seconds,
            "batch_eval_seconds": self.trace_batch_eval_seconds,
            "batch_mean_seconds": (
                sum(durations) / len(durations) if durations else 0.0
            ),
            "batch_p95_seconds": durations[p95_index] if durations else 0.0,
            "first_batch_epoch": self.trace_intervals[0][0]
            if self.trace_intervals
            else None,
            "last_batch_epoch": self.trace_intervals[-1][1]
            if self.trace_intervals
            else None,
            "intervals": list(self.trace_intervals),
            "dropped_intervals": self.trace_dropped_intervals,
        }

    def run_one_net_at_locations(
        self,
        net: Net,
        locations: Dict[str, Point],
        excluded_pin_names: Set[str] | None = None,
    ) -> float:
        """Evaluate one net after temporarily applying the given pin locations."""
        if self.session is None:
            raise RuntimeError("FeedthroughContext has already been closed.")

        excluded_pin_names = excluded_pin_names or set()
        old_pin_locations = [(pin, pin.x, pin.y) for pin in net.pins]
        old_pin_scopes = [(pin, pin.scope) for pin in net.pins]
        old_net_pins = list(net.pins)
        old_net_feedthrough = getattr(net, "feedthrough", 0.0)
        try:
            if excluded_pin_names:
                net.pins = [pin for pin in net.pins if pin.full_name not in excluded_pin_names]
                for pin in old_net_pins:
                    if pin.full_name in excluded_pin_names:
                        pin.scope = []
            for pin in net.pins:
                pin.x, pin.y = locations[pin.full_name]
            with redirect_stdout(StringIO()):
                return float(self.session.run_one_net(self.placedb, net))
        finally:
            for pin, x, y in old_pin_locations:
                pin.x = x
                pin.y = y
            for pin, scope in old_pin_scopes:
                pin.scope = scope
            net.pins = old_net_pins
            net.feedthrough = old_net_feedthrough

    def run_one_net_at_locations_cached(
        self,
        net: Net,
        cache_key: Tuple[object, ...],
        locations_factory: Callable[[], Dict[str, Point]],
        excluded_pin_names: Set[str] | None = None,
    ) -> float:
        """Evaluate one net with a full-flow cache and lazy location construction."""
        if cache_key in self.feedthrough_cache:
            self.cache_hits += 1
            return self.feedthrough_cache[cache_key]
        self.cache_misses += 1
        value = self.run_one_net_at_locations(net, locations_factory(), excluded_pin_names)
        self.feedthrough_cache[cache_key] = value
        return value

    def run_many_net_at_locations_cached(self, requests) -> list[float]:
        """Batch cache misses while preserving per-net exact evaluator values."""
        trace_started = time.time() if self.trace_enabled else 0.0
        hits_before = self.cache_hits
        misses_before = self.cache_misses
        if not requests:
            self.last_batch_location_seconds = 0.0
            self.last_batch_eval_seconds = 0.0
            return []
        self.batch_calls += 1
        self.batch_request_count += len(requests)
        values = [None] * len(requests)
        misses = []
        for index, (net, cache_key, locations_factory, excluded_pin_names) in enumerate(requests):
            if cache_key in self.feedthrough_cache:
                self.cache_hits += 1
                values[index] = self.feedthrough_cache[cache_key]
            else:
                self.cache_misses += 1
                misses.append((index, net, cache_key, locations_factory, excluded_pin_names))
        if misses:
            self.batch_miss_count += len(misses)
            prepared = []
            location_seconds = 0.0
            for _, net, _, locations_factory, excluded_pin_names in misses:
                started = time.perf_counter()
                locations = locations_factory()
                location_seconds += time.perf_counter() - started
                old_pin_locations = [(pin, pin.x, pin.y) for pin in net.pins]
                old_pin_scopes = [(pin, pin.scope) for pin in net.pins]
                old_net_pins = list(net.pins)
                old_net_feedthrough = getattr(net, "feedthrough", 0.0)
                try:
                    if excluded_pin_names:
                        net.pins = [
                            pin for pin in net.pins
                            if pin.full_name not in excluded_pin_names
                        ]
                        for pin in old_net_pins:
                            if pin.full_name in excluded_pin_names:
                                pin.scope = []
                    for pin in net.pins:
                        pin.x, pin.y = locations[pin.full_name]
                    nets_text, build_info = self.ftpred_loader.build_nets_text(
                        self.placedb, net, return_stats=True
                    )
                    prepared.append((net, nets_text, build_info))
                finally:
                    for pin, x, y in old_pin_locations:
                        pin.x, pin.y = x, y
                    for pin, scope in old_pin_scopes:
                        pin.scope = scope
                    net.pins = old_net_pins
                    net.feedthrough = old_net_feedthrough
            eval_started = time.perf_counter()
            if hasattr(self.session, "run_many_nets"):
                computed = self.session.run_many_nets(self.placedb, prepared)
            else:
                computed = [
                    self.run_one_net_at_locations(net, locations_factory(), excluded_pin_names)
                    for _, net, _, locations_factory, excluded_pin_names in misses
                ]
            eval_elapsed = time.perf_counter() - eval_started
            self.last_batch_location_seconds = location_seconds
            self.last_batch_eval_seconds = eval_elapsed
            for (index, _, cache_key, _, _), value in zip(misses, computed):
                values[index] = value
                self.feedthrough_cache[cache_key] = value
        else:
            self.last_batch_location_seconds = 0.0
            self.last_batch_eval_seconds = 0.0
        if self.trace_enabled:
            ended = time.time()
            wall = max(0.0, ended - trace_started)
            self.trace_batch_count += 1
            self.trace_batch_net_count += len(requests)
            self.trace_batch_miss_count += self.cache_misses - misses_before
            self.trace_batch_wall_seconds += wall
            self.trace_batch_eval_seconds += self.last_batch_eval_seconds
            if len(self.trace_intervals) < 200000:
                self.trace_intervals.append(
                    (
                        trace_started,
                        ended,
                        wall,
                        self.cache_hits - hits_before,
                        self.cache_misses - misses_before,
                        self.last_batch_eval_seconds,
                        len(requests),
                    )
                )
            else:
                self.trace_dropped_intervals += 1
        return [float(value) for value in values]


class RewardEvaluator:
    """Evaluate MCTS rewards with per-net reference normalization."""

    def __init__(
        self,
        nets: Iterable[Net],
        placedb: PlaceDB,
        *,
        wirelength_weight: float = 1.0,
        feedthrough_weight: float = 0.0,
        enable_feedthrough: bool = True,
        normalization_floor: float = 1.0,
        reward_scale: float = 1.0,
        feedthrough_context: FeedthroughContext | None = None,
        excluded_pin_names: Set[str] | None = None,
        skipped_pin_names: Set[str] | None = None,
    ):
        self.nets = list(nets)
        self.placedb = placedb
        self.wirelength_weight = wirelength_weight
        self.feedthrough_weight = feedthrough_weight
        self.enable_feedthrough = enable_feedthrough and feedthrough_weight != 0.0
        self.normalization_floor = normalization_floor
        self.reward_scale = reward_scale
        self.feedthrough_context = feedthrough_context
        self.excluded_pin_names = set(excluded_pin_names or ())
        self.excluded_pin_names.update(skipped_pin_names or ())
        # Backward-compatible diagnostics alias.  New code uses the physical
        # meaning explicitly and never fills it from deferred groups.
        self.skipped_pin_names = self.excluded_pin_names
        self._net_active_pins = {
            id(net): tuple(
                pin for pin in net.pins if pin.full_name not in self.excluded_pin_names
            )
            for net in self.nets
        }
        self.metric_nets = [
            net for net in self.nets if len(self._net_active_pins[id(net)]) > 1
        ]
        self.skipped_single_pin_net_count = len(self.nets) - len(self.metric_nets)
        self._pin_estimate_cache: Dict[str, Point] = {}
        self.timing_profile: Dict[str, float] = {
            "reward_total": 0.0,
            "reward_hpwl": 0.0,
            "reward_feedthrough": 0.0,
            "reward_feedthrough_location": 0.0,
            "reward_feedthrough_eval": 0.0,
            "reward_hpwl_reference": 0.0,
        }
        self.last_batch_location_seconds = 0.0
        self.last_batch_eval_seconds = 0.0
        self.hpwl_profile: Dict[str, int] = {
            "candidate_calls": 0,
            "candidate_pin_visits": 0,
            "temporary_location_hits": 0,
            "base_location_hits": 0,
            "reference_calls": 0,
            "reference_pin_visits": 0,
        }
        self._net_pin_names = {
            id(net): {pin.full_name for pin in self._net_active_pins[id(net)]}
            for net in self.metric_nets
        }
        self._net_base_locations = {
            id(net): {
                pin.full_name: self._pin_base_location(pin)
                for pin in self._net_active_pins[id(net)]
            }
            for net in self.metric_nets
        }
        # FeedthroughContext is intentionally shared by successive local trees.
        # Include every physical endpoint's base location in candidate cache keys,
        # because a deferred endpoint can move from its module centroid to a real
        # assigned-segment midpoint between trees without becoming an action here.
        self._net_base_location_signatures = {
            net_id: tuple(
                sorted(
                    (pin_name, self._rounded_point(point))
                    for pin_name, point in locations.items()
                )
            )
            for net_id, locations in self._net_base_locations.items()
        }

        if self.enable_feedthrough and self.feedthrough_context is None:
            raise ValueError("feedthrough_context is required when feedthrough reward is enabled.")

        self.reference_metrics = {
            id(net): self._build_reference_metrics(net)
            for net in self.metric_nets
        }

    def close(self) -> None:
        """Release local evaluator caches without closing the shared predictor."""
        self._pin_estimate_cache.clear()

    def evaluate(self, temporary_locations: Dict[str, Point]) -> float:
        """Return weighted normalized reward for a complete candidate assignment."""
        started = time.perf_counter()
        total_reward = 0.0
        wire_rewards = []
        feedthrough_requests = []
        for net in self.metric_nets:
            reference = self.reference_metrics[id(net)]
            hpwl_started = time.perf_counter()
            candidate_hpwl = self._candidate_hpwl(net, temporary_locations)
            self.timing_profile["reward_hpwl"] += time.perf_counter() - hpwl_started
            wirelength_reward = self._normalized_improvement(reference.hpwl, candidate_hpwl)
            wire_rewards.append((reference, wirelength_reward))
            if self.enable_feedthrough:
                key, locations_factory = self._candidate_feedthrough_request(
                    net, temporary_locations
                )
                feedthrough_requests.append(
                    (net, key, locations_factory, self.excluded_pin_names)
                )
        feedthrough_values = []
        if self.enable_feedthrough:
            ft_started = time.perf_counter()
            if hasattr(self.feedthrough_context, "run_many_net_at_locations_cached"):
                feedthrough_values = self.feedthrough_context.run_many_net_at_locations_cached(
                    feedthrough_requests
                )
                self.timing_profile["reward_feedthrough_eval"] += (
                    self.feedthrough_context.last_batch_eval_seconds
                )
                self.timing_profile["reward_feedthrough_location"] += (
                    self.feedthrough_context.last_batch_location_seconds
                )
            else:
                feedthrough_values = [
                    self._feedthrough_at_locations(net, key, factory, skipped)
                    for net, key, factory, skipped in feedthrough_requests
                ]
            self.timing_profile["reward_feedthrough"] += time.perf_counter() - ft_started
        for index, (reference, wirelength_reward) in enumerate(wire_rewards):
            feedthrough_reward_value = (
                self._normalized_improvement(reference.feedthrough, feedthrough_values[index])
                if self.enable_feedthrough
                else 0.0
            )
            total_reward += (
                self.wirelength_weight * wirelength_reward
                + self.feedthrough_weight * feedthrough_reward_value
            )
        self.timing_profile["reward_total"] += time.perf_counter() - started
        return total_reward * self.reward_scale

    def _build_reference_metrics(self, net: Net) -> NetReferenceMetrics:
        reference_started = time.perf_counter()
        reference_locations = {
            pin.full_name: self.placedb.get_module(pin.parent_inst).get_centroid()
            for pin in self._net_active_pins[id(net)]
        }
        reference_hpwl = self._hpwl_from_location_map(
            self._net_active_pins[id(net)],
            reference_locations,
        )
        self.timing_profile["reward_hpwl_reference"] += (
            time.perf_counter() - reference_started
        )
        self.hpwl_profile["reference_calls"] += 1
        self.hpwl_profile["reference_pin_visits"] += len(self._net_active_pins[id(net)])
        reference_feedthrough = (
            self._feedthrough_at_locations(
                net,
                ("reference", id(net), tuple(sorted(self.excluded_pin_names))),
                lambda: reference_locations,
                self.excluded_pin_names,
            )
            if self.enable_feedthrough
            else 0.0
        )
        return NetReferenceMetrics(reference_hpwl, reference_feedthrough)

    @staticmethod
    def _hpwl_from_location_map(pins, locations: Dict[str, Point]) -> float:
        """Compute HPWL from a complete cached location map without allocations."""
        if len(pins) <= 1:
            return 0.0
        first_x, first_y = locations[pins[0].full_name]
        min_x = max_x = first_x
        min_y = max_y = first_y
        for pin in pins[1:]:
            x, y = locations[pin.full_name]
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
        return (max_x - min_x) + (max_y - min_y)

    def _candidate_hpwl(
        self,
        net: Net,
        temporary_locations: Dict[str, Point],
    ) -> float:
        """Compute candidate HPWL from cached base locations in one pass."""
        net_id = id(net)
        pins = self._net_active_pins[net_id]
        base_locations = self._net_base_locations[net_id]
        locations_get = temporary_locations.get
        first_pin_name = pins[0].full_name
        first_point = locations_get(first_pin_name, _MISSING_POINT)
        temporary_hits = 0
        if first_point is _MISSING_POINT:
            first_point = base_locations[first_pin_name]
            base_hits = 1
        else:
            temporary_hits = 1
            base_hits = 0
        min_x = max_x = first_point[0]
        min_y = max_y = first_point[1]
        for pin in pins[1:]:
            pin_name = pin.full_name
            point = locations_get(pin_name, _MISSING_POINT)
            if point is _MISSING_POINT:
                point = base_locations[pin_name]
                base_hits += 1
            else:
                temporary_hits += 1
            x, y = point
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
        self.hpwl_profile["candidate_calls"] += 1
        self.hpwl_profile["candidate_pin_visits"] += len(pins)
        self.hpwl_profile["temporary_location_hits"] += temporary_hits
        self.hpwl_profile["base_location_hits"] += base_hits
        return (max_x - min_x) + (max_y - min_y)

    def _candidate_feedthrough(
        self,
        net: Net,
        temporary_locations: Dict[str, Point],
    ) -> float:
        key, locations_factory = self._candidate_feedthrough_request(net, temporary_locations)
        return self._feedthrough_at_locations(
            net, key, locations_factory, self.excluded_pin_names
        )

    def _candidate_feedthrough_request(
        self,
        net: Net,
        temporary_locations: Dict[str, Point],
    ):
        net_id = id(net)
        base_locations = self._net_base_locations[net_id]
        net_pin_names = self._net_pin_names[net_id]
        changed = []
        for pin_name, point in temporary_locations.items():
            if pin_name not in net_pin_names:
                continue
            rounded = self._rounded_point(point)
            if rounded != self._rounded_point(base_locations[pin_name]):
                changed.append((pin_name, rounded))
        key = (
            "candidate",
            net_id,
            tuple(sorted(self.excluded_pin_names)),
            self._net_base_location_signatures[net_id],
            tuple(sorted(changed)),
        )
        return key, lambda: self._net_locations_from_changes(net, temporary_locations)

    def _feedthrough_at_locations(
        self,
        net: Net,
        cache_key: Tuple[object, ...],
        locations_factory: Callable[[], Dict[str, Point]],
        excluded_pin_names: Set[str] | None = None,
    ) -> float:
        if self.feedthrough_context is None:
            return 0.0
        loc_elapsed = 0.0

        def timed_locations_factory():
            nonlocal loc_elapsed
            started = time.perf_counter()
            locations = locations_factory()
            loc_elapsed += time.perf_counter() - started
            return locations

        eval_started = time.perf_counter()
        if not hasattr(self.feedthrough_context, "run_one_net_at_locations_cached"):
            if not excluded_pin_names:
                value = self.feedthrough_context.run_one_net_at_locations(
                    net,
                    timed_locations_factory(),
                )
            else:
                value = self.feedthrough_context.run_one_net_at_locations(
                    net,
                    timed_locations_factory(),
                    excluded_pin_names,
                )
            self.timing_profile["reward_feedthrough_eval"] += time.perf_counter() - eval_started
            self.timing_profile["reward_feedthrough_location"] += loc_elapsed
            return value
        if not excluded_pin_names:
            value = self.feedthrough_context.run_one_net_at_locations_cached(
                net,
                cache_key,
                timed_locations_factory,
            )
        else:
            value = self.feedthrough_context.run_one_net_at_locations_cached(
                net,
                cache_key,
                timed_locations_factory,
                excluded_pin_names,
            )
        self.timing_profile["reward_feedthrough_eval"] += time.perf_counter() - eval_started
        self.timing_profile["reward_feedthrough_location"] += loc_elapsed
        return value

    def _normalized_improvement(self, reference: float, candidate: float) -> float:
        denominator = max(abs(reference), self.normalization_floor)
        return (reference - candidate) / denominator

    def _pin_base_location(self, pin) -> Point:
        if pin.full_name not in self._pin_estimate_cache:
            self._pin_estimate_cache[pin.full_name] = self.placedb.get_pin_location_estimate(pin)
        return self._pin_estimate_cache[pin.full_name]

    def _net_locations_from_changes(
        self,
        net: Net,
        temporary_locations: Dict[str, Point],
    ) -> Dict[str, Point]:
        base_locations = self._net_base_locations[id(net)]
        net_pin_names = self._net_pin_names[id(net)]
        locations = dict(base_locations)
        for pin_name, point in temporary_locations.items():
            if pin_name in net_pin_names:
                locations[pin_name] = point
        return locations

    @staticmethod
    def _rounded_point(point: Point) -> Point:
        return (round(float(point[0]), 6), round(float(point[1]), 6))


def assignment_reward(
    nets: Iterable[Net],
    placedb: PlaceDB,
    temporary_locations: Dict[str, Point],
    feedthrough_weight: float = 0.0,
) -> float:
    """综合 HPWL 和 feedthrough，返回 MCTS 使用的 reward。"""
    evaluator = RewardEvaluator(
        nets,
        placedb,
        feedthrough_weight=feedthrough_weight,
        enable_feedthrough=False,
    )
    return evaluator.evaluate(temporary_locations)


@dataclass
class NetMetrics:
    """记录最终分配后单条 net 的 HPWL 与 feedthrough。"""

    net_id: int
    pin_count: int
    hpwl: float
    feedthrough: float


def _loader_candidates(source_path: Path) -> List[Path]:
    """Return ftpred_loader.py candidates beside a predictor/evaluator path."""
    if source_path.is_file():
        return [source_path.parent / "ftpred_loader.py"]
    return [
        source_path / "ftpred_loader.py",
        source_path / "feedthrough" / "ftpred_loader.py",
    ]


def load_ftpred_loader(source_path: Path, role: str) -> ModuleType:
    """Load the predictor/evaluator's own ftpred_loader.py with an isolated name."""
    source_path = Path(source_path)
    for candidate in _loader_candidates(source_path):
        if candidate.exists():
            loader_path = candidate
            break
    else:
        checked = ", ".join(str(path) for path in _loader_candidates(source_path))
        raise FileNotFoundError(
            f"未找到 {role} 使用的 ftpred_loader.py，已检查：{checked}"
        )

    digest = hashlib.sha1(str(loader_path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"_stage1_{role}_ftpred_loader_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {role} ftpred_loader：{loader_path}")

    loader_dir = str(loader_path.parent)
    added_to_path = False
    if loader_dir not in sys.path:
        sys.path.insert(0, loader_dir)
        added_to_path = True
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if added_to_path:
            try:
                sys.path.remove(loader_dir)
            except ValueError:
                pass


def _platform_executable_names(base_name: str) -> List[str]:
    """Return executable name variants for the current platform."""
    names = [base_name]
    if os.name == "nt" and not base_name.lower().endswith(".exe"):
        names.insert(0, f"{base_name}.exe")
    return names


def _predictor_candidates(source_path: Path) -> List[Path]:
    """返回 feedthrough 预测器/评估器可能的可执行文件路径。"""
    if source_path.is_file():
        return [source_path]

    executable_names: List[str] = []
    for base_name in ("ftpred", source_path.name):
        for name in _platform_executable_names(base_name):
            if name not in executable_names:
                executable_names.append(name)

    candidates: List[Path] = []
    for name in executable_names:
        candidates.extend(
            [
                source_path / name,
                source_path / "build" / "Release" / name,
                source_path / "build" / name,
            ]
        )
    return candidates


def _cmake_executable() -> str:
    """查找 CMake 命令，兼容通过 bundled Python 安装的 CMake。"""
    command = shutil.which("cmake")
    if command:
        return command
    scripts_candidate = Path(sys.executable).parent / "Scripts" / "cmake.exe"
    if scripts_candidate.exists():
        return str(scripts_candidate)
    return "cmake"


def ensure_ftpred_executable(
    source_dir: Path,
    auto_build: bool = False,
    cmake_generator: str | None = None,
) -> Path:
    """查找预测器可执行文件；默认不编译，显式 auto_build=True 时才调用 CMake。"""
    source_dir = Path(source_dir)
    for candidate in _predictor_candidates(source_dir):
        if candidate.exists():
            return candidate

    if not auto_build:
        raise FileNotFoundError(
            f"未找到 feedthrough 可执行文件，请检查路径或预先编译：{source_dir}"
        )

    build_dir = source_dir / "build"
    try:
        configure_command = [
            _cmake_executable(),
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        if cmake_generator:
            configure_command.extend(["-G", cmake_generator])
        subprocess.run(configure_command, check=True)
        subprocess.run(
            [_cmake_executable(), "--build", str(build_dir), "--config", "Release"],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "首次使用 feedthrough 预测器需要完成 CMake 编译，但编译失败。"
        ) from exc

    for candidate in _predictor_candidates(source_dir):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("CMake 已执行，但仍未找到 ftpred 可执行文件。")


def final_net_metrics(
    placedb: PlaceDB,
    feedthrough_source_dir: Path,
    enable_feedthrough: bool = True,
    auto_build_feedthrough: bool = False,
    cmake_generator: str | None = None,
    feedthrough_context: FeedthroughContext | None = None,
) -> List[NetMetrics]:
    """计算最终分配后所有 net 的 HPWL 与 feedthrough。

    feedthrough 开启时只建立一次 ``FtpredBinSession``，随后对全部 net
    循环调用该常驻进程，避免每条 net 重复启动预测器。
    """
    metrics = [
        NetMetrics(
            net_id=net.net_id,
            pin_count=len(net.pins),
            hpwl=net_hpwl(net, placedb, {}),
            feedthrough=0.0,
        )
        for net in placedb.nets_list
    ]
    if not enable_feedthrough:
        return metrics

    if feedthrough_context is not None:
        for metric, net in zip(metrics, placedb.nets_list):
            if len(net.pins) <= 1:
                continue
            locations = {
                pin.full_name: placedb.get_pin_location_estimate(pin)
                for pin in net.pins
            }
            metric.feedthrough = float(feedthrough_context.run_one_net_at_locations(net, locations))
        return metrics

    executable = ensure_ftpred_executable(
        feedthrough_source_dir,
        auto_build=auto_build_feedthrough,
        cmake_generator=cmake_generator,
    )
    ftpred_loader = load_ftpred_loader(feedthrough_source_dir, "evaluate")
    modules_text = ftpred_loader.build_modules_text(placedb)
    with ftpred_loader.FtpredBinSession(str(executable), modules_text) as session:
        for metric, net in zip(metrics, placedb.nets_list):
            if len(net.pins) <= 1:
                continue
            # loader 会为每条 net 打印解析明细；最终报告已统一记录指标，
            # 这里收起内部进度输出，保持主程序输出简洁。
            with redirect_stdout(StringIO()):
                metric.feedthrough = float(session.run_one_net(placedb, net))
    return metrics


def summarize_metrics(metrics: List[NetMetrics]) -> Dict[str, float | int]:
    """汇总所有 net 指标，生成数量、总值和平均值。"""
    count = len(metrics)
    metric_count = sum(1 for metric in metrics if metric.pin_count > 1)
    total_hpwl = sum(metric.hpwl for metric in metrics)
    total_feedthrough = sum(metric.feedthrough for metric in metrics)
    return {
        "net_count": count,
        "metric_net_count": metric_count,
        "skipped_single_pin_net_count": count - metric_count,
        "total_hpwl": total_hpwl,
        "average_hpwl": total_hpwl / metric_count if metric_count else 0.0,
        "total_feedthrough": total_feedthrough,
        "average_feedthrough": total_feedthrough / metric_count if metric_count else 0.0,
    }


def metrics_to_records(metrics: List[NetMetrics]) -> List[dict]:
    """将指标对象转换为 JSON 可序列化列表。"""
    return [asdict(metric) for metric in metrics]
