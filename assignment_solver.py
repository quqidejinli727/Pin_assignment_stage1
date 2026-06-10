"""End-to-end MCTS pin assignment flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

from PlaceDB import Net, Pin, PlaceDB
from homology import HomologyManager, PinHomologyGroup
from mcts import MCTSSolver, get_simulation_budget
from scoring import FeedthroughContext, RewardEvaluator, final_net_metrics
from segment import SegmentManager
from segment_subdivision import percentile_edge_length


class AssignmentSolver:
    """串联解析、同构分组、MCTS 搜索、实际提交和结果输出。"""

    def __init__(
        self,
        block_json_path: str,
        pingroup_json_path: str,
        simulations: int | None = None,
        random_seed: int = 7,
        allow_overflow_fallback: bool = True,
        homology_use_fanout_reuse_for_sorting: bool = True,
        enable_segment_subdivision: bool = True,
        segment_length_percentile: int = 50,
        mcts_search_mode: str = "hybrid",
        mcts_enable_search_diagnostics: bool = False,
        mcts_budget_decay: float = 0.6,
        mcts_tail_decay: float = 0.9,
        mcts_typical_depth: int = 6,
        mcts_space_scale_divisor: float = 1_000_000.0,
        mcts_max_space_factor: float = 10.0,
        mcts_min_layer_simulations: int = 256,
        mcts_tail_depth: int = 8,
        mcts_early_stop_std_multiplier: float = 2.0,
        mcts_enable_tail_early_stop: bool = True,
        mcts_basic_dynamic_simulations: bool = True,
        mcts_basic_space_scale_divisor: float = 100_000.0,
        mcts_basic_max_space_factor: float = 8.0,
        mcts_basic_min_simulations: int = 1024,
        mcts_basic_depth1_simulations: int = 32,
        mcts_basic_depth2_simulations: int = 512,
        mcts_basic_disable_pruning_depth_limit: int = 2,
        mcts_hybrid_basic_depth_limit: int = 4,
        mcts_hybrid_basic_log_space_limit: float = 10.819778284410283,
        mcts_hybrid_beam_width: int = 3,
        mcts_hybrid_tail_beam_width: int = 4,
        mcts_hybrid_tail_depth: int = 24,
        mcts_hybrid_budget_decay: float = 0.65,
        mcts_hybrid_tail_budget_decay: float = 0.92,
        mcts_hybrid_min_layer_simulations: int = 96,
        mcts_hybrid_max_layer_simulations: int = 1024,
        mcts_hybrid_max_tree_simulations: int = 30_000,
        mcts_hybrid_enable_layer_early_stop: bool = True,
        mcts_hybrid_early_stop_std_multiplier: float = 2.0,
        mcts_hybrid_time_limit_seconds: float = 0.0,
        mcts_hybrid_enable_ultradeep_profile: bool = False,
        mcts_hybrid_ultradeep_depth: int = 100,
        mcts_hybrid_max_expanded_depth: int = 64,
        mcts_hybrid_ultradeep_beam_width: int = 1,
        mcts_hybrid_ultradeep_min_layer_simulations: int = 32,
        mcts_hybrid_ultradeep_max_layer_simulations: int = 128,
        mcts_hybrid_use_fast_completion_for_ultradeep: bool = True,
        homology_group_commit_coverage_threshold: float = 1.0,
        mcts_tree_min_committable_group_ratio: float = 0.3,
        mcts_enable_candidate_pruning: bool = True,
        mcts_candidate_top_k: int = 12,
        mcts_candidate_tail_top_k: int = 8,
        mcts_candidate_min_count: int = 16,
        mcts_candidate_score_tolerance: float = 0.03,
        wirelength_reward_weight: float = 0.9,
        feedthrough_weight: float = 0.0,
        reward_normalization_floor: float = 1.0,
        reward_scale: float = 100.0,
        feedthrough_source_dir: str | Path | None = None,
        feedthrough_predict_source_dir: str | Path | None = None,
        feedthrough_evaluate_source_dir: str | Path | None = None,
        feedthrough_reward_source: str = "evaluate",
        enable_feedthrough: bool = True,
        auto_build_feedthrough: bool = False,
        cmake_generator: str | None = None,
    ):
        """初始化数据库、同构管理器、segment 管理器和求解参数。"""
        self.placedb = PlaceDB(block_json_path, pingroup_json_path)
        self.homology = HomologyManager(
            self.placedb,
            use_fanout_reuse_for_sorting=homology_use_fanout_reuse_for_sorting,
        )
        self.max_segment_length = (
            percentile_edge_length(block_json_path, segment_length_percentile)
            if enable_segment_subdivision
            else None
        )
        self.segment_manager = SegmentManager(
            self.placedb,
            max_segment_length=self.max_segment_length,
        )
        self.simulations = simulations
        self.random_seed = random_seed
        self.allow_overflow_fallback = allow_overflow_fallback
        self.feedthrough_weight = feedthrough_weight
        self.enable_feedthrough = enable_feedthrough
        self.wirelength_reward_weight = wirelength_reward_weight
        self.reward_normalization_floor = reward_normalization_floor
        self.reward_scale = reward_scale
        self.feedthrough_source_dir = Path(feedthrough_source_dir) if feedthrough_source_dir else None
        self.feedthrough_predict_source_dir = (
            Path(feedthrough_predict_source_dir)
            if feedthrough_predict_source_dir
            else self.feedthrough_source_dir
        )
        self.feedthrough_evaluate_source_dir = (
            Path(feedthrough_evaluate_source_dir)
            if feedthrough_evaluate_source_dir
            else self.feedthrough_source_dir
        )
        self.feedthrough_reward_source = feedthrough_reward_source
        self.auto_build_feedthrough = auto_build_feedthrough
        self.cmake_generator = cmake_generator
        self.feedthrough_context: FeedthroughContext | None = None
        self.homology_group_commit_coverage_threshold = homology_group_commit_coverage_threshold
        self.mcts_tree_min_committable_group_ratio = mcts_tree_min_committable_group_ratio
        self.skipped_mcts_trees: List[Dict[str, object]] = []
        self.mcts_options = {
            "search_mode": mcts_search_mode,
            "enable_search_diagnostics": mcts_enable_search_diagnostics,
            "budget_decay": mcts_budget_decay,
            "tail_decay": mcts_tail_decay,
            "typical_depth": mcts_typical_depth,
            "space_scale_divisor": mcts_space_scale_divisor,
            "max_space_factor": mcts_max_space_factor,
            "min_layer_simulations": mcts_min_layer_simulations,
            "tail_depth": mcts_tail_depth,
            "early_stop_std_multiplier": mcts_early_stop_std_multiplier,
            "enable_tail_early_stop": mcts_enable_tail_early_stop,
            "basic_dynamic_simulations": mcts_basic_dynamic_simulations,
            "basic_space_scale_divisor": mcts_basic_space_scale_divisor,
            "basic_max_space_factor": mcts_basic_max_space_factor,
            "basic_min_simulations": mcts_basic_min_simulations,
            "basic_depth1_simulations": mcts_basic_depth1_simulations,
            "basic_depth2_simulations": mcts_basic_depth2_simulations,
            "basic_disable_pruning_depth_limit": mcts_basic_disable_pruning_depth_limit,
            "hybrid_basic_depth_limit": mcts_hybrid_basic_depth_limit,
            "hybrid_basic_log_space_limit": mcts_hybrid_basic_log_space_limit,
            "hybrid_beam_width": mcts_hybrid_beam_width,
            "hybrid_tail_beam_width": mcts_hybrid_tail_beam_width,
            "hybrid_tail_depth": mcts_hybrid_tail_depth,
            "hybrid_budget_decay": mcts_hybrid_budget_decay,
            "hybrid_tail_budget_decay": mcts_hybrid_tail_budget_decay,
            "hybrid_min_layer_simulations": mcts_hybrid_min_layer_simulations,
            "hybrid_max_layer_simulations": mcts_hybrid_max_layer_simulations,
            "hybrid_max_tree_simulations": mcts_hybrid_max_tree_simulations,
            "hybrid_enable_layer_early_stop": mcts_hybrid_enable_layer_early_stop,
            "hybrid_early_stop_std_multiplier": mcts_hybrid_early_stop_std_multiplier,
            "hybrid_time_limit_seconds": mcts_hybrid_time_limit_seconds,
            "hybrid_enable_ultradeep_profile": mcts_hybrid_enable_ultradeep_profile,
            "hybrid_ultradeep_depth": mcts_hybrid_ultradeep_depth,
            "hybrid_max_expanded_depth": mcts_hybrid_max_expanded_depth,
            "hybrid_ultradeep_beam_width": mcts_hybrid_ultradeep_beam_width,
            "hybrid_ultradeep_min_layer_simulations": mcts_hybrid_ultradeep_min_layer_simulations,
            "hybrid_ultradeep_max_layer_simulations": mcts_hybrid_ultradeep_max_layer_simulations,
            "hybrid_use_fast_completion_for_ultradeep": mcts_hybrid_use_fast_completion_for_ultradeep,
            "enable_candidate_pruning": mcts_enable_candidate_pruning,
            "candidate_top_k": mcts_candidate_top_k,
            "candidate_tail_top_k": mcts_candidate_tail_top_k,
            "candidate_min_count": mcts_candidate_min_count,
            "candidate_score_tolerance": mcts_candidate_score_tolerance,
            "wirelength_weight": wirelength_reward_weight,
            "feedthrough_weight": feedthrough_weight,
            "reward_normalization_floor": reward_normalization_floor,
            "reward_scale": reward_scale,
            "enable_feedthrough": enable_feedthrough,
        }
        self.assignment_rounds = 0
        self.assignment_progress_index = 0
        self.assignment_issues: List[Dict[str, object]] = []

    def solve(self) -> Dict[str, object]:
        """执行完整分配流程，并返回最终输出数据结构。"""
        self._open_feedthrough_context_if_needed()
        try:
            return self._solve_with_open_context()
        except Exception:
            self.close_feedthrough_context()
            raise

    def _solve_with_open_context(self) -> Dict[str, object]:
        """Run assignment while keeping the optional feedthrough context open."""
        for seed_group in self.homology.unassigned_groups():
            if seed_group.assigned:
                continue
            nets = self.homology.get_related_nets(seed_group)
            if not nets:
                self._assign_group_greedily(seed_group, "no_related_net")
                continue

            pins_in = self._collect_pins(nets)
            related_groups = [
                group
                for group in self.homology.groups_for_pins(pins_in)
                if not group.assigned
            ]
            if not related_groups:
                self._assign_group_greedily(seed_group, "no_related_unassigned_group")
                continue

            pin_full_names = {pin.full_name for pin in pins_in}
            committable_groups = self._committable_groups(related_groups, pin_full_names)
            committable_group_ratio = self._group_ratio(
                len(committable_groups),
                len(related_groups),
            )
            if committable_group_ratio <= self.mcts_tree_min_committable_group_ratio:
                self._record_skipped_mcts_tree(
                    seed_group,
                    related_groups,
                    committable_groups,
                    committable_group_ratio,
                )
                continue

            budget = (
                self.simulations
                if self.simulations is not None
                else get_simulation_budget(len(related_groups))
            )
            mcts = MCTSSolver(
                placedb=self.placedb,
                segment_manager=self.segment_manager,
                groups=related_groups,
                nets=nets,
                simulations=budget,
                random_seed=self.random_seed + self.assignment_rounds,
                feedthrough_context=self.feedthrough_context,
                **self.mcts_options,
            )
            proposed_assignment = mcts.search()
            self._commit_contained_groups(related_groups, pins_in, proposed_assignment)
            self.assignment_rounds += 1

        self._finalize_unassigned_groups()
        return self.build_output()

    def _open_feedthrough_context_if_needed(self) -> None:
        """Open one shared feedthrough context for the full solve when reward needs it."""
        if self.feedthrough_context is not None:
            return
        if not self.enable_feedthrough or self.feedthrough_weight == 0.0:
            return
        source_dir, role = self._feedthrough_reward_context_source()
        if source_dir is None:
            raise ValueError(
                f"feedthrough_{role}_source_dir is required when feedthrough reward is enabled."
            )
        self.feedthrough_context = FeedthroughContext(
            self.placedb,
            source_dir,
            auto_build_feedthrough=self.auto_build_feedthrough,
            cmake_generator=self.cmake_generator,
            role=role,
        )

    def _feedthrough_reward_context_source(self) -> tuple[Path | None, str]:
        """Return the configured feedthrough source directory and loader role for reward."""
        source = self.feedthrough_reward_source.lower().strip()
        if source == "predict":
            return self.feedthrough_predict_source_dir, "predict"
        if source == "evaluate":
            return self.feedthrough_evaluate_source_dir, "evaluate"
        raise ValueError(
            "feedthrough_reward_source must be 'predict' or 'evaluate', "
            f"got {self.feedthrough_reward_source!r}."
        )

    def close_feedthrough_context(self) -> None:
        """Close the shared feedthrough context if it is open."""
        if self.feedthrough_context is not None:
            self.feedthrough_context.close()
            self.feedthrough_context = None

    def final_net_metrics(
        self,
        feedthrough_source_dir: str | Path | None = None,
        enable_feedthrough: bool = True,
        auto_build_feedthrough: bool = False,
        cmake_generator: str | None = None,
    ):
        """Compute final metrics with the feedthrough evaluator executable."""
        source_dir = (
            Path(feedthrough_source_dir)
            if feedthrough_source_dir
            else self.feedthrough_evaluate_source_dir
        )
        if source_dir is None:
            raise ValueError("feedthrough_evaluate_source_dir is required for final metrics.")
        return final_net_metrics(
            self.placedb,
            source_dir,
            enable_feedthrough=enable_feedthrough,
            auto_build_feedthrough=auto_build_feedthrough,
            cmake_generator=cmake_generator,
        )

    def _collect_pins(self, nets: List[Net]) -> List[Pin]:
        """从一批 Net 中收集去重后的 pins_in。"""
        pins_by_name: Dict[str, Pin] = {}
        for net in nets:
            for pin in net.pins:
                pins_by_name[pin.full_name] = pin
        return list(pins_by_name.values())

    def _commit_contained_groups(
        self,
        related_groups: List[PinHomologyGroup],
        pins_in: List[Pin],
        proposed_assignment: Dict[str, str],
    ) -> int:
        """提交当前 pins_in 中完整包含的同构组分配结果。"""
        pin_full_names: Set[str] = {pin.full_name for pin in pins_in}
        contained_groups = self._committable_groups(related_groups, pin_full_names)
        committed_count = 0
        for group in contained_groups:
            if group.assigned:
                continue
            segment_id = proposed_assignment.get(group.name)
            if segment_id is None:
                if self._assign_group_greedily(group, "missing_mcts_assignment"):
                    committed_count += 1
                continue
            invalid_reason = self._validate_segment_assignment(group, segment_id)
            if invalid_reason is not None:
                if self._assign_group_greedily(group, invalid_reason):
                    committed_count += 1
                continue
            self._commit_group_assignment(group, segment_id)
            committed_count += 1
        return committed_count

    def _committable_groups(
        self,
        related_groups: List[PinHomologyGroup],
        pin_full_names: Set[str],
    ) -> List[PinHomologyGroup]:
        """Return groups fully covered or sufficiently represented by current pins_in."""
        threshold = max(0.0, min(1.0, self.homology_group_commit_coverage_threshold))
        committable = []
        for group in related_groups:
            if not group.pins:
                continue
            covered_count = sum(1 for pin in group.pins if pin.full_name in pin_full_names)
            coverage_ratio = covered_count / len(group.pins)
            if coverage_ratio >= threshold:
                committable.append(group)
        return committable

    def _record_skipped_mcts_tree(
        self,
        seed_group: PinHomologyGroup,
        related_groups: List[PinHomologyGroup],
        committable_groups: List[PinHomologyGroup],
        committable_group_ratio: float,
    ) -> None:
        """Record one low-yield local MCTS tree candidate skipped before search."""
        self.skipped_mcts_trees.append(
            {
                "seed_group": seed_group.name,
                "search_group_count": len(related_groups),
                "committable_group_count": len(committable_groups),
                "committable_group_ratio": committable_group_ratio,
                "search_pin_count": self._group_pin_count(related_groups),
                "committable_pin_count": self._group_pin_count(committable_groups),
                "threshold": self.mcts_tree_min_committable_group_ratio,
            }
        )

    @staticmethod
    def _group_pin_count(groups: List[PinHomologyGroup]) -> int:
        """Return the total number of pins in a list of homology groups."""
        return sum(len(group.pins) for group in groups)

    @staticmethod
    def _group_ratio(numerator: int, denominator: int) -> float:
        """Return a safe group-count ratio."""
        return numerator / denominator if denominator else 0.0

    def _validate_segment_assignment(
        self,
        group: PinHomologyGroup,
        segment_id: str,
    ) -> str | None:
        """校验 MCTS 给出的 segment 是否存在、类型匹配且容量可用。"""
        segment = self.segment_manager.abstract_segments.get(segment_id)
        if segment is None:
            return "invalid_mcts_segment"
        if segment.module_name != group.module_name:
            return "segment_module_mismatch"
        if not segment.can_fit(group.max_pin_width):
            return "capacity_exceeded"
        for pin in group.pins:
            if (pin.parent_inst, segment_id) not in self.segment_manager.instance_lookup:
                return "missing_segment_instance"
        return None

    def _assign_group_greedily(self, group: PinHomologyGroup, fallback_reason: str) -> bool:
        """在 MCTS 未给出可提交方案时，用贪心策略兜底分配。"""
        if group.assigned:
            return True
        candidates = self.segment_manager.candidates_for_module(group.module_name)
        if not candidates:
            self._record_assignment_issue(group, "no_candidate_segment", fallback_reason)
            return False

        feasible = [segment for segment in candidates if segment.can_fit(group.max_pin_width)]
        if not feasible:
            if self.allow_overflow_fallback:
                segment = max(candidates, key=lambda item: (item.remaining_capacity, item.capacity))
                self._record_assignment_issue(
                    group,
                    "forced_overflow_assignment",
                    fallback_reason,
                )
                self._commit_group_assignment(group, segment.segment_id, allow_overflow=True)
                return True
            self._record_assignment_issue(group, "capacity_exceeded", fallback_reason)
            return False

        segment = self._best_greedy_segment_by_reward(group, feasible)
        self._commit_group_assignment(group, segment.segment_id)
        return True

    def _best_greedy_segment_by_reward(
        self,
        group: PinHomologyGroup,
        feasible_segments,
    ):
        """Choose the feasible segment that maximizes the same weighted reward as MCTS."""
        related_nets = self.homology.get_related_nets(group)
        if not related_nets:
            return max(feasible_segments, key=lambda item: item.remaining_capacity)

        evaluator = RewardEvaluator(
            related_nets,
            self.placedb,
            wirelength_weight=self.wirelength_reward_weight,
            feedthrough_weight=self.feedthrough_weight,
            enable_feedthrough=self.enable_feedthrough,
            normalization_floor=self.reward_normalization_floor,
            reward_scale=self.reward_scale,
            feedthrough_context=self.feedthrough_context,
        )
        try:
            best_segment = None
            best_reward = float("-inf")
            for segment in feasible_segments:
                temporary_locations = self._temporary_locations_for_group(group, segment.segment_id)
                if len(temporary_locations) != len(group.pins):
                    continue
                reward = evaluator.evaluate(temporary_locations)
                if (
                    best_segment is None
                    or reward > best_reward
                    or (
                        reward == best_reward
                        and segment.remaining_capacity > best_segment.remaining_capacity
                    )
                ):
                    best_segment = segment
                    best_reward = reward
            if best_segment is not None:
                return best_segment
            return max(feasible_segments, key=lambda item: item.remaining_capacity)
        finally:
            evaluator.close()

    def _temporary_locations_for_group(
        self,
        group: PinHomologyGroup,
        segment_id: str,
    ) -> Dict[str, tuple[float, float]]:
        """Return temporary candidate locations for assigning one group to a segment."""
        locations = {}
        for pin in group.pins:
            try:
                instance = self.segment_manager.get_instance(pin.parent_inst, segment_id)
            except KeyError:
                continue
            locations[pin.full_name] = instance.midpoint
        return locations

    def _commit_group_assignment(
        self,
        group: PinHomologyGroup,
        segment_id: str,
        allow_overflow: bool = False,
    ) -> None:
        """把单个同构组实际写入 segment，并更新同构分配状态。"""
        self.segment_manager.apply_assignment(
            group.name,
            group.pins,
            segment_id,
            allow_overflow=allow_overflow,
        )
        self.homology.mark_assigned(group, segment_id)
        self._print_assignment_progress(group)

    def _print_assignment_progress(self, group: PinHomologyGroup) -> None:
        """打印同构组分配进度，编号按实际分配顺序从 0 开始。"""
        print(
            f"homology_batch_index={self.assignment_progress_index}, "
            f"group={group.name}, pin_count={len(group.pins)}"
        )
        self.assignment_progress_index += 1

    def _finalize_unassigned_groups(self) -> None:
        """最终扫描所有未分配组，尽量用兜底策略完成分配。"""
        for group in self.homology.unassigned_groups():
            self._assign_group_greedily(group, "final_unassigned_sweep")

    def _record_assignment_issue(
        self,
        group: PinHomologyGroup,
        reason: str,
        detail: str,
    ) -> None:
        """记录分配异常或 overflow 兜底，便于输出诊断。"""
        self.assignment_issues.append(
            {
                "group": group.name,
                "reason": reason,
                "detail": detail,
                "pin_count": len(group.pins),
                "pin_names": [pin.full_name for pin in group.pins],
            }
        )

    def build_output(self) -> Dict[str, object]:
        """构建包含 summary、诊断信息和 segment 结果的输出字典。"""
        assigned_pin_count = sum(1 for pin in self.placedb.pin_dict.values() if pin.assigned_segment_id)
        unassigned_groups = [
            {
                "group": group.name,
                "pin_count": len(group.pins),
                "pin_names": [pin.full_name for pin in group.pins],
            }
            for group in self.homology.unassigned_groups()
        ]
        capacity_violations = self.segment_manager.capacity_violations()
        skipped_mcts_search_group_count = sum(
            int(item["search_group_count"]) for item in self.skipped_mcts_trees
        )
        skipped_mcts_search_pin_count = sum(
            int(item["search_pin_count"]) for item in self.skipped_mcts_trees
        )
        return {
            "summary": {
                "module_count": len(self.placedb.all_modules_list),
                "net_count": len(self.placedb.nets_list),
                "pin_count": self.placedb.total_pin_count,
                "assigned_pin_count": assigned_pin_count,
                "unassigned_pin_count": self.placedb.total_pin_count - assigned_pin_count,
                "homology_group_count": len(self.homology.pin_groups),
                "unassigned_group_count": len(unassigned_groups),
                "assignment_issue_count": len(self.assignment_issues),
                "capacity_violation_count": len(capacity_violations),
                "assignment_rounds": self.assignment_rounds,
                "skipped_mcts_tree_count": len(self.skipped_mcts_trees),
                "skipped_mcts_search_group_count": skipped_mcts_search_group_count,
                "skipped_mcts_search_pin_count": skipped_mcts_search_pin_count,
                "max_segment_length": self.max_segment_length,
            },
            "unassigned_groups": unassigned_groups,
            "assignment_issues": self.assignment_issues,
            "skipped_mcts_trees": self.skipped_mcts_trees,
            "capacity_violations": capacity_violations,
            "segments": self.segment_manager.to_output_dict(),
        }

    def write_output(self, output_path: str) -> Dict[str, object]:
        """运行求解并把输出 JSON 写到指定路径。"""
        try:
            result = self.solve()
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        finally:
            self.close_feedthrough_context()
