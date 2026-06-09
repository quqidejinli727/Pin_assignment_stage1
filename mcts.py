"""MCTS tree for one local ``pins_in`` assignment batch."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from PlaceDB import Net, PlaceDB
from geometry_utils import Point
from homology import PinHomologyGroup
from scoring import FeedthroughContext, RewardEvaluator, net_hpwl
from segment import AbstractSegment, SegmentManager, SegmentUsage


Action = Tuple[str, str]


@dataclass
class MCTSNode:
    """MCTS 树节点，表示已为前若干同构组选择了 segment。"""

    group_index: int
    usage: SegmentUsage
    assignments: Dict[str, str]
    parent: Optional["MCTSNode"] = None
    action: Optional[Action] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: Optional[List[Action]] = None

    @property
    def average_reward(self) -> float:
        """返回该节点的平均 reward，用于选择最佳路径。"""
        if self.visits == 0:
            return float("-inf")
        return self.total_reward / self.visits


@dataclass
class SearchProfile:
    """Static search-space summary for one local MCTS tree."""

    depth: int
    branch_counts: List[int]
    average_branching: float
    max_branching: int
    log_total_space: float

    @property
    def has_dead_layer(self) -> bool:
        return any(count <= 0 for count in self.branch_counts)


class MCTSSolver:
    """针对一批 pins_in 相关同构组运行一棵 MCTS 树。"""

    def __init__(
        self,
        placedb: PlaceDB,
        segment_manager: SegmentManager,
        groups: List[PinHomologyGroup],
        nets: Iterable[Net],
        simulations: int = 128,
        exploration_constant: float = 1.414,
        random_seed: int = 7,
        search_mode: str = "layered",
        budget_decay: float = 0.6,
        tail_decay: float = 0.9,
        typical_depth: int = 6,
        space_scale_divisor: float = 1_000_000.0,
        max_space_factor: float = 10.0,
        min_layer_simulations: int = 256,
        tail_depth: int = 8,
        early_stop_std_multiplier: float = 2.0,
        enable_tail_early_stop: bool = True,
        enable_search_diagnostics: bool = False,
        basic_dynamic_simulations: bool = True,
        basic_space_scale_divisor: float = 1_000_000.0,
        basic_max_space_factor: float = 10.0,
        basic_min_simulations: int = 256,
        basic_depth1_simulations: int = 32,
        basic_depth2_simulations: int = 256,
        basic_disable_pruning_depth_limit: int = 1,
        hybrid_basic_depth_limit: int = 10,
        hybrid_basic_log_space_limit: float = math.log(1_000_000.0),
        hybrid_beam_width: int = 4,
        hybrid_tail_beam_width: int = 8,
        hybrid_tail_depth: int = 20,
        hybrid_budget_decay: float = 0.7,
        hybrid_tail_budget_decay: float = 0.9,
        hybrid_min_layer_simulations: int = 128,
        hybrid_max_layer_simulations: int = 2048,
        hybrid_max_tree_simulations: int = 20_000,
        hybrid_enable_layer_early_stop: bool = True,
        hybrid_early_stop_std_multiplier: float = 2.0,
        hybrid_time_limit_seconds: float = 0.0,
        hybrid_enable_ultradeep_profile: bool = False,
        hybrid_ultradeep_depth: int = 100,
        hybrid_max_expanded_depth: int = 64,
        hybrid_ultradeep_beam_width: int = 1,
        hybrid_ultradeep_min_layer_simulations: int = 32,
        hybrid_ultradeep_max_layer_simulations: int = 128,
        hybrid_use_fast_completion_for_ultradeep: bool = True,
        enable_candidate_pruning: bool = True,
        candidate_top_k: int = 12,
        candidate_tail_top_k: int = 8,
        candidate_min_count: int = 16,
        candidate_score_tolerance: float = 0.05,
        wirelength_weight: float = 1.0,
        feedthrough_weight: float = 0.0,
        reward_normalization_floor: float = 1.0,
        reward_scale: float = 1.0,
        enable_feedthrough: bool = True,
        feedthrough_context: FeedthroughContext | None = None,
    ):
        """初始化 MCTS 搜索所需的数据、参数和随机数种子。"""
        self.placedb = placedb
        self.segment_manager = segment_manager
        self.groups = groups
        self.nets = list(nets)
        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self.random = random.Random(random_seed)
        self.search_mode = search_mode
        self.budget_decay = budget_decay
        self.tail_decay = tail_decay
        self.typical_depth = typical_depth
        self.space_scale_divisor = space_scale_divisor
        self.max_space_factor = max_space_factor
        self.min_layer_simulations = min_layer_simulations
        self.tail_depth = tail_depth
        self.early_stop_std_multiplier = early_stop_std_multiplier
        self.enable_tail_early_stop = enable_tail_early_stop
        self.enable_search_diagnostics = enable_search_diagnostics
        self.basic_dynamic_simulations = basic_dynamic_simulations
        self.basic_space_scale_divisor = basic_space_scale_divisor
        self.basic_max_space_factor = basic_max_space_factor
        self.basic_min_simulations = basic_min_simulations
        self.basic_depth1_simulations = basic_depth1_simulations
        self.basic_depth2_simulations = basic_depth2_simulations
        self.basic_disable_pruning_depth_limit = basic_disable_pruning_depth_limit
        self._active_basic_depth: int | None = None
        self.hybrid_basic_depth_limit = hybrid_basic_depth_limit
        self.hybrid_basic_log_space_limit = hybrid_basic_log_space_limit
        self.hybrid_beam_width = max(1, hybrid_beam_width)
        self.hybrid_tail_beam_width = max(1, hybrid_tail_beam_width)
        self.hybrid_tail_depth = hybrid_tail_depth
        self.hybrid_budget_decay = hybrid_budget_decay
        self.hybrid_tail_budget_decay = hybrid_tail_budget_decay
        self.hybrid_min_layer_simulations = hybrid_min_layer_simulations
        self.hybrid_max_layer_simulations = hybrid_max_layer_simulations
        self.hybrid_max_tree_simulations = hybrid_max_tree_simulations
        self.hybrid_enable_layer_early_stop = hybrid_enable_layer_early_stop
        self.hybrid_early_stop_std_multiplier = hybrid_early_stop_std_multiplier
        self.hybrid_time_limit_seconds = hybrid_time_limit_seconds
        self.hybrid_enable_ultradeep_profile = hybrid_enable_ultradeep_profile
        self.hybrid_ultradeep_depth = hybrid_ultradeep_depth
        self.hybrid_max_expanded_depth = hybrid_max_expanded_depth
        self.hybrid_ultradeep_beam_width = max(1, hybrid_ultradeep_beam_width)
        self.hybrid_ultradeep_min_layer_simulations = hybrid_ultradeep_min_layer_simulations
        self.hybrid_ultradeep_max_layer_simulations = hybrid_ultradeep_max_layer_simulations
        self.hybrid_use_fast_completion_for_ultradeep = hybrid_use_fast_completion_for_ultradeep
        self.enable_candidate_pruning = enable_candidate_pruning
        self.candidate_top_k = max(1, candidate_top_k)
        self.candidate_tail_top_k = max(1, candidate_tail_top_k)
        self.candidate_min_count = max(1, candidate_min_count)
        self.candidate_score_tolerance = max(0.0, candidate_score_tolerance)
        self._candidate_score_cache: Dict[Tuple[str, str], float] = {}
        self.last_search_profile: SearchProfile | None = None
        self.last_search_diagnostics: Dict[str, object] = {}
        self.reward_evaluator = RewardEvaluator(
            self.nets,
            self.placedb,
            wirelength_weight=wirelength_weight,
            feedthrough_weight=feedthrough_weight,
            enable_feedthrough=enable_feedthrough,
            normalization_floor=reward_normalization_floor,
            reward_scale=reward_scale,
            feedthrough_context=feedthrough_context,
        )

    def search(self) -> Dict[str, str]:
        """执行已配置的 MCTS 搜索策略。"""
        try:
            if self.search_mode == "basic":
                return self._search_basic()
            if self.search_mode == "layered":
                return self._search_layered()
            if self.search_mode == "hybrid":
                return self._search_hybrid()
            raise ValueError(
                f"Unsupported MCTS search mode: {self.search_mode!r}. "
                "Use 'basic', 'layered', or 'hybrid'."
            )
        finally:
            self.reward_evaluator.close()

    def _search_layered(self) -> Dict[str, str]:
        """逐层执行 MCTS 搜索，并返回同构组到抽象 segment 的分配方案。"""
        root = MCTSNode(
            group_index=0,
            usage=self.segment_manager.snapshot_usage(),
            assignments={},
        )

        if not self.groups:
            return {}

        total_budget = self._total_simulation_budget(root.usage)
        current = root
        depth = 1
        while current.group_index < len(self.groups):
            children = self._ensure_children(current)
            if not children:
                if current is root:
                    return self._greedy_assignment(root.usage)
                return self._complete_greedily(current.assignments, current.usage)

            for _ in range(self._layer_budget(total_budget, depth)):
                child = self._select_layer_child(children)
                reward = self._simulate(child)
                self._record_layer_result(current, child, reward)

            best = max(children, key=lambda child: self._ucb(child))
            next_depth = depth + 1
            if (
                self.enable_tail_early_stop
                and next_depth > self.tail_depth
                and self._has_decisive_ucb_lead(children)
            ):
                current = best
                break

            current = best
            depth = next_depth

        if len(current.assignments) < len(self.groups):
            return self._complete_greedily(current.assignments, current.usage)
        return current.assignments

    def _search_basic(self) -> Dict[str, str]:
        """执行基础版 MCTS：固定模拟次数后直接抽取当前树最佳路径。"""
        root = MCTSNode(
            group_index=0,
            usage=self.segment_manager.snapshot_usage(),
            assignments={},
        )

        if not self.groups:
            return {}

        profile = self._search_profile(root.usage)
        self.last_search_profile = profile
        self._active_basic_depth = profile.depth
        if profile.depth == 1:
            return self._search_depth1_greedy(root)

        for _ in range(self._basic_simulation_budget(root.usage, profile)):
            node = self._select(root)
            if node.group_index < len(self.groups):
                node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

        best = self._best_child_by_reward(root)
        if best is None:
            return self._greedy_assignment(root.usage)
        return self._extract_best_path(best)

    def _search_depth1_greedy(self, root: MCTSNode) -> Dict[str, str]:
        """For a one-layer tree, exhaustively score all candidates with the full reward."""
        group = self.groups[0]
        feasible = self._candidate_segments(group, root.usage)
        if not feasible:
            return self._greedy_assignment(root.usage)
        segment = self._best_completion_segment_by_reward(group, feasible, root.assignments)
        return {group.name: segment.segment_id}

    def _search_hybrid(self) -> Dict[str, str]:
        """Run adaptive Hybrid search: Basic for small trees, beam-layered otherwise."""
        root = MCTSNode(
            group_index=0,
            usage=self.segment_manager.snapshot_usage(),
            assignments={},
        )
        if not self.groups:
            return {}

        profile = self._search_profile(root.usage)
        self.last_search_profile = profile
        if self._hybrid_uses_basic(profile):
            self.last_search_diagnostics = {
                "mode": "hybrid",
                "route": "basic",
                "depth": profile.depth,
                "log_total_space": profile.log_total_space,
            }
            return self._search_basic()

        return self._search_hybrid_beam(root, profile)

    def _search_hybrid_beam(
        self,
        root: MCTSNode,
        profile: SearchProfile,
    ) -> Dict[str, str]:
        """Beam-layered Hybrid search for medium/deep local trees."""
        beam = [root]
        total_budget = self._hybrid_total_budget(profile)
        used_simulations = 0
        start_time = time.monotonic()
        ultradeep_profile = self._hybrid_uses_ultradeep_profile(profile)
        route = (
            "ultradeep"
            if ultradeep_profile
            else "tail" if self._hybrid_uses_tail_profile(profile) else "beam"
        )
        expanded_depth = self._hybrid_expanded_depth(profile)
        self.last_search_diagnostics = {
            "mode": "hybrid",
            "route": route,
            "depth": profile.depth,
            "expanded_depth": expanded_depth,
            "log_total_space": profile.log_total_space,
            "total_budget": total_budget,
            "layer_budgets": [],
        }

        for depth in range(1, expanded_depth + 1):
            if not beam or self._hybrid_time_exceeded(start_time):
                break
            if used_simulations >= total_budget:
                break

            tail_profile = self._hybrid_uses_tail_profile(profile, depth)
            ultradeep_layer = ultradeep_profile
            children: List[MCTSNode] = []
            for node in beam:
                if node.group_index >= len(self.groups):
                    children.append(node)
                    continue
                actions = self._actions_for_node(node, tail_profile=tail_profile or ultradeep_layer)
                if not actions:
                    continue
                for action in actions:
                    children.append(self._create_child(node, action))

            if not children:
                best_partial = max(beam, key=lambda node: len(node.assignments))
                if best_partial is root:
                    return self._greedy_assignment(root.usage)
                return self._complete_greedily(best_partial.assignments, best_partial.usage)

            layer_budget = min(
                self._hybrid_layer_budget(total_budget, depth, tail_profile, ultradeep_layer),
                total_budget - used_simulations,
            )
            self.last_search_diagnostics["layer_budgets"].append(layer_budget)
            for _ in range(layer_budget):
                if self._hybrid_time_exceeded(start_time):
                    break
                child = self._select_layer_child(children)
                reward = self._simulate(child)
                self._record_layer_result(child.parent or root, child, reward)
                used_simulations += 1
                if used_simulations >= total_budget:
                    break

            beam = self._select_hybrid_beam(children, tail_profile, ultradeep_layer)
            if (
                self.hybrid_enable_layer_early_stop
                and self._has_decisive_score_lead(
                    children,
                    self.hybrid_early_stop_std_multiplier,
                )
            ):
                break

        best = self._best_node_from_beam(beam)
        if best is None:
            return self._greedy_assignment(root.usage)
        if len(best.assignments) < len(self.groups):
            if ultradeep_profile and self.hybrid_use_fast_completion_for_ultradeep:
                return self._complete_fast_by_heuristic(best.assignments, best.usage)
            return self._complete_greedily(best.assignments, best.usage)
        return best.assignments

    def _basic_simulation_budget(
        self,
        usage: SegmentUsage,
        profile: SearchProfile | None = None,
    ) -> int:
        """Return the actual Basic-mode simulation count for this local tree."""
        profile = profile or self._search_profile(usage)
        if profile.depth == 1 and self.basic_depth1_simulations > 0:
            return max(1, self.basic_depth1_simulations)
        if profile.depth == 2 and self.basic_depth2_simulations > 0:
            return max(1, self.basic_depth2_simulations)
        if not self.basic_dynamic_simulations:
            return self.simulations
        space_factor = self._total_search_space_factor(usage)
        scaled_budget = math.ceil(self.simulations * space_factor)
        return max(self.basic_min_simulations, scaled_budget)

    def _search_profile(self, usage: SegmentUsage) -> SearchProfile:
        """Build a static search-space profile for the current local tree."""
        branch_counts = [
            len(self._raw_feasible_segments(group, usage))
            for group in self.groups
            if not group.assigned
        ]
        log_total_space = 0.0
        for count in branch_counts:
            if count <= 0:
                log_total_space = float("-inf")
                break
            log_total_space += math.log(count)
        average_branching = (
            sum(branch_counts) / len(branch_counts)
            if branch_counts
            else 0.0
        )
        return SearchProfile(
            depth=len(branch_counts),
            branch_counts=branch_counts,
            average_branching=average_branching,
            max_branching=max(branch_counts, default=0),
            log_total_space=log_total_space,
        )

    def _hybrid_uses_basic(self, profile: SearchProfile) -> bool:
        """Return whether Hybrid should route this tree to the Basic fast path."""
        return (
            profile.depth <= self.hybrid_basic_depth_limit
            and profile.log_total_space <= self.hybrid_basic_log_space_limit
        )

    def _hybrid_uses_tail_profile(
        self,
        profile: SearchProfile,
        depth: int | None = None,
    ) -> bool:
        """Return whether Hybrid should use the tail profile for this tree/layer."""
        current_depth = depth if depth is not None else profile.depth
        return (
            current_depth >= self.hybrid_tail_depth
            or profile.depth >= self.hybrid_tail_depth
            or profile.log_total_space > self.hybrid_basic_log_space_limit * 2
            or profile.max_branching >= self.candidate_min_count * 4
        )

    def _hybrid_uses_ultradeep_profile(self, profile: SearchProfile) -> bool:
        """Return whether Hybrid should use the ultra-deep bounded prefix profile."""
        return (
            self.hybrid_enable_ultradeep_profile
            and self.hybrid_ultradeep_depth > 0
            and profile.depth >= self.hybrid_ultradeep_depth
        )

    def _hybrid_expanded_depth(self, profile: SearchProfile) -> int:
        """Return how many layers Hybrid should explicitly search before completion."""
        if not self._hybrid_uses_ultradeep_profile(profile):
            return len(self.groups)
        if self.hybrid_max_expanded_depth <= 0:
            return len(self.groups)
        return min(len(self.groups), self.hybrid_max_expanded_depth)

    def _hybrid_total_budget(self, profile: SearchProfile) -> int:
        """Return the capped total simulation budget for Hybrid beam search."""
        if profile.has_dead_layer:
            return self.hybrid_min_layer_simulations
        if self.basic_space_scale_divisor > 0 and profile.log_total_space != float("-inf"):
            log_factor = profile.log_total_space - math.log(self.basic_space_scale_divisor)
            factor = math.exp(min(log_factor, math.log(max(1.0, self.basic_max_space_factor))))
            factor = max(1.0, factor)
        else:
            factor = 1.0
        budget = max(
            self.hybrid_min_layer_simulations,
            math.ceil(self.simulations * factor),
        )
        if self.hybrid_max_tree_simulations > 0:
            budget = min(budget, self.hybrid_max_tree_simulations)
        return budget

    def _hybrid_layer_budget(
        self,
        total_budget: int,
        depth: int,
        tail_profile: bool,
        ultradeep_profile: bool = False,
    ) -> int:
        """Return a capped per-layer simulation budget for Hybrid."""
        if tail_profile:
            multiplier = (
                self.hybrid_budget_decay ** max(self.hybrid_tail_depth - 1, 0)
                * self.hybrid_tail_budget_decay ** max(depth - self.hybrid_tail_depth, 0)
            )
        else:
            multiplier = self.hybrid_budget_decay ** (depth - 1)
        budget = max(
            self.hybrid_min_layer_simulations,
            math.ceil(total_budget * multiplier),
        )
        if ultradeep_profile:
            budget = max(
                self.hybrid_ultradeep_min_layer_simulations,
                math.ceil(total_budget * multiplier),
            )
            if self.hybrid_ultradeep_max_layer_simulations > 0:
                budget = min(budget, self.hybrid_ultradeep_max_layer_simulations)
        if self.hybrid_max_layer_simulations > 0:
            budget = min(budget, self.hybrid_max_layer_simulations)
        return budget

    def _hybrid_time_exceeded(self, start_time: float) -> bool:
        """Return whether the optional Hybrid wall-clock limit has been exceeded."""
        if self.hybrid_time_limit_seconds <= 0:
            return False
        return time.monotonic() - start_time >= self.hybrid_time_limit_seconds

    def _select_hybrid_beam(
        self,
        children: List[MCTSNode],
        tail_profile: bool,
        ultradeep_profile: bool = False,
    ) -> List[MCTSNode]:
        """Keep the best Hybrid child nodes for the next layer."""
        if ultradeep_profile:
            beam_width = self.hybrid_ultradeep_beam_width
        else:
            beam_width = self.hybrid_tail_beam_width if tail_profile else self.hybrid_beam_width
        ranked = sorted(children, key=self._node_selection_score, reverse=True)
        return ranked[:beam_width]

    def _best_node_from_beam(self, beam: List[MCTSNode]) -> Optional[MCTSNode]:
        """Return the best current Hybrid beam node."""
        if not beam:
            return None
        return max(beam, key=self._node_selection_score)

    def _node_selection_score(self, node: MCTSNode) -> float:
        """Score nodes for Hybrid beam retention."""
        if node.visits > 0:
            return node.average_reward
        return float("-inf")

    def _has_decisive_score_lead(
        self,
        children: List[MCTSNode],
        multiplier: float,
    ) -> bool:
        """Judge whether the best visited child has a clear average-reward lead."""
        visited_scores = [child.average_reward for child in children if child.visits > 0]
        if len(visited_scores) < 2:
            return False
        ordered = sorted(visited_scores, reverse=True)
        mean = sum(visited_scores) / len(visited_scores)
        variance = sum((score - mean) ** 2 for score in visited_scores) / len(visited_scores)
        std = math.sqrt(variance)
        return ordered[0] - ordered[1] > multiplier * std

    def _total_search_space_factor(self, usage: SegmentUsage) -> float:
        """Estimate the total combinational search space as a capped scale factor."""
        active_groups = [group for group in self.groups if not group.assigned]
        if not active_groups:
            return 0.0
        if self.basic_space_scale_divisor <= 0:
            return 1.0

        max_space_factor = max(1.0, self.basic_max_space_factor)
        log_total_space = 0.0
        for group in active_groups:
            branch_count = sum(
                1
                for segment in self.segment_manager.candidates_for_module(group.module_name)
                if usage.can_assign(segment, group.max_pin_width)
            )
            if branch_count <= 0:
                return 0.0
            log_total_space += math.log(branch_count)

        log_raw_factor = log_total_space - math.log(self.basic_space_scale_divisor)
        if log_raw_factor >= math.log(max_space_factor):
            return max_space_factor
        return math.exp(log_raw_factor)

    def _total_simulation_budget(self, usage: SegmentUsage) -> int:
        """根据搜索空间因子放大输入基准模拟次数。"""
        space_factor = self._search_space_factor(usage)
        scaled_budget = math.ceil(self.simulations * space_factor)
        return max(self.min_layer_simulations, scaled_budget)

    def _search_space_factor(self, usage: SegmentUsage) -> float:
        """估算搜索空间放大系数。"""
        active_groups = [group for group in self.groups if not group.assigned]
        if not active_groups or self.space_scale_divisor <= 0:
            return 0.0
        branch_counts = []
        for group in active_groups:
            count = sum(
                1
                for segment in self.segment_manager.candidates_for_module(group.module_name)
                if usage.can_assign(segment, group.max_pin_width)
            )
            branch_counts.append(count)
        average_branching = sum(branch_counts) / len(branch_counts) if branch_counts else 0.0
        factor = (average_branching ** self.typical_depth) / self.space_scale_divisor
        return min(factor, self.max_space_factor)

    def _layer_budget(self, total_budget: int, depth: int) -> int:
        """返回当前深度的一层局部竞争模拟次数。"""
        if depth <= self.tail_depth:
            multiplier = self.budget_decay ** (depth - 1)
        else:
            multiplier = (
                self.budget_decay ** (self.tail_depth - 1)
                * self.tail_decay ** (depth - self.tail_depth)
            )
        return max(self.min_layer_simulations, math.ceil(total_budget * multiplier))

    def _ensure_children(self, node: MCTSNode) -> List[MCTSNode]:
        """为当前层创建所有合法 action 子节点。"""
        if node.children:
            return node.children
        actions = list(self._available_actions(node))
        for action in actions:
            self._create_child(node, action)
        node.untried_actions = []
        return node.children

    def _create_child(self, node: MCTSNode, action: Action) -> MCTSNode:
        """按给定 action 创建一个子节点。"""
        group_name, segment_id = action
        group = self.groups[node.group_index]
        segment = self.segment_manager.abstract_segments[segment_id]

        usage = node.usage.clone()
        usage.assign(segment, group.max_pin_width)
        assignments = dict(node.assignments)
        assignments[group_name] = segment_id

        child = MCTSNode(
            group_index=node.group_index + 1,
            usage=usage,
            assignments=assignments,
            parent=node,
            action=action,
        )
        node.children.append(child)
        return child

    def _select_layer_child(self, children: List[MCTSNode]) -> MCTSNode:
        """在当前层选择一个 child 进行快速评估。"""
        unvisited = [child for child in children if child.visits == 0]
        if unvisited:
            return self.random.choice(unvisited)
        return max(children, key=lambda child: self._ucb(child))

    def _record_layer_result(
        self,
        parent: MCTSNode,
        child: MCTSNode,
        reward: float,
    ) -> None:
        """只更新当前层竞争统计。"""
        parent.visits += 1
        child.visits += 1
        child.total_reward += reward

    def _has_decisive_ucb_lead(self, children: List[MCTSNode]) -> bool:
        """判断当前层最佳 UCB 是否显著领先。"""
        if len(children) < 2:
            return False
        scores = [self._ucb(child) for child in children]
        if any(math.isinf(score) or math.isnan(score) for score in scores):
            return False
        ordered = sorted(scores, reverse=True)
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        std = math.sqrt(variance)
        return ordered[0] - ordered[1] > self.early_stop_std_multiplier * std

    def _select(self, node: MCTSNode) -> MCTSNode:
        """从根节点向下选择，直到遇到可扩展节点或叶节点。"""
        while node.group_index < len(self.groups):
            actions = self._available_actions(node)
            if actions:
                return node
            if not node.children:
                return node
            node = max(node.children, key=lambda child: self._ucb(child))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """从当前节点随机扩展一个尚未尝试的合法 action。"""
        actions = self._available_actions(node)
        if not actions:
            return node
        action = actions.pop(self.random.randrange(len(actions)))
        return self._create_child(node, action)

    def _available_actions(self, node: MCTSNode) -> List[Action]:
        """枚举当前层同构组可选择且满足容量的 segment action。"""
        if node.group_index >= len(self.groups):
            return []
        if node.untried_actions is not None:
            return node.untried_actions

        node.untried_actions = self._actions_for_node(node, tail_profile=False)
        return node.untried_actions

    def _actions_for_node(
        self,
        node: MCTSNode,
        tail_profile: bool = False,
    ) -> List[Action]:
        """Return capacity-feasible actions, optionally candidate-pruned."""
        group = self.groups[node.group_index]
        segments = self._candidate_segments(group, node.usage, tail_profile=tail_profile)
        return [(group.name, segment.segment_id) for segment in segments]

    def _raw_feasible_segments(
        self,
        group: PinHomologyGroup,
        usage: SegmentUsage,
    ) -> List[AbstractSegment]:
        """Return all capacity-feasible segments for one group."""
        return [
            segment
            for segment in self.segment_manager.candidates_for_module(group.module_name)
            if usage.can_assign(segment, group.max_pin_width)
        ]

    def _candidate_segments(
        self,
        group: PinHomologyGroup,
        usage: SegmentUsage,
        tail_profile: bool = False,
    ) -> List[AbstractSegment]:
        """Return feasible segments after optional conservative pruning."""
        feasible = self._raw_feasible_segments(group, usage)
        if not self.enable_candidate_pruning or self._basic_pruning_disabled():
            return feasible
        if len(feasible) < self.candidate_min_count:
            return feasible

        top_k = self.candidate_tail_top_k if tail_profile else self.candidate_top_k
        top_k = min(max(1, top_k), len(feasible))
        scored = sorted(
            ((self._candidate_pruning_score(group, segment), segment) for segment in feasible),
            key=lambda item: (item[0], item[1].remaining_capacity, item[1].segment_id),
            reverse=True,
        )
        best_score = scored[0][0]
        tolerance = abs(best_score) * self.candidate_score_tolerance
        kept = [
            segment
            for index, (score, segment) in enumerate(scored)
            if index < top_k or score >= best_score - tolerance
        ]
        return kept or feasible

    def _basic_pruning_disabled(self) -> bool:
        """Return whether the active basic/basic-like tree should bypass pruning."""
        return (
            self._active_basic_depth is not None
            and self._active_basic_depth <= self.basic_disable_pruning_depth_limit
        )

    def _candidate_pruning_score(
        self,
        group: PinHomologyGroup,
        segment: AbstractSegment,
    ) -> float:
        """Score one segment using a local HPWL-only heuristic for pruning."""
        key = (group.name, segment.segment_id)
        if key in self._candidate_score_cache:
            return self._candidate_score_cache[key]

        temporary_locations = {}
        for pin in group.pins:
            try:
                instance = self.segment_manager.get_instance(pin.parent_inst, segment.segment_id)
            except KeyError:
                continue
            temporary_locations[pin.full_name] = instance.midpoint

        related_pin_names = {pin.full_name for pin in group.pins}
        score = 0.0
        for net in self.nets:
            if not any(pin.full_name in related_pin_names for pin in net.pins):
                continue
            score -= net_hpwl(net, self.placedb, temporary_locations)
        score += 1e-6 * segment.remaining_capacity
        self._candidate_score_cache[key] = score
        return score

    def _simulate(self, node: MCTSNode) -> float:
        """从当前节点开始随机补全剩余分配并计算 reward。"""
        usage = node.usage.clone()
        assignments = dict(node.assignments)
        for index in range(node.group_index, len(self.groups)):
            group = self.groups[index]
            feasible = self._candidate_segments(group, usage)
            if not feasible:
                return -1.0e30
            segment = self.random.choice(feasible)
            usage.assign(segment, group.max_pin_width)
            assignments[group.name] = segment.segment_id

        temporary_locations = self._temporary_locations(assignments)
        return self.reward_evaluator.evaluate(temporary_locations)

    def _temporary_locations(self, assignments: Dict[str, str]) -> Dict[str, Point]:
        """把临时分配方案转换成 Pin 到 segment 中点坐标的映射。"""
        locations = {}
        for group in self.groups:
            segment_id = assignments.get(group.name)
            if segment_id is None:
                continue
            for pin in group.pins:
                try:
                    instance = self.segment_manager.get_instance(pin.parent_inst, segment_id)
                except KeyError:
                    continue
                locations[pin.full_name] = instance.midpoint
        return locations

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """把一次模拟的 reward 从叶节点回传到根节点。"""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def _ucb(self, child: MCTSNode) -> float:
        """计算 UCB 值，平衡已知收益和探索新分支。"""
        if child.visits == 0:
            return float("inf")
        parent_visits = max(child.parent.visits if child.parent else 1, 1)
        exploitation = child.total_reward / child.visits
        exploration = self.exploration_constant * math.sqrt(
            math.log(parent_visits) / child.visits
        )
        return exploitation + exploration

    def _best_child_by_reward(self, node: MCTSNode) -> Optional[MCTSNode]:
        """返回某节点下平均 reward 最高的子节点。"""
        if not node.children:
            return None
        return max(node.children, key=lambda child: child.average_reward)

    def _extract_best_path(self, start: MCTSNode) -> Dict[str, str]:
        """沿平均 reward 最优的子节点提取最终分配路径。"""
        node = start
        while node.children:
            node = max(node.children, key=lambda child: child.average_reward)
        if len(node.assignments) < len(self.groups):
            return self._complete_greedily(node.assignments, node.usage)
        return node.assignments

    def _complete_greedily(
        self,
        assignments: Dict[str, str],
        usage: SegmentUsage,
    ) -> Dict[str, str]:
        """当最佳路径未覆盖所有组时，用贪心方式补全剩余组。"""
        completed = dict(assignments)
        local_usage = usage.clone()
        assigned_names = set(completed)
        for group in self.groups:
            if group.assigned or group.name in assigned_names:
                continue
            feasible = self._candidate_segments(group, local_usage)
            if not feasible:
                continue
            segment = self._best_completion_segment_by_reward(group, feasible, completed)
            local_usage.assign(segment, group.max_pin_width)
            completed[group.name] = segment.segment_id
        return completed

    def _best_completion_segment_by_reward(
        self,
        group: PinHomologyGroup,
        feasible_segments: List[AbstractSegment],
        assignments: Dict[str, str],
    ) -> AbstractSegment:
        """Choose the greedy completion segment with the best current MCTS reward."""
        best_segment = None
        best_reward = float("-inf")
        for segment in feasible_segments:
            candidate_assignments = dict(assignments)
            candidate_assignments[group.name] = segment.segment_id
            reward = self.reward_evaluator.evaluate(
                self._temporary_locations(candidate_assignments)
            )
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
        return best_segment if best_segment is not None else max(
            feasible_segments,
            key=lambda item: item.remaining_capacity,
        )

    def _complete_fast_by_heuristic(
        self,
        assignments: Dict[str, str],
        usage: SegmentUsage,
    ) -> Dict[str, str]:
        """Complete very deep trees with a capacity-safe HPWL heuristic instead of FT reward."""
        completed = dict(assignments)
        local_usage = usage.clone()
        assigned_names = set(completed)
        for group in self.groups:
            if group.assigned or group.name in assigned_names:
                continue
            feasible = self._candidate_segments(group, local_usage, tail_profile=True)
            if not feasible:
                continue
            segment = max(
                feasible,
                key=lambda item: (
                    self._candidate_pruning_score(group, item),
                    item.remaining_capacity,
                    item.segment_id,
                ),
            )
            local_usage.assign(segment, group.max_pin_width)
            completed[group.name] = segment.segment_id
        return completed

    def _greedy_assignment(self, usage: SegmentUsage) -> Dict[str, str]:
        """在没有可用 MCTS 子节点时，直接生成贪心分配方案。"""
        return self._complete_greedily({}, usage)


def get_simulation_budget(group_count: int, base: int = 128) -> int:
    """根据当前树深度返回 MCTS 模拟次数。

    当前原型只做轻量动态调整，避免大数据集运行过慢。
    """
    return max(base, min(512, base + group_count * 16))
