"""MCTS tree for one local ``pins_in`` assignment batch."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from PlaceDB import Net, PlaceDB
from geometry_utils import Point
from homology import PinHomologyGroup
from scoring import FeedthroughContext, RewardEvaluator
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
        basic_dynamic_simulations: bool = True,
        basic_space_scale_divisor: float = 1_000_000.0,
        basic_max_space_factor: float = 10.0,
        basic_min_simulations: int = 256,
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
        self.basic_dynamic_simulations = basic_dynamic_simulations
        self.basic_space_scale_divisor = basic_space_scale_divisor
        self.basic_max_space_factor = basic_max_space_factor
        self.basic_min_simulations = basic_min_simulations
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
            raise ValueError(
                f"Unsupported MCTS search mode: {self.search_mode!r}. "
                "Use 'layered' or 'basic'."
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

        for _ in range(self._basic_simulation_budget(root.usage)):
            node = self._select(root)
            if node.group_index < len(self.groups):
                node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

        best = self._best_child_by_reward(root)
        if best is None:
            return self._greedy_assignment(root.usage)
        return self._extract_best_path(best)

    def _basic_simulation_budget(self, usage: SegmentUsage) -> int:
        """Return the actual Basic-mode simulation count for this local tree."""
        if not self.basic_dynamic_simulations:
            return self.simulations
        space_factor = self._total_search_space_factor(usage)
        scaled_budget = math.ceil(self.simulations * space_factor)
        return max(self.basic_min_simulations, scaled_budget)

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

        group = self.groups[node.group_index]
        actions = []
        for segment in self.segment_manager.candidates_for_module(group.module_name):
            if node.usage.can_assign(segment, group.max_pin_width):
                actions.append((group.name, segment.segment_id))
        node.untried_actions = actions
        return node.untried_actions

    def _simulate(self, node: MCTSNode) -> float:
        """从当前节点开始随机补全剩余分配并计算 reward。"""
        usage = node.usage.clone()
        assignments = dict(node.assignments)
        for index in range(node.group_index, len(self.groups)):
            group = self.groups[index]
            feasible = [
                segment
                for segment in self.segment_manager.candidates_for_module(group.module_name)
                if usage.can_assign(segment, group.max_pin_width)
            ]
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
            feasible = [
                segment
                for segment in self.segment_manager.candidates_for_module(group.module_name)
                if local_usage.can_assign(segment, group.max_pin_width)
            ]
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

    def _greedy_assignment(self, usage: SegmentUsage) -> Dict[str, str]:
        """在没有可用 MCTS 子节点时，直接生成贪心分配方案。"""
        return self._complete_greedily({}, usage)


def get_simulation_budget(group_count: int, base: int = 128) -> int:
    """根据当前树深度返回 MCTS 模拟次数。

    当前原型只做轻量动态调整，避免大数据集运行过慢。
    """
    return max(base, min(512, base + group_count * 16))
