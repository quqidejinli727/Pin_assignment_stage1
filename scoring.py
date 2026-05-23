"""Reward and metric functions for pin assignment."""

from __future__ import annotations

from typing import Dict, Iterable

from PlaceDB import Net, Pin, PlaceDB
from geometry_utils import Point, hpwl


def feedthrough_reward(_: Iterable[Net]) -> float:
    """feedthrough 指标预留接口，当前原型权重为 0。"""
    return 0.0


def net_hpwl(
    net: Net,
    placedb: PlaceDB,
    temporary_locations: Dict[str, Point],
) -> float:
    """计算单条 Net 在临时 Pin 坐标下的 HPWL。"""
    points = []
    for pin in net.pins:
        if pin.full_name in temporary_locations:
            points.append(temporary_locations[pin.full_name])
        else:
            points.append(placedb.get_pin_location_estimate(pin))
    return hpwl(points)


def assignment_reward(
    nets: Iterable[Net],
    placedb: PlaceDB,
    temporary_locations: Dict[str, Point],
    feedthrough_weight: float = 0.0,
) -> float:
    """综合 HPWL 和 feedthrough，返回 MCTS 使用的 reward。"""
    total_hpwl = sum(net_hpwl(net, placedb, temporary_locations) for net in nets)
    return -total_hpwl + feedthrough_weight * feedthrough_reward(nets)
