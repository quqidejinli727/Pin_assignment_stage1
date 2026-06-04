"""Reward、HPWL 和 feedthrough 最终评估函数。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PlaceDB import Net, PlaceDB
from feedthrough import ftpred_loader
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
    ):
        self.placedb = placedb
        executable = ensure_ftpred_executable(
            feedthrough_source_dir,
            auto_build=auto_build_feedthrough,
            cmake_generator=cmake_generator,
        )
        modules_text = ftpred_loader.build_modules_text(placedb)
        self.session = ftpred_loader.FtpredBinSession(str(executable), modules_text)

    def close(self) -> None:
        """Close the long-lived predictor session."""
        if self.session is not None:
            self.session.close()
            self.session = None

    def run_one_net_at_locations(self, net: Net, locations: Dict[str, Point]) -> float:
        """Evaluate one net after temporarily applying the given pin locations."""
        if self.session is None:
            raise RuntimeError("FeedthroughContext has already been closed.")

        old_pin_locations = [(pin, pin.x, pin.y) for pin in net.pins]
        old_net_feedthrough = getattr(net, "feedthrough", 0.0)
        try:
            for pin in net.pins:
                pin.x, pin.y = locations[pin.full_name]
            with redirect_stdout(StringIO()):
                return float(self.session.run_one_net(self.placedb, net))
        finally:
            for pin, x, y in old_pin_locations:
                pin.x = x
                pin.y = y
            net.feedthrough = old_net_feedthrough


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
    ):
        self.nets = list(nets)
        self.placedb = placedb
        self.wirelength_weight = wirelength_weight
        self.feedthrough_weight = feedthrough_weight
        self.enable_feedthrough = enable_feedthrough and feedthrough_weight != 0.0
        self.normalization_floor = normalization_floor
        self.reward_scale = reward_scale
        self.feedthrough_context = feedthrough_context
        self._candidate_feedthrough_cache: Dict[Tuple[int, Tuple[Point, ...]], float] = {}

        if self.enable_feedthrough and self.feedthrough_context is None:
            raise ValueError("feedthrough_context is required when feedthrough reward is enabled.")

        self.reference_metrics = {
            id(net): self._build_reference_metrics(net)
            for net in self.nets
        }

    def close(self) -> None:
        """Release local evaluator caches without closing the shared predictor."""
        self._candidate_feedthrough_cache.clear()

    def evaluate(self, temporary_locations: Dict[str, Point]) -> float:
        """Return weighted normalized reward for a complete candidate assignment."""
        total_reward = 0.0
        for net in self.nets:
            reference = self.reference_metrics[id(net)]
            candidate_hpwl = net_hpwl(net, self.placedb, temporary_locations)
            wirelength_reward = self._normalized_improvement(reference.hpwl, candidate_hpwl)

            feedthrough_reward_value = 0.0
            if self.enable_feedthrough:
                candidate_feedthrough = self._candidate_feedthrough(net, temporary_locations)
                feedthrough_reward_value = self._normalized_improvement(
                    reference.feedthrough,
                    candidate_feedthrough,
                )

            total_reward += (
                self.wirelength_weight * wirelength_reward
                + self.feedthrough_weight * feedthrough_reward_value
            )
        return total_reward * self.reward_scale

    def _build_reference_metrics(self, net: Net) -> NetReferenceMetrics:
        reference_locations = {
            pin.full_name: self.placedb.get_module(pin.parent_inst).get_centroid()
            for pin in net.pins
        }
        reference_hpwl = net_hpwl(net, self.placedb, reference_locations)
        reference_feedthrough = (
            self._feedthrough_at_locations(net, reference_locations)
            if self.enable_feedthrough
            else 0.0
        )
        return NetReferenceMetrics(reference_hpwl, reference_feedthrough)

    def _candidate_feedthrough(
        self,
        net: Net,
        temporary_locations: Dict[str, Point],
    ) -> float:
        locations = {
            pin.full_name: temporary_locations.get(
                pin.full_name,
                self.placedb.get_pin_location_estimate(pin),
            )
            for pin in net.pins
        }
        key = (id(net), tuple(self._rounded_point(locations[pin.full_name]) for pin in net.pins))
        if key not in self._candidate_feedthrough_cache:
            self._candidate_feedthrough_cache[key] = self._feedthrough_at_locations(net, locations)
        return self._candidate_feedthrough_cache[key]

    def _feedthrough_at_locations(self, net: Net, locations: Dict[str, Point]) -> float:
        if self.feedthrough_context is None:
            return 0.0
        return self.feedthrough_context.run_one_net_at_locations(net, locations)

    def _normalized_improvement(self, reference: float, candidate: float) -> float:
        denominator = max(abs(reference), self.normalization_floor)
        return (reference - candidate) / denominator

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
    hpwl: float
    feedthrough: float


def _predictor_candidates(source_dir: Path) -> List[Path]:
    """返回不同平台和 CMake generator 可能生成的可执行文件路径。"""
    name = "ftpred.exe" if os.name == "nt" else "ftpred"
    return [
        source_dir / "build" / "Release" / name,
        source_dir / "build" / name,
    ]


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
    for candidate in _predictor_candidates(source_dir):
        if candidate.exists():
            return candidate

    if not auto_build:
        raise FileNotFoundError(
            f"未找到 ftpred 可执行文件，请先在 {source_dir} 下执行 CMake 编译。"
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
            hpwl=net_hpwl(net, placedb, {}),
            feedthrough=0.0,
        )
        for net in placedb.nets_list
    ]
    if not enable_feedthrough:
        return metrics

    if feedthrough_context is not None:
        for metric, net in zip(metrics, placedb.nets_list):
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
    modules_text = ftpred_loader.build_modules_text(placedb)
    with ftpred_loader.FtpredBinSession(str(executable), modules_text) as session:
        for metric, net in zip(metrics, placedb.nets_list):
            # loader 会为每条 net 打印解析明细；最终报告已统一记录指标，
            # 这里收起内部进度输出，保持主程序输出简洁。
            with redirect_stdout(StringIO()):
                metric.feedthrough = float(session.run_one_net(placedb, net))
    return metrics


def summarize_metrics(metrics: List[NetMetrics]) -> Dict[str, float | int]:
    """汇总所有 net 指标，生成数量、总值和平均值。"""
    count = len(metrics)
    total_hpwl = sum(metric.hpwl for metric in metrics)
    total_feedthrough = sum(metric.feedthrough for metric in metrics)
    return {
        "net_count": count,
        "total_hpwl": total_hpwl,
        "average_hpwl": total_hpwl / count if count else 0.0,
        "total_feedthrough": total_feedthrough,
        "average_feedthrough": total_feedthrough / count if count else 0.0,
    }


def metrics_to_records(metrics: List[NetMetrics]) -> List[dict]:
    """将指标对象转换为 JSON 可序列化列表。"""
    return [asdict(metric) for metric in metrics]
