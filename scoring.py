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
from typing import Dict, Iterable, List

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


def assignment_reward(
    nets: Iterable[Net],
    placedb: PlaceDB,
    temporary_locations: Dict[str, Point],
    feedthrough_weight: float = 0.0,
) -> float:
    """综合 HPWL 和 feedthrough，返回 MCTS 使用的 reward。"""
    total_hpwl = sum(net_hpwl(net, placedb, temporary_locations) for net in nets)
    return -total_hpwl + feedthrough_weight * feedthrough_reward(nets)


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
    auto_build: bool = True,
    cmake_generator: str | None = None,
) -> Path:
    """查找预测器可执行文件；不存在时仅首次按需调用 CMake 编译。"""
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
    auto_build_feedthrough: bool = True,
    cmake_generator: str | None = None,
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
