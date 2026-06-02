"""比较算法最终摆放结果与 benchmark case 的逐网指标。

两份输入文件均采用 pingroup JSON 格式：最外层数组中的每一项表示一条 net，
每个 Pin 的 ``scope`` 字段记录最终摆放位置：``[x, y]``，或表示带宽 Pin 两端的
``[[x1, y1], [x2, y2]]``。后者在分析时取两端中点作为 Pin 坐标。脚本按 Pin 全名集合
匹配两份文件中的同一条 net，计算 HPWL 差值并按区间统计分布。

若提供 ``--block``，脚本还会在两种摆放坐标上分别运行项目内已有的 ftpred
预测器，得到逐网 feedthrough。未提供 block 或已有 metrics 时，报告中的
feedthrough 字段记为 ``N/A``。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PlaceDB import PlaceDB
from feedthrough import ftpred_loader
from geometry_utils import Point, hpwl
from scoring import ensure_ftpred_executable


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_DIR = PROJECT_DIR / "mcts_result" / "analysis"
DEFAULT_FEEDTHROUGH_DIR = PROJECT_DIR / "feedthrough"

NetKey = Tuple[str, ...]


@dataclass
class PinReportInfo:
    """报告中展示的单个 Pin 属性。"""

    width: float | None
    reuse_count: int | None


@dataclass
class PlacementNet:
    """保存一份 pingroup 摆放中的单条 net 及其 Pin 坐标。"""

    key: NetKey
    net_id: int
    pins: Dict[str, Point]
    pin_info: Dict[str, PinReportInfo]

    @property
    def pin_names(self) -> List[str]:
        """返回排序后的 Pin 全名列表。"""
        return list(self.key)

    @property
    def hpwl(self) -> float:
        """依据所有 Pin 的 scope 坐标计算该 net 的 HPWL。"""
        return hpwl(list(self.pins.values()))


@dataclass
class NetComparison:
    """保存同一条 net 在 final_result 和 case 中的比较结果。"""

    key: NetKey
    final_net_id: int
    case_net_id: int
    final_hpwl: float
    case_hpwl: float
    delta_hpwl: float
    group: str
    pin_info: Dict[str, PinReportInfo]
    final_feedthrough: float | None = None
    case_feedthrough: float | None = None

    @property
    def delta_feedthrough(self) -> float | None:
        """返回 feedthrough 差值；不存在一侧指标时返回 None。"""
        if self.final_feedthrough is None or self.case_feedthrough is None:
            return None
        return self.final_feedthrough - self.case_feedthrough


def pin_full_name(pin_record: dict) -> str:
    """从 pingroup 记录构造与 PlaceDB 一致的 Pin 实例全名。"""
    try:
        return f"{pin_record['parent_inst']}.{pin_record['pingroup_name']}"
    except KeyError as error:
        raise ValueError(f"Pin 记录缺少字段: {error.args[0]}") from error


def pin_homology_name(pin_record: dict) -> str | None:
    """返回用于统计复用实例数的同构 Pin 分组名称。"""
    parent_module = pin_record.get("parent_module")
    pingroup_name = pin_record.get("pingroup_name")
    if isinstance(parent_module, str) and isinstance(pingroup_name, str):
        return f"{parent_module}.{pingroup_name}"
    return None


def pin_width(pin_record: dict) -> float | None:
    """读取 Pin 宽度；旧输入未提供时允许在报告中显示 N/A。"""
    width = pin_record.get("width")
    if isinstance(width, (int, float)):
        return float(width)
    return None


def is_empty_scope(scope: object) -> bool:
    """判断输入 scope 是否为空坐标。"""
    return scope is None or (
        isinstance(scope, (list, tuple)) and len(scope) == 0
    )


def scope_location(scope: object, context: str) -> Point:
    """读取 ``scope`` 坐标；带宽 Pin 的两端坐标按中点参与分析。"""
    if isinstance(scope, (list, tuple)) and len(scope) == 2:
        if all(isinstance(value, (int, float)) for value in scope):
            return (float(scope[0]), float(scope[1]))
        if all(
            isinstance(endpoint, (list, tuple))
            and len(endpoint) == 2
            and all(isinstance(value, (int, float)) for value in endpoint)
            for endpoint in scope
        ):
            return (
                (float(scope[0][0]) + float(scope[1][0])) / 2.0,
                (float(scope[0][1]) + float(scope[1][1])) / 2.0,
            )
    raise ValueError(
        f"{context} 的 scope 必须是最终摆放坐标 [x, y] 或两端坐标 "
        f"[[x1, y1], [x2, y2]]，实际值为 {scope!r}。"
    )


def pin_key_from_records(
    pin_records: Sequence[dict],
    input_path: Path,
    net_id: int,
) -> NetKey:
    """从一条 net 的 Pin 记录中构造用于匹配 final/case 的 Pin 集合 key。"""
    names = []
    seen = set()
    for pin_record in pin_records:
        if not isinstance(pin_record, dict):
            raise ValueError(f"{input_path} 中 net {net_id} 存在非对象 Pin 记录。")
        name = pin_full_name(pin_record)
        if name in seen:
            raise ValueError(f"{input_path} 中 net {net_id} 重复包含 Pin {name}。")
        names.append(name)
        seen.add(name)
    return tuple(sorted(names))


def load_pingroup_nets(
    path: str | Path,
    skip_keys: set[NetKey] | None = None,
) -> Dict[NetKey, PlacementNet]:
    """读取带有 scope 坐标的 pingroup JSON，并按 Pin 集合作为 net 标识。"""
    input_path = Path(path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{input_path} 的顶层结构必须是 net 数组。")

    reuse_counts = Counter(
        group_name
        for pin_records in data
        if isinstance(pin_records, list)
        for pin_record in pin_records
        if isinstance(pin_record, dict)
        for group_name in [pin_homology_name(pin_record)]
        if group_name is not None
    )
    skip_keys = skip_keys or set()
    nets: Dict[NetKey, PlacementNet] = {}
    for net_id, pin_records in enumerate(data):
        if not isinstance(pin_records, list):
            raise ValueError(f"{input_path} 中 net {net_id} 必须是 Pin 数组。")
        pins: Dict[str, Point] = {}
        pin_info: Dict[str, PinReportInfo] = {}
        key = pin_key_from_records(pin_records, input_path, net_id)
        if key in skip_keys:
            continue
        for pin_record in pin_records:
            name = pin_full_name(pin_record)
            scope = pin_record.get("scope")
            if len(pin_records) == 1 and is_empty_scope(scope):
                pins[name] = (0.0, 0.0)
            else:
                pins[name] = scope_location(
                    scope,
                    f"{input_path} 中 net {net_id} 的 Pin {name}",
                )
            homology_name = pin_homology_name(pin_record)
            pin_info[name] = PinReportInfo(
                width=pin_width(pin_record),
                reuse_count=(
                    reuse_counts[homology_name] if homology_name is not None else None
                ),
            )

        if key in nets:
            raise ValueError(
                f"{input_path} 存在两条具有相同 Pin 集合的 net，无法唯一匹配: {key}"
            )
        nets[key] = PlacementNet(key=key, net_id=net_id, pins=pins, pin_info=pin_info)
    return nets


def load_analysis_nets(
    final_path: str | Path,
    case_path: str | Path,
) -> tuple[Dict[NetKey, PlacementNet], Dict[NetKey, PlacementNet], int]:
    """加载 final/case；case 多 Pin net 的空 scope 会驱动两侧同步跳过。"""
    case_input_path = Path(case_path)
    data = json.loads(case_input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{case_input_path} 的顶层结构必须是 net 数组。")

    skipped_keys: set[NetKey] = set()
    skipped_pin_count = 0
    for net_id, pin_records in enumerate(data):
        if not isinstance(pin_records, list):
            raise ValueError(f"{case_input_path} 中 net {net_id} 必须是 Pin 数组。")
        if len(pin_records) > 1 and any(
            isinstance(pin_record, dict) and is_empty_scope(pin_record.get("scope"))
            for pin_record in pin_records
        ):
            key = pin_key_from_records(pin_records, case_input_path, net_id)
            skipped_keys.add(key)
            skipped_pin_count += len(pin_records)

    final_nets = load_pingroup_nets(final_path, skip_keys=skipped_keys)
    case_nets = load_pingroup_nets(case_path, skip_keys=skipped_keys)
    return final_nets, case_nets, skipped_pin_count


def delta_group(delta: float, width: float) -> str:
    """按 HPWL 差值分组；负值表示 final_result 比 case 更优。"""
    if width <= 0:
        raise ValueError("group_width 必须大于 0。")
    if math.isclose(delta, 0.0, abs_tol=1e-9):
        return "= 0"
    if delta > 0:
        lower = math.floor(delta / width) * width
        return f"[{_number(lower)}, {_number(lower + width)})"
    index = max(1, math.ceil(abs(delta) / width - 1e-12))
    lower = -index * width
    upper = -(index - 1) * width
    return f"[{_number(lower)}, {_number(upper)})"


def _number(value: float) -> str:
    """使用紧凑格式显示分组边界。"""
    return f"{value:g}"


def load_feedthrough_metrics(path: str | Path | None) -> Dict[int, float]:
    """从已有 metrics.json 读取 ``net_id -> feedthrough`` 映射。"""
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(item["net_id"]): float(item["feedthrough"])
        for item in data.get("net_metrics", [])
    }


def _prepare_predictor_db(
    pingroup_path: str | Path,
    placement_nets: Dict[NetKey, PlacementNet],
    block_path: str | Path,
) -> tuple[PlaceDB, dict[NetKey, object]]:
    """加载一份摆放，并把其 scope 坐标写入预测器使用的 PlaceDB。"""
    db = PlaceDB(str(block_path), str(pingroup_path))
    db_net_by_key = {
        tuple(sorted(pin.full_name for pin in net.pins)): net for net in db.nets_list
    }
    missing_keys = set(placement_nets) - set(db_net_by_key)
    if missing_keys:
        raise ValueError(
            f"PlaceDB 解析出的 net 集合缺少分析输入中的 net: {len(missing_keys)}。"
        )

    for key, placement_net in placement_nets.items():
        for pin_name, location in placement_net.pins.items():
            pin = db.pin_dict.get(pin_name)
            if pin is None:
                raise ValueError(f"预测器输入中无法找到 Pin: {pin_name}")
            pin.x, pin.y = location
            pin.assigned_segment_coord = location
    return db, db_net_by_key


def predictor_feedthrough_for_placements(
    final_path: str | Path,
    final_nets: Dict[NetKey, PlacementNet],
    case_path: str | Path,
    case_nets: Dict[NetKey, PlacementNet],
    block_path: str | Path,
    feedthrough_dir: str | Path,
    auto_build: bool,
    cmake_generator: str | None,
) -> tuple[Dict[NetKey, float], Dict[NetKey, float]]:
    """只启动一次预测器，依次计算 final_result 与 case 的逐网值。"""
    final_db, final_db_nets = _prepare_predictor_db(final_path, final_nets, block_path)
    case_db, case_db_nets = _prepare_predictor_db(case_path, case_nets, block_path)

    executable = ensure_ftpred_executable(
        Path(feedthrough_dir),
        auto_build=auto_build,
        cmake_generator=cmake_generator,
    )
    modules_text = ftpred_loader.build_modules_text(final_db)
    final_values: Dict[NetKey, float] = {}
    case_values: Dict[NetKey, float] = {}
    with ftpred_loader.FtpredBinSession(str(executable), modules_text) as session:
        for key in sorted(final_nets):
            with redirect_stdout(StringIO()):
                final_values[key] = float(session.run_one_net(final_db, final_db_nets[key]))
        for key in sorted(case_nets):
            with redirect_stdout(StringIO()):
                case_values[key] = float(session.run_one_net(case_db, case_db_nets[key]))
    return final_values, case_values


def compare_nets(
    final_nets: Dict[NetKey, PlacementNet],
    case_nets: Dict[NetKey, PlacementNet],
    group_width: float,
    final_feedthrough_by_id: Dict[int, float] | None = None,
    case_feedthrough_by_id: Dict[int, float] | None = None,
    final_feedthrough_by_key: Dict[NetKey, float] | None = None,
    case_feedthrough_by_key: Dict[NetKey, float] | None = None,
) -> List[NetComparison]:
    """匹配两套摆放中的 net，并生成逐网指标比较记录。"""
    final_keys = set(final_nets)
    case_keys = set(case_nets)
    if final_keys != case_keys:
        raise ValueError(
            f"两份摆放的 net 集合不一致: final_only={len(final_keys - case_keys)}, "
            f"case_only={len(case_keys - final_keys)}。"
        )

    final_feedthrough_by_id = final_feedthrough_by_id or {}
    case_feedthrough_by_id = case_feedthrough_by_id or {}
    final_feedthrough_by_key = final_feedthrough_by_key or {}
    case_feedthrough_by_key = case_feedthrough_by_key or {}
    comparisons = []
    for key in sorted(final_keys):
        final_net = final_nets[key]
        case_net = case_nets[key]
        difference = final_net.hpwl - case_net.hpwl
        comparisons.append(
            NetComparison(
                key=key,
                final_net_id=final_net.net_id,
                case_net_id=case_net.net_id,
                final_hpwl=final_net.hpwl,
                case_hpwl=case_net.hpwl,
                delta_hpwl=difference,
                group=delta_group(difference, group_width),
                pin_info=final_net.pin_info,
                final_feedthrough=final_feedthrough_by_key.get(
                    key, final_feedthrough_by_id.get(final_net.net_id)
                ),
                case_feedthrough=case_feedthrough_by_key.get(
                    key, case_feedthrough_by_id.get(case_net.net_id)
                ),
            )
        )
    return sorted(comparisons, key=lambda item: item.final_net_id)


def metric_summary(comparisons: Sequence[NetComparison]) -> dict:
    """汇总 HPWL 与可用的 feedthrough 整体指标。"""
    count = len(comparisons)
    final_hpwl = sum(item.final_hpwl for item in comparisons)
    case_hpwl = sum(item.case_hpwl for item in comparisons)
    with_feedthrough = [
        item for item in comparisons if item.delta_feedthrough is not None
    ]
    return {
        "net_count": count,
        "optimized_net_count": sum(1 for item in comparisons if item.delta_hpwl < -1e-9),
        "worsened_net_count": sum(1 for item in comparisons if item.delta_hpwl > 1e-9),
        "unchanged_net_count": sum(
            1 for item in comparisons if math.isclose(item.delta_hpwl, 0.0, abs_tol=1e-9)
        ),
        "final_total_hpwl": final_hpwl,
        "case_total_hpwl": case_hpwl,
        "delta_total_hpwl": final_hpwl - case_hpwl,
        "final_average_hpwl": final_hpwl / count if count else 0.0,
        "case_average_hpwl": case_hpwl / count if count else 0.0,
        "feedthrough_net_count": len(with_feedthrough),
        "final_total_feedthrough": sum(item.final_feedthrough for item in with_feedthrough),
        "case_total_feedthrough": sum(item.case_feedthrough for item in with_feedthrough),
        "final_average_feedthrough": (
            sum(item.final_feedthrough for item in with_feedthrough) / len(with_feedthrough)
            if with_feedthrough else None
        ),
        "case_average_feedthrough": (
            sum(item.case_feedthrough for item in with_feedthrough) / len(with_feedthrough)
            if with_feedthrough else None
        ),
    }


def write_analysis_report(
    output_path: str | Path,
    final_path: str | Path,
    case_path: str | Path,
    group_width: float,
    comparisons: Sequence[NetComparison],
    skipped_case_pin_count: int = 0,
) -> None:
    """输出可读 TXT 报告，包含分组分布与每条 net 的明细。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = metric_summary(comparisons)
    distribution = Counter(item.group for item in comparisons)
    lines = [
        "Final Result vs Benchmark Case Analysis",
        "=" * 72,
        f"final_result: {Path(final_path)}",
        f"case:         {Path(case_path)}",
        "input_format: pingroup JSON; Pin scope is [x, y] or endpoint pair "
        "[[x1, y1], [x2, y2]] (analyzed at midpoint)",
        "HPWL delta definition: final_result - case (negative means optimized)",
        f"HPWL group width: {group_width:g}",
        "",
        "Summary",
        "-" * 72,
        f"net_count: {summary['net_count']}",
        f"optimized_net_count: {summary['optimized_net_count']}",
        f"worsened_net_count: {summary['worsened_net_count']}",
        f"unchanged_net_count: {summary['unchanged_net_count']}",
        f"final_total_hpwl: {summary['final_total_hpwl']:.6f}",
        f"case_total_hpwl: {summary['case_total_hpwl']:.6f}",
        f"delta_total_hpwl: {summary['delta_total_hpwl']:.6f}",
        f"final_average_hpwl: {summary['final_average_hpwl']:.6f}",
        f"case_average_hpwl: {summary['case_average_hpwl']:.6f}",
        f"skipped_case_pin_count: {skipped_case_pin_count}",
        f"feedthrough_net_count: {summary['feedthrough_net_count']}",
        f"final_total_feedthrough: {_metric(summary['final_total_feedthrough'], summary['feedthrough_net_count'])}",
        f"case_total_feedthrough: {_metric(summary['case_total_feedthrough'], summary['feedthrough_net_count'])}",
        f"final_average_feedthrough: {_optional(summary['final_average_feedthrough'])}",
        f"case_average_feedthrough: {_optional(summary['case_average_feedthrough'])}",
        "",
        "HPWL Delta Group Distribution",
        "-" * 72,
    ]
    for group, count in sorted(distribution.items(), key=lambda pair: _group_sort_key(pair[0])):
        lines.append(f"{group}: {count}")

    lines.extend(["", "Per-Net Details", "-" * 72])
    for item in comparisons:
        lines.extend(
            [
                f"Net final_id={item.final_net_id}, case_id={item.case_net_id}, group={item.group}",
                f"  HPWL: final={item.final_hpwl:.6f}, case={item.case_hpwl:.6f}, delta={item.delta_hpwl:.6f}",
                f"  Feedthrough: final={_optional(item.final_feedthrough)}, "
                f"case={_optional(item.case_feedthrough)}, delta={_optional(item.delta_feedthrough)}",
                "  Pins:",
            ]
        )
        for pin_name in item.key:
            info = item.pin_info[pin_name]
            lines.append(
                f"    {pin_name}  width={_optional_number(info.width)}  "
                f"reuse_count={_optional_integer(info.reuse_count)}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selected_groups(
    output_path: str | Path,
    comparisons: Sequence[NetComparison],
    selected_groups: Iterable[str],
) -> None:
    """按指定差值组输出对应 net 编号与 Pin 全名。"""
    selected = set(selected_groups)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Selected HPWL Delta Groups", "=" * 72]
    for group in selected:
        matches = [item for item in comparisons if item.group == group]
        lines.append(f"\nGroup {group}: {len(matches)} nets")
        for item in matches:
            lines.append(
                f"  final_net_id={item.final_net_id}, case_net_id={item.case_net_id}, "
                f"delta_hpwl={item.delta_hpwl:.6f}"
            )
            lines.extend(f"    {pin_name}" for pin_name in item.key)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _optional(value: float | None) -> str:
    """格式化可选指标值。"""
    return "N/A" if value is None else f"{value:.6f}"


def _optional_number(value: float | None) -> str:
    """以紧凑格式显示可选 Pin 宽度。"""
    return "N/A" if value is None else f"{value:g}"


def _optional_integer(value: int | None) -> str:
    """显示可选复用次数。"""
    return "N/A" if value is None else str(value)


def _metric(value: float, available_count: int) -> str:
    """无 feedthrough 数据时在汇总处输出 N/A。"""
    return "N/A" if available_count == 0 else f"{value:.6f}"


def _group_sort_key(group: str) -> float:
    """按分组区间下界排序。"""
    if group == "= 0":
        return 0.0
    return float(group.split(",", 1)[0].strip("["))


def parse_args() -> argparse.Namespace:
    """解析输入输出路径、差值分组和可选 feedthrough 参数。"""
    parser = argparse.ArgumentParser(
        description="Compare final_result and benchmark case in pingroup scope format."
    )
    parser.add_argument(
        "--final",
        required=True,
        help="最终摆放 pingroup JSON（scope=[x,y] 或 [[x1,y1],[x2,y2]]）。",
    )
    parser.add_argument(
        "--case",
        required=True,
        help="benchmark 摆放 pingroup JSON（scope=[x,y] 或 [[x1,y1],[x2,y2]]）。",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--group-width", type=float, default=300.0)
    parser.add_argument(
        "--select-group",
        action="append",
        default=[],
        help='输出指定差值组的 net；可重复指定，例如 "[-300, 0)"。',
    )
    parser.add_argument("--final-metrics", default=None, help="可选 final metrics.json。")
    parser.add_argument("--case-metrics", default=None, help="可选 benchmark metrics.json。")
    parser.add_argument(
        "--block",
        default=None,
        help="可选 block JSON；提供后会以两份 scope 坐标重新计算 feedthrough。",
    )
    parser.add_argument("--feedthrough-dir", default=str(DEFAULT_FEEDTHROUGH_DIR))
    parser.add_argument("--no-auto-build", action="store_true")
    parser.add_argument("--cmake-generator", default="MinGW Makefiles")
    return parser.parse_args()


def main() -> None:
    """运行摆放比较并生成主报告及可选分组清单。"""
    args = parse_args()
    final_nets, case_nets, skipped_case_pin_count = load_analysis_nets(
        args.final,
        args.case,
    )

    final_by_key: Dict[NetKey, float] = {}
    case_by_key: Dict[NetKey, float] = {}
    if args.block:
        final_by_key, case_by_key = predictor_feedthrough_for_placements(
            args.final,
            final_nets,
            args.case,
            case_nets,
            args.block,
            args.feedthrough_dir,
            auto_build=not args.no_auto_build,
            cmake_generator=args.cmake_generator,
        )

    comparisons = compare_nets(
        final_nets,
        case_nets,
        args.group_width,
        final_feedthrough_by_id=load_feedthrough_metrics(args.final_metrics),
        case_feedthrough_by_id=load_feedthrough_metrics(args.case_metrics),
        final_feedthrough_by_key=final_by_key,
        case_feedthrough_by_key=case_by_key,
    )
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_path = output_dir / f"comparison_{timestamp}.txt"
    write_analysis_report(
        report_path,
        args.final,
        args.case,
        args.group_width,
        comparisons,
        skipped_case_pin_count=skipped_case_pin_count,
    )
    print(f"Analysis report: {report_path}")

    if args.select_group:
        selected_path = output_dir / f"selected_groups_{timestamp}.txt"
        write_selected_groups(selected_path, comparisons, args.select_group)
        print(f"Selected groups: {selected_path}")


if __name__ == "__main__":
    main()
