"""将内部 segment 分配结果导出为外部接口要求的 JSON 格式。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from PlaceDB import Pin, PlaceDB
from homology import HomologyManager
from segment import AbstractSegment, SegmentManager
from segment_subdivision import percentile_edge_length


def _assigned_segment_keys(segments: SegmentManager) -> list[str]:
    """返回至少包含一个已分配 Pin 实例的抽象 segment ID。"""
    return [
        segment_id
        for segment_id, segment in sorted(segments.abstract_segments.items())
        if any(instance.assigned_pins for instance in segment.instances.values())
    ]


def _stable_ids(
    placedb: PlaceDB,
    homology: HomologyManager,
    segments: SegmentManager,
    segment_keys: list[str],
) -> dict:
    """为接口字段创建稳定的 block、pin、group、segment 整数 ID。"""
    full_names = [
        pin.full_name
        for net in placedb.nets_list
        for pin in net.pins
    ]
    if len(full_names) != len(set(full_names)):
        duplicates = sorted({name for name in full_names if full_names.count(name) > 1})
        raise ValueError(f"接口导出要求 Pin 实例名唯一，发现重复 Pin: {duplicates[:5]}")

    segment_inst_keys = sorted(
        (module_inst, segment_id)
        for segment_id in segment_keys
        for module_inst in segments.abstract_segments[segment_id].instances
    )
    return {
        "block": {
            module.name: index for index, module in enumerate(placedb.all_modules_list)
        },
        "pin": {name: index for index, name in enumerate(sorted(full_names))},
        "group": {
            name: index for index, name in enumerate(sorted(homology.pin_groups))
        },
        "segment": {key: index for index, key in enumerate(segment_keys)},
        "segment_inst": {
            key: index for index, key in enumerate(segment_inst_keys)
        },
    }


def _pin_record(
    pin: Pin,
    ids: dict,
    net_by_pin: Dict[str, int],
) -> dict:
    """转换单个实际 Pin 的接口字段。"""
    return {
        "id": ids["pin"][pin.full_name],
        "name": pin.full_name,
        "block_id": ids["block"][pin.parent_inst],
        "width": pin.width,
        "net_id": net_by_pin[pin.full_name],
        "isomorphic_group_id": ids["group"][pin.homology_name],
    }


def _segment_info(
    segment_id: int,
    block_id: int,
    segment: AbstractSegment,
    instance,
) -> dict:
    """转换 segment 或 segment_inst 共用的几何描述字段。"""
    return {
        "id": segment_id,
        "edge_id": segment.edge_id,
        "block_id": block_id,
        "x1": instance.start[0],
        "y1": instance.start[1],
        "x2": instance.end[0],
        "y2": instance.end[1],
        "length": segment.capacity,
        "max_capacity": segment.capacity,
        "direction": _segment_direction(instance),
        "cx": instance.midpoint[0],
        "cy": instance.midpoint[1],
        "midpoint": list(instance.midpoint),
    }


def _segment_direction(instance, tolerance: float = 1e-9) -> int:
    """按接口约定返回 segment 方向：水平为 1，竖直为 0。"""
    dx = abs(instance.end[0] - instance.start[0])
    dy = abs(instance.end[1] - instance.start[1])
    if dy <= tolerance and dx > tolerance:
        return 1
    if dx <= tolerance and dy > tolerance:
        return 0
    raise ValueError(
        "接口 direction 仅支持水平或竖直 segment，"
        f"发现斜向/零长度线段: start={instance.start}, end={instance.end}。"
    )


def build_interface_result(
    placedb: PlaceDB,
    homology: HomologyManager,
    segments: SegmentManager,
) -> dict:
    """从求解器内存对象构建接口格式的最终分配结果。"""
    segment_keys = _assigned_segment_keys(segments)
    ids = _stable_ids(placedb, homology, segments, segment_keys)
    pin_objects = {
        pin.full_name: pin for net in placedb.nets_list for pin in net.pins
    }
    net_by_pin = {
        pin.full_name: net.net_id
        for net in placedb.nets_list
        for pin in net.pins
    }
    output_segments = {}

    for internal_id in segment_keys:
        segment = segments.abstract_segments[internal_id]
        external_id = ids["segment"][internal_id]
        reference_module_inst = sorted(segment.instances)[0]
        reference_instance = segment.instances[reference_module_inst]
        reference_module = placedb.get_module(reference_module_inst)
        reference_block_id = ids["block"][reference_module_inst]
        info = _segment_info(
            external_id,
            reference_block_id,
            segment,
            reference_instance,
        )
        instance_records = {}
        total_used = 0.0

        for module_inst in sorted(segment.instances):
            instance = segment.instances[module_inst]
            module = placedb.get_module(module_inst)
            block_id = ids["block"][module_inst]
            assigned_pins = [
                _pin_record(pin_objects[pin_name], ids, net_by_pin)
                for pin_name in instance.assigned_pins
            ]
            used_capacity = sum(pin["width"] for pin in assigned_pins)
            total_used += used_capacity
            instance_records[str(block_id)] = {
                "segment_inst_id": ids["segment_inst"][(module_inst, internal_id)],
                "segment_id": external_id,
                "segment_info": _segment_info(
                    external_id,
                    reference_block_id,
                    segment,
                    reference_instance,
                ),
                "block_id": block_id,
                "block_name": module_inst,
                "assigned_pins": assigned_pins,
                "used_capacity": used_capacity,
                "remaining_capacity": segment.capacity - used_capacity,
                "coordinates": [
                    instance.start[0],
                    instance.start[1],
                    instance.end[0],
                    instance.end[1],
                ],
                "center_point": list(instance.midpoint),
                "direction": _segment_direction(instance),
                "edge_id": segment.edge_id,
                "max_capacity": segment.capacity,
                "length": segment.capacity,
            }

        total_capacity = segment.capacity * len(segment.instances)
        output_segments[str(external_id)] = {
            "segment_id": external_id,
            "segment_info": info,
            "segment_insts": instance_records,
            "total_used_capacity": total_used,
            "total_remaining_capacity": total_capacity - total_used,
            "direction": _segment_direction(reference_instance),
            "edge_id": segment.edge_id,
            "max_capacity": segment.capacity,
            "length": segment.capacity,
        }

    return {
        "total_segments": len(output_segments),
        "segment_assignments": output_segments,
    }


def write_interface_result(
    output_dir: str | Path,
    placedb: PlaceDB,
    homology: HomologyManager,
    segments: SegmentManager,
    timestamp: str | None = None,
) -> Path:
    """生成带时间戳的接口格式 JSON 文件并返回路径。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d%H%M%S%f")
    path = directory / f"segment_assignments_{timestamp}.json"
    path.write_text(
        json.dumps(build_interface_result(placedb, homology, segments), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_config_record(
    output_dir: str | Path,
    config_record: dict,
    timestamp: str,
    result_path: str | Path | None = None,
) -> Path:
    """Write a timestamp-matched config record for one Stage1 interface result."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"stage1_config_{timestamp}.json"
    record = dict(config_record)
    if result_path is not None:
        record["interface_result_path"] = str(result_path)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_existing_assignment(
    assignment_path: str | Path,
    block_path: str | Path,
    pingroup_path: str | Path,
    output_dir: str | Path,
    enable_segment_subdivision: bool = False,
    segment_length_percentile: int = 50,
) -> Path:
    """读取已有内部 outputs JSON，并转换为接口格式而无需重新搜索。"""
    from segment import SegmentManager

    assignment = json.loads(Path(assignment_path).read_text(encoding="utf-8"))
    placedb = PlaceDB(str(block_path), str(pingroup_path))
    homology = HomologyManager(placedb)
    max_length = (
        percentile_edge_length(block_path, segment_length_percentile)
        if enable_segment_subdivision
        else None
    )
    segments = SegmentManager(placedb, max_segment_length=max_length)

    for internal_id, segment_data in assignment.get("segments", {}).items():
        if internal_id not in segments.abstract_segments:
            raise ValueError(
                "已有 outputs 与当前 segment 构造不一致；"
                "如启用了裁剪，请使用同一配置重新运行主程序后导出。"
            )
        abstract = segments.abstract_segments[internal_id]
        abstract.used_width = float(segment_data.get("used_width", 0.0))
        abstract.assigned_groups = list(segment_data.get("assigned_groups", []))
        for module_inst, instance_data in segment_data.get("segment_instances", {}).items():
            instance = segments.get_instance(module_inst, internal_id)
            instance.assigned_pins = list(instance_data.get("assigned_pins", []))
            for pin_name in instance.assigned_pins:
                pin = placedb.pin_dict.get(pin_name)
                if pin is not None:
                    pin.assigned_segment_id = internal_id
                    pin.assigned_segment_coord = instance.midpoint
                    pin.segment_endpoints = (instance.start, instance.end)
    return write_interface_result(output_dir, placedb, homology, segments)


def main() -> None:
    """独立运行一次求解并导出接口结果，用于手动生成 final_result。"""
    from assignment_solver import AssignmentSolver

    parser = argparse.ArgumentParser(description="Export interface-format segment assignment result.")
    parser.add_argument("--block", required=True)
    parser.add_argument("--pingroup", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assignment", default=None, help="Convert an existing internal outputs JSON.")
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--disable-subdivision", action="store_true")
    parser.add_argument("--segment-percentile", type=int, default=50)
    args = parser.parse_args()

    if args.assignment:
        print(
            export_existing_assignment(
                args.assignment,
                args.block,
                args.pingroup,
                args.output_dir,
                enable_segment_subdivision=not args.disable_subdivision,
                segment_length_percentile=args.segment_percentile,
            )
        )
        return

    solver = AssignmentSolver(
        args.block,
        args.pingroup,
        simulations=args.simulations,
        enable_segment_subdivision=not args.disable_subdivision,
        segment_length_percentile=args.segment_percentile,
    )
    solver.solve()
    print(write_interface_result(args.output_dir, solver.placedb, solver.homology, solver.segment_manager))


if __name__ == "__main__":
    main()
