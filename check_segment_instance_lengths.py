"""Check that reused segment instances have equal geometric length.

The Stage1 assignment JSON stores each abstract segment once, with all reused
module instances under ``segment_instances``.  This script verifies that every
instance of the same abstract segment has the same start-end length.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_ASSIGNMENT_PATH = Path(r"C:\Users\DELL\Desktop\stage1_assignment.json")


def _point(value: Any) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"Expected [x, y], got {value!r}")
    return (float(value[0]), float(value[1]))


def _segment_length(instance: dict[str, Any]) -> float:
    start = _point(instance.get("start"))
    end = _point(instance.get("end"))
    return math.hypot(end[0] - start[0], end[1] - start[1])


def _converted_segment_length(instance: dict[str, Any]) -> float:
    coordinates = instance.get("coordinates")
    if isinstance(coordinates, list | tuple) and len(coordinates) == 4:
        x1, y1, x2, y2 = (float(value) for value in coordinates)
        return math.hypot(x2 - x1, y2 - y1)
    if "length" in instance:
        return float(instance["length"])
    segment_info = instance.get("segment_info", {})
    if "length" in segment_info:
        return float(segment_info["length"])
    raise ValueError(f"Cannot find coordinates or length in instance {instance!r}")


def _check_group_lengths(
    group_id: str,
    group: dict[str, Any],
    instances: dict[str, Any],
    length_fn,
    tolerance: float,
) -> dict[str, Any] | None:
    if not isinstance(instances, dict) or len(instances) <= 1:
        return None

    lengths = {
        instance_name: length_fn(instance)
        for instance_name, instance in sorted(instances.items())
    }
    reference_name, reference_length = next(iter(lengths.items()))
    mismatches = []
    for instance_name, length in list(lengths.items())[1:]:
        diff = abs(length - reference_length)
        if diff > tolerance:
            mismatches.append(
                {
                    "segment_id": group_id,
                    "module_name": group.get("module_name"),
                    "reference_instance": reference_name,
                    "reference_length": reference_length,
                    "instance": instance_name,
                    "instance_length": length,
                    "difference": diff,
                }
            )
    return {"checked": 1, "mismatches": mismatches}


def _check_stage1_format(segments: dict[str, Any], tolerance: float):
    checked_groups = 0
    mismatches = []
    for segment_id, segment in sorted(segments.items()):
        result = _check_group_lengths(
            segment_id,
            segment,
            segment.get("segment_instances", {}),
            _segment_length,
            tolerance,
        )
        if result is None:
            continue
        checked_groups += result["checked"]
        mismatches.extend(result["mismatches"])
    return checked_groups, mismatches


def _check_converted_format(segment_assignments: dict[str, Any], tolerance: float):
    checked_groups = 0
    mismatches = []
    for segment_key, segment in sorted(segment_assignments.items()):
        segment_id = str(segment.get("segment_id", segment_key))
        result = _check_group_lengths(
            segment_id,
            segment,
            segment.get("segment_insts", {}),
            _converted_segment_length,
            tolerance,
        )
        if result is None:
            continue
        checked_groups += result["checked"]
        mismatches.extend(result["mismatches"])
    return checked_groups, mismatches


def check_segment_instance_lengths(
    assignment_path: Path,
    tolerance: float,
) -> tuple[int, list[dict[str, Any]]]:
    data = json.loads(assignment_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Assignment JSON must be a dict.")

    if isinstance(data.get("segments"), dict):
        return _check_stage1_format(data["segments"], tolerance)
    if isinstance(data.get("segment_assignments"), dict):
        return _check_converted_format(data["segment_assignments"], tolerance)
    raise ValueError(
        "Unsupported assignment format: expected 'segments' or 'segment_assignments'."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check reused segment instance lengths in a Stage1 assignment JSON."
    )
    parser.add_argument(
        "assignment",
        nargs="?",
        type=Path,
        default=DEFAULT_ASSIGNMENT_PATH,
        help=f"Path to stage1_assignment.json, default: {DEFAULT_ASSIGNMENT_PATH}",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Allowed absolute length difference for reused segment instances.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checked_groups, mismatches = check_segment_instance_lengths(
        args.assignment,
        args.tolerance,
    )
    print(f"Checked reused segment groups: {checked_groups}")
    if not mismatches:
        print("All reused segment instances have matching lengths.")
        return 0

    print(f"Found length mismatches: {len(mismatches)}")
    for item in mismatches[:50]:
        print(
            "segment_id={segment_id}, module_name={module_name}, "
            "reference={reference_instance}({reference_length:.12g}), "
            "instance={instance}({instance_length:.12g}), diff={difference:.12g}".format(
                **item
            )
        )
    if len(mismatches) > 50:
        print(f"... omitted {len(mismatches) - 50} additional mismatches")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
