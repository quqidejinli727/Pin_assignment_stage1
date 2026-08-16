"""Convert one official FloorSet-Lite test sample to project JSON inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .capacity_calibration import (
    CapacityCalibrationConfig,
    calibrate_case,
    connectivity_weight_audit,
)
from .torch_zip import load_torch_zip


CONVERTER_VERSION = "floorset-lite-single-v0.2"
FLOORSET_REPOSITORY = "https://github.com/IntelLabs/FloorSet"
FLOORSET_PAPER = "https://arxiv.org/abs/2405.05480"
FLOORSET_DATA_LICENSE = "CC-BY-4.0"


@dataclass(frozen=True)
class ConversionConfig:
    """Configuration for one FloorSet-Lite sample conversion."""

    sample_id: str
    source_commit: str
    base_seed: int = 7
    width_scale: float = 0.01
    width_min: float = 0.001
    io_box_scale: float = 1.0e-4
    synthetic_homology: bool = True
    reader: str = "auto"
    width_policy: str = "floorset_first"

    @property
    def derived_seed(self) -> int:
        material = f"{self.base_seed}|{self.sample_id}|{CONVERTER_VERSION}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_array(value: Any, name: str, dimensions: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if dimensions is not None and array.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions, got {array.shape}.")
    return array


def _to_numpy(value: Any) -> Any:
    """Recursively convert PyTorch tensors to NumPy arrays."""
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if isinstance(value, list):
        return [_to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_numpy(item) for item in value)
    return value


def _load_archive(path: str | Path, reader: str) -> Any:
    """Load with local PyTorch when available, otherwise use the restricted reader."""
    normalized = reader.lower().strip()
    if normalized not in {"auto", "torch", "restricted"}:
        raise ValueError(f"Unsupported reader {reader!r}; use auto, torch, or restricted.")
    if normalized in {"auto", "torch"}:
        try:
            import torch
        except (ImportError, OSError):
            if normalized == "torch":
                raise
        else:
            return _to_numpy(torch.load(Path(path), map_location="cpu", weights_only=True))
    return load_torch_zip(path)


def load_lite_test_sample(
    input_path: str | Path,
    label_path: str | Path,
    layout_index: int = 0,
    reader: str = "auto",
) -> dict[str, np.ndarray]:
    """Load one layout using the structure in FloorSet's official test loader."""
    inputs = _load_archive(input_path, reader)
    labels = _load_archive(label_path, reader)
    if not 0 <= layout_index < len(inputs) or not 0 <= layout_index < len(labels):
        raise IndexError(f"Layout index {layout_index} is unavailable.")
    input_layout = inputs[layout_index]
    label_layout = labels[layout_index]
    if len(input_layout) != 4 or len(label_layout) != 2:
        raise ValueError(
            f"Unexpected Lite test payload: input={len(input_layout)}, label={len(label_layout)}."
        )
    block_data = _as_array(input_layout[0], "block_data", 2)
    if block_data.shape[1] < 6:
        raise ValueError(f"block_data must contain area plus five constraints, got {block_data.shape}.")
    return {
        "area_target": block_data[:, 0],
        "placement_constraints": block_data[:, 1:6],
        "b2b_connectivity": _as_array(input_layout[1], "b2b_connectivity", 2),
        "p2b_connectivity": _as_array(input_layout[2], "p2b_connectivity", 2),
        "pins_pos": _as_array(input_layout[3], "pins_pos", 2),
        "metrics": _as_array(label_layout[0], "metrics"),
        "fp_sol": _as_array(label_layout[1], "fp_sol"),
    }


def _valid_rows(array: np.ndarray, endpoint_columns: int) -> np.ndarray:
    if array.size == 0:
        return array.reshape((0, array.shape[-1] if array.ndim == 2 else endpoint_columns))
    return array[np.all(array[:, :endpoint_columns] >= 0, axis=1)]


def _polygon_from_solution(solution: np.ndarray, block_index: int) -> list[list[float]]:
    if solution.ndim == 2 and solution.shape[1] == 4:
        width, height, x, y = (float(value) for value in solution[block_index])
        if width <= 0 or height <= 0:
            raise ValueError(f"Block {block_index} has invalid width/height: {width}, {height}.")
        return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    if solution.ndim == 3 and solution.shape[2] == 2:
        points = [
            [float(point[0]), float(point[1])]
            for point in solution[block_index]
            if not (float(point[0]) == -1.0 and float(point[1]) == -1.0)
        ]
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        if len(points) < 3:
            raise ValueError(f"Block {block_index} has fewer than three polygon vertices.")
        return points
    raise ValueError(f"Unsupported FloorSet-Lite solution shape: {solution.shape}.")


def _polygon_area(points: list[list[float]]) -> float:
    if any(not math.isfinite(float(coordinate)) for point in points for coordinate in point):
        return 0.0
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
    ) / 2.0


def _module_type(block_index: int, constraints: np.ndarray) -> str:
    mib_id = int(round(float(constraints[block_index, 2])))
    return f"MIB_{mib_id:04d}" if mib_id > 0 else f"BLOCK_{block_index:04d}"


def _port_names(
    block_count: int,
    b2b: np.ndarray,
    p2b: np.ndarray,
) -> dict[tuple[str, int, int], str]:
    incident: dict[int, list[tuple[str, int, int]]] = {index: [] for index in range(block_count)}
    for edge_index, row in enumerate(b2b):
        left, right = int(row[0]), int(row[1])
        incident[left].append(("b2b", right, edge_index))
        incident[right].append(("b2b", left, edge_index))
    for edge_index, row in enumerate(p2b):
        pin_index, block_index = int(row[0]), int(row[1])
        incident[block_index].append(("p2b", pin_index, edge_index))
    result: dict[tuple[str, int, int], str] = {}
    for block_index, entries in incident.items():
        for ordinal, entry in enumerate(sorted(entries)):
            result[(entry[0], entry[2], block_index)] = f"P{ordinal:04d}"
    return result


def _pin(
    parent_inst: str,
    parent_module: str,
    pingroup_name: str,
    width: float,
    *,
    source_connectivity_weight: float | None = None,
    source_connectivity_kind: str | None = None,
    source_edge_index: int | None = None,
) -> dict[str, Any]:
    pin = {
        "parent_inst": parent_inst,
        "parent_module": parent_module,
        "pingroup_name": pingroup_name,
        "scope": [],
        "successors": [],
        "width": float(width),
    }
    if source_connectivity_weight is not None:
        pin["source_connectivity_weight"] = float(source_connectivity_weight)
    if source_connectivity_kind is not None:
        pin["source_connectivity_kind"] = source_connectivity_kind
    if source_edge_index is not None:
        pin["source_edge_index"] = int(source_edge_index)
    return pin


def _edge_width(row: np.ndarray, config: ConversionConfig) -> float:
    weight = float(row[2]) if row.shape[0] > 2 else 1.0
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"Invalid connectivity weight: {weight!r}")
    return max(config.width_min, weight * config.width_scale)


def convert_record(record: dict[str, np.ndarray], config: ConversionConfig) -> dict[str, Any]:
    """Convert a normalized official Lite record to project input objects."""
    areas = record["area_target"]
    constraints = record["placement_constraints"]
    solution = record["fp_sol"]
    block_count = int(np.count_nonzero(areas >= 0))
    if block_count <= 0 or constraints.shape[0] < block_count or solution.shape[0] < block_count:
        raise ValueError("FloorSet record has inconsistent block counts.")
    b2b = _valid_rows(record["b2b_connectivity"], 2)
    p2b = _valid_rows(record["p2b_connectivity"], 2)
    pins_pos = record["pins_pos"]
    if b2b.shape[1] < 2 or p2b.shape[1] < 2 or pins_pos.shape[1] != 2:
        raise ValueError("Connectivity and pin tensors have unsupported shapes.")
    if b2b.size and (np.max(b2b[:, :2]) >= block_count):
        raise ValueError("b2b_connectivity references an invalid block index.")
    if p2b.size:
        if np.max(p2b[:, 1]) >= block_count or np.max(p2b[:, 0]) >= pins_pos.shape[0]:
            raise ValueError("p2b_connectivity references an invalid pin or block index.")

    polygons = [_polygon_from_solution(solution, index) for index in range(block_count)]
    for index, polygon in enumerate(polygons):
        if _polygon_area(polygon) <= 0:
            raise ValueError(f"Block {index} has a degenerate polygon.")
    module_types = [_module_type(index, constraints) for index in range(block_count)]
    block_names = [f"TOP.B{index:04d}" for index in range(block_count)]

    io_indices = sorted({int(row[0]) for row in p2b})
    xs = [point[0] for polygon in polygons for point in polygon]
    ys = [point[1] for polygon in polygons for point in polygon]
    xs.extend(float(pins_pos[index, 0]) for index in io_indices)
    ys.extend(float(pins_pos[index, 1]) for index in io_indices)
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    io_half_size = max(span * config.io_box_scale, 1.0e-6)

    children = [
        {
            "name": block_names[index],
            "module_name": module_types[index],
            "direction": 0,
            "color": f"#{(index * 2654435761) & 0xFFFFFF:06x}",
            "vertex": polygons[index],
            "children": [],
        }
        for index in range(block_count)
    ]
    io_names: dict[int, str] = {}
    for pin_index in io_indices:
        x, y = (float(value) for value in pins_pos[pin_index])
        if x == -1.0 and y == -1.0:
            raise ValueError(f"Referenced external pin {pin_index} is padding.")
        name = f"TOP.IO{pin_index:04d}"
        io_names[pin_index] = name
        children.append(
            {
                "name": name,
                "module_name": f"IO_{pin_index:04d}",
                "direction": 0,
                "color": "#333333",
                "vertex": [
                    [x - io_half_size, y - io_half_size],
                    [x + io_half_size, y - io_half_size],
                    [x + io_half_size, y + io_half_size],
                    [x - io_half_size, y + io_half_size],
                ],
                "children": [],
            }
        )

    root_margin = max(span * 1.0e-6, 1.0e-6)
    block_json = {
        "name": "TOP",
        "module_name": "TOP",
        "direction": 0,
        "color": "#000000",
        "vertex": [
            [min(xs) - root_margin, min(ys) - root_margin],
            [max(xs) + root_margin, min(ys) - root_margin],
            [max(xs) + root_margin, max(ys) + root_margin],
            [min(xs) - root_margin, max(ys) + root_margin],
        ],
        "children": children,
    }

    port_names = _port_names(block_count, b2b, p2b)
    nets: list[list[dict[str, Any]]] = []
    for edge_index, row in enumerate(b2b):
        left, right = int(row[0]), int(row[1])
        width = _edge_width(row, config)
        source_weight = float(row[2]) if row.shape[0] > 2 else 1.0
        nets.append(
            [
                _pin(
                    block_names[left], module_types[left], port_names[("b2b", edge_index, left)], width,
                    source_connectivity_weight=source_weight,
                    source_connectivity_kind="b2b_weight",
                    source_edge_index=edge_index,
                ),
                _pin(
                    block_names[right], module_types[right], port_names[("b2b", edge_index, right)], width,
                    source_connectivity_weight=source_weight,
                    source_connectivity_kind="b2b_weight",
                    source_edge_index=edge_index,
                ),
            ]
        )
    p2b_by_pin: dict[int, list[tuple[int, np.ndarray]]] = {}
    for edge_index, row in enumerate(p2b):
        p2b_by_pin.setdefault(int(row[0]), []).append((edge_index, row))
    for pin_index in sorted(p2b_by_pin):
        rows = sorted(p2b_by_pin[pin_index], key=lambda item: (int(item[1][1]), item[0]))
        terminal_width = max(_edge_width(row, config) for _, row in rows)
        terminal_weight = max(float(row[2]) if row.shape[0] > 2 else 1.0 for _, row in rows)
        net = [
            _pin(
                io_names[pin_index], f"IO_{pin_index:04d}", "TERM", terminal_width,
                source_connectivity_weight=terminal_weight,
                source_connectivity_kind="p2b_weight",
            )
        ]
        for edge_index, row in rows:
            block_index = int(row[1])
            source_weight = float(row[2]) if row.shape[0] > 2 else 1.0
            net.append(
                _pin(
                    block_names[block_index],
                    module_types[block_index],
                    port_names[("p2b", edge_index, block_index)],
                    _edge_width(row, config),
                    source_connectivity_weight=source_weight,
                    source_connectivity_kind="p2b_weight",
                    source_edge_index=edge_index,
                )
            )
        nets.append(net)

    converted = {
        "block": block_json,
        "pingroup": nets,
        "statistics": {
            "floorplan_block_count": block_count,
            "generated_module_instance_count": len(children),
            "b2b_net_count": int(b2b.shape[0]),
            "p2b_edge_count": int(p2b.shape[0]),
            "p2b_net_count": len(p2b_by_pin),
            "generated_net_count": len(nets),
            "generated_pin_count": sum(len(net) for net in nets),
            "generated_pingroup_count": len(
                {
                    f"{pin['parent_module']}.{pin['pingroup_name']}"
                    for net in nets
                    for pin in net
                }
            ),
            "mib_module_type_count": len({name for name in module_types if name.startswith("MIB_")}),
        },
    }
    if config.width_policy == "floorset_first":
        calibrated = calibrate_case(
            converted["block"],
            converted["pingroup"],
            CapacityCalibrationConfig(policy="floorset_first", seed=config.derived_seed),
        )
        converted["block"] = calibrated["block"]
        converted["pingroup"] = calibrated["pingroup"]
        converted["capacity_calibration"] = calibrated["report"]
    elif config.width_policy != "legacy":
        raise ValueError("width_policy must be 'floorset_first' or 'legacy'")
    return converted


def _tensor_summary(name: str, value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.number) else np.asarray([])
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "element_count": int(array.size),
        "padding_minus_one_count": int(np.count_nonzero(array == -1)),
        "minimum": float(np.min(finite)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def validate_artifacts(
    block: dict[str, Any],
    pingroup: list[list[dict[str, Any]]],
    *,
    allowed_singleton_net_indices: set[int] | None = None,
) -> dict[str, Any]:
    """Validate project-level references and geometry without mutating artifacts."""
    errors: list[dict[str, str]] = []
    instances: dict[str, dict[str, Any]] = {}
    allowed_singletons = allowed_singleton_net_indices or set()
    accepted_singleton_count = 0

    def walk(module: dict[str, Any]) -> None:
        name = str(module.get("name", ""))
        if not name or name in instances:
            errors.append({"code": "duplicate_or_empty_module", "detail": name})
        instances[name] = module
        vertex = module.get("vertex", [])
        if len(vertex) < 3 or _polygon_area(vertex) <= 0:
            errors.append({"code": "invalid_polygon", "detail": name})
        for child in module.get("children", []):
            walk(child)

    walk(block)
    seen_full_names: set[str] = set()
    for net_index, net in enumerate(pingroup):
        if len(net) < 2:
            if len(net) == 1 and net_index in allowed_singletons:
                accepted_singleton_count += 1
            else:
                errors.append({"code": "short_net", "detail": str(net_index)})
        for pin in net:
            parent_inst = str(pin.get("parent_inst", ""))
            module = instances.get(parent_inst)
            if module is None:
                errors.append({"code": "missing_parent", "detail": parent_inst})
                continue
            if pin.get("parent_module") != module.get("module_name"):
                errors.append({"code": "module_type_mismatch", "detail": parent_inst})
            if float(pin.get("width", 0.0)) <= 0:
                errors.append({"code": "invalid_pin_width", "detail": parent_inst})
            full_name = f"{parent_inst}.{pin.get('pingroup_name', '')}"
            if full_name in seen_full_names:
                errors.append({"code": "duplicate_pin_full_name", "detail": full_name})
            seen_full_names.add(full_name)
    unconsumed_allowances = allowed_singletons - {
        index for index, net in enumerate(pingroup) if len(net) == 1
    }
    for net_index in sorted(unconsumed_allowances):
        errors.append({"code": "invalid_singleton_allowance", "detail": str(net_index)})
    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "module_instance_count": len(instances) - 1,
        "net_count": len(pingroup),
        "pin_count": sum(len(net) for net in pingroup),
        "accepted_singleton_net_count": accepted_singleton_count,
        "unconsumed_singleton_allowance_count": len(unconsumed_allowances),
    }


def convert_lite_sample(
    input_path: str | Path,
    label_path: str | Path,
    output_dir: str | Path,
    config: ConversionConfig,
) -> dict[str, Any]:
    """Convert, validate, and write one official FloorSet-Lite sample."""
    source_input = Path(input_path).resolve()
    source_label = Path(label_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    record = load_lite_test_sample(source_input, source_label, reader=config.reader)
    converted = convert_record(record, config)
    validation = validate_artifacts(converted["block"], converted["pingroup"])
    if not validation["valid"]:
        raise ValueError(f"Converted case failed validation: {validation['errors'][:5]!r}")

    tensor_report = {
        "sample_id": config.sample_id,
        "fields": [_tensor_summary(name, value) for name, value in sorted(record.items())],
        "connectivity_weight_audit": connectivity_weight_audit(record),
    }
    provenance = {
        "source_repository": FLOORSET_REPOSITORY,
        "source_commit": config.source_commit,
        "source_sample_id": config.sample_id,
        "source_input_sha256": _sha256(source_input),
        "source_label_sha256": _sha256(source_label),
        "source_data_license": FLOORSET_DATA_LICENSE,
        "paper": FLOORSET_PAPER,
        "derived_artifact_label": "FloorSet-derived synthetic PinAssign benchmark",
    }
    manifest = {
        "converter_version": CONVERTER_VERSION,
        "config": asdict(config),
        "derived_seed": config.derived_seed,
        "statistics": converted["statistics"],
        "policies": {
            "geometry": "official Lite fp_sol polygons; repeated closing point removed",
            "module_type": "positive MIB id -> shared MIB type; otherwise unique block type",
            "terminal": "one synthetic rectangular IO module and one multi-endpoint net per referenced FloorSet external pin",
            "pin_width": (
                "FloorSet connectivity-weight rank -> capacity pressure -> width"
                if config.width_policy == "floorset_first"
                else "max(width_min, connectivity_weight * width_scale)"
            ),
            "source_weight_semantics": "connectivity cost prior; not physical pin width",
            "homology": "canonical incident-edge ordinal per block; synthetic when module type is reused",
            "target_scale_matrix": "not_applicable_to_single_sample_v0",
        },
        "provenance_file": "provenance.json",
        "validation_file": "validation.json",
        "tensor_report_file": "tensor_report.json",
        "capacity_report_file": "capacity_report.json" if "capacity_calibration" in converted else None,
    }

    _json_dump(destination / "block.json", converted["block"])
    _json_dump(destination / "pingroup.json", converted["pingroup"])
    _json_dump(destination / "manifest.json", manifest)
    _json_dump(destination / "provenance.json", provenance)
    _json_dump(destination / "validation.json", validation)
    _json_dump(destination / "tensor_report.json", tensor_report)
    if "capacity_calibration" in converted:
        _json_dump(destination / "capacity_report.json", converted["capacity_calibration"])
    artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(destination.glob("*.json"))
    }
    return {
        "output_dir": str(destination),
        "statistics": converted["statistics"],
        "validation": validation,
        "artifact_sha256": artifact_hashes,
    }


def _source_commit(source_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Official litedata_*.pth file.")
    parser.add_argument("--label", required=True, type=Path, help="Matching litelabel_*.pth file.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width-scale", type=float, default=0.01)
    parser.add_argument("--width-min", type=float, default=0.001)
    parser.add_argument("--width-policy", choices=("floorset_first", "legacy"), default="floorset_first")
    parser.add_argument(
        "--reader",
        choices=("auto", "torch", "restricted"),
        default="auto",
        help="Tensor archive reader; auto prefers local PyTorch.",
    )
    args = parser.parse_args()
    if args.width_scale <= 0 or args.width_min <= 0:
        parser.error("--width-scale and --width-min must be positive.")
    commit = args.source_commit
    if not commit and args.source_root:
        commit = _source_commit(args.source_root)
    config = ConversionConfig(
        sample_id=args.sample_id,
        source_commit=commit or "unknown",
        base_seed=args.seed,
        width_scale=args.width_scale,
        width_min=args.width_min,
        width_policy=args.width_policy,
        reader=args.reader,
    )
    result = convert_lite_sample(args.input, args.label, args.output_dir, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
