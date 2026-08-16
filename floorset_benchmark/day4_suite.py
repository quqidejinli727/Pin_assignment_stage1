"""Reproducible Day 4 calibration suite runner."""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

from .calibration import validate_case
from .scaler import ScaleConfig, write_scaled_case


SOURCE_COMMIT = "aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa"


def run_suite(source_dir: str | Path, output_root: str | Path) -> dict:
    source_dir, output_root = Path(source_dir), Path(output_root)
    block = json.loads((source_dir / "block.json").read_text(encoding="utf-8"))
    pingroup = json.loads((source_dir / "pingroup.json").read_text(encoding="utf-8"))
    cases = [("1k_r60", 380, 1000, 0.60), ("1k_r95", 380, 1000, 0.95), ("3k_r60", 1200, 3000, 0.60), ("3k_r95", 1200, 3000, 0.95)]
    entries = []
    for case_id, modules, pingroups, reuse in cases:
        case_dir = output_root / f"floorset_day4_calibration_{case_id}"
        config = ScaleConfig(modules, reuse, 7, f"day4-{case_id}", "floorset_config21_1", 1.0, SOURCE_COMMIT, "cycle8", pingroups)
        tracemalloc.start()
        started = time.perf_counter()
        stats = write_scaled_case(block, pingroup, case_dir, config)
        generation_seconds = time.perf_counter() - started
        peak_generation_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()
        report = validate_case(case_dir, pingroups)
        (case_dir / "calibration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        json_bytes = sum(path.stat().st_size for path in case_dir.glob("*.json") if path.name != "calibration_report.json")
        entries.append({"case_id": case_id, "target_modules": modules, "target_pingroup_count": pingroups, "target_reuse_rate": reuse, "generation_seconds": generation_seconds, "generation_peak_tracemalloc_mb": peak_generation_mb, "artifact_json_bytes": json_bytes, "artifact_json_mb": json_bytes / (1024 * 1024), "report": report})
    baseline = {entry["target_reuse_rate"]: entry for entry in entries if entry["case_id"].startswith("3k_")}
    predictions: dict[str, dict[str, dict[str, float]]] = {}
    for target in (20000, 30000, 200000):
        predictions[f"{target // 1000}K"] = {}
        for reuse, entry in sorted(baseline.items()):
            report = entry["report"]
            scale = (target / 3000.0) * 2.0
            predictions[f"{target // 1000}K"][f"R{int(reuse * 100)}"] = {
                "generation_seconds": entry["generation_seconds"] * scale,
                "placedb_load_seconds": report["timing_seconds"]["placedb_load"] * scale,
                "segment_build_seconds": report["timing_seconds"]["segment_build"] * scale,
                "batch_plan_seconds": report["timing_seconds"]["batch_plan"] * scale,
                "validation_seconds": report["timing_seconds"]["validation"] * scale,
                "artifact_json_bytes": entry["artifact_json_bytes"] * scale,
                "artifact_json_mb": entry["artifact_json_mb"] * scale,
                "peak_tracemalloc_mb": report["peak_tracemalloc_mb"] * scale,
            }
    summary = {
        "source_dir": str(source_dir), "source_commit": SOURCE_COMMIT, "formal_matrix_generated": False, "cases": entries,
        "prediction": {
            "method": "3K calibration baseline, linear by PinGroup count, uniformly multiplied by 2.0",
            "baseline_modules": 1200, "formal_default_modules": 240,
            "validation_includes_polygon_overlap_check": True,
            "predictions": predictions,
            "formal_targets": {"20K": {"decision": "budget only"}, "30K": {"decision": "budget only"}, "200K": {"decision": "No-Go pending memory/partition test"}},
            "tracemalloc_note": "Python allocations only; native RSS is not included.",
        },
    }
    (output_root / "day4_suite_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_suite(args.source_dir, args.output_root), ensure_ascii=False, indent=2, sort_keys=True))
