from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from floorset_benchmark.calibration import _overlap_count, validate_case
from floorset_benchmark.scaler import ScaleConfig, scale_case
from floorset_benchmark.single_converter import ConversionConfig, convert_record


class CalibrationValidationTests(unittest.TestCase):
    def test_bbox_contact_without_polygon_area_is_not_overlap(self) -> None:
        block = {"children": [
            {"name": "TOP.A", "vertex": [[0, 0], [2, 0], [0, 2]]},
            {"name": "TOP.B", "vertex": [[1.5, 1.5], [3.5, 1.5], [1.5, 3.5]]},
        ]}
        self.assertEqual(_overlap_count(block), 0)

    def _case(self):
        record = {
            "area_target": np.array([100.0, 100.0], dtype=np.float32),
            "placement_constraints": np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=np.float32),
            "b2b_connectivity": np.array([[0, 1, 2.0]], dtype=np.float32),
            "p2b_connectivity": np.array([[0, 0, 1.0]], dtype=np.float32),
            "pins_pos": np.array([[-5.0, 5.0]], dtype=np.float32),
            "metrics": np.arange(8, dtype=np.float32),
            "fp_sol": np.array([[[0, 0], [10, 0], [10, 10], [0, 10]], [[20, 0], [30, 0], [30, 10], [20, 10]]], dtype=np.float32),
        }
        converted = convert_record(record, ConversionConfig(sample_id="cal", source_commit="test"))
        return scale_case(converted["block"], converted["pingroup"], ScaleConfig(20, 0.6, target_pingroup_count=20))

    def test_calibration_validator_accepts_exact_case_and_rejects_count(self) -> None:
        scaled = self._case()
        with TemporaryDirectory() as temp:
            case = Path(temp)
            (case / "block.json").write_text(json.dumps(scaled["block"]), encoding="utf-8")
            (case / "pingroup.json").write_text(json.dumps(scaled["pingroup"]), encoding="utf-8")
            report = validate_case(case, 20)
            self.assertTrue(report["validation"]["valid"], report["validation"]["errors"])
            bad = validate_case(case, 21)
            self.assertFalse(bad["validation"]["valid"])
            self.assertIn("pingroup_count_mismatch", {error["code"] for error in bad["validation"]["errors"]})

    def test_calibration_validator_rejects_bad_parent(self) -> None:
        scaled = self._case()
        scaled["pingroup"][0][0]["parent_inst"] = "TOP.BAD"
        with TemporaryDirectory() as temp:
            case = Path(temp)
            (case / "block.json").write_text(json.dumps(scaled["block"]), encoding="utf-8")
            (case / "pingroup.json").write_text(json.dumps(scaled["pingroup"]), encoding="utf-8")
            report = validate_case(case, 20)
            self.assertFalse(report["validation"]["valid"])
            self.assertIn("missing_parent", {error["code"] for error in report["validation"]["errors"]})

    def test_target_pingroup_scaling_is_byte_deterministic(self) -> None:
        first, second = self._case(), self._case()
        self.assertEqual(first["statistics"]["pingroup_count"], 20)
        for key in ("block", "pingroup", "lineage"):
            self.assertEqual(json.dumps(first[key], sort_keys=True), json.dumps(second[key], sort_keys=True))

    def test_calibration_validator_rejects_bad_width_and_duplicate_full_pin(self) -> None:
        scaled = self._case()
        scaled["pingroup"][0][0]["width"] = 0.0
        with TemporaryDirectory() as temp:
            case = Path(temp)
            (case / "block.json").write_text(json.dumps(scaled["block"]), encoding="utf-8")
            (case / "pingroup.json").write_text(json.dumps(scaled["pingroup"]), encoding="utf-8")
            report = validate_case(case, 20)
            self.assertFalse(report["validation"]["valid"])
            self.assertIn("invalid_pin_width", {error["code"] for error in report["validation"]["errors"]})
        scaled = self._case()
        scaled["pingroup"][0][1]["parent_inst"] = scaled["pingroup"][0][0]["parent_inst"]
        scaled["pingroup"][0][1]["parent_module"] = scaled["pingroup"][0][0]["parent_module"]
        scaled["pingroup"][0][1]["pingroup_name"] = scaled["pingroup"][0][0]["pingroup_name"]
        with TemporaryDirectory() as temp:
            case = Path(temp)
            (case / "block.json").write_text(json.dumps(scaled["block"]), encoding="utf-8")
            (case / "pingroup.json").write_text(json.dumps(scaled["pingroup"]), encoding="utf-8")
            report = validate_case(case, 20)
            self.assertFalse(report["validation"]["valid"])
            self.assertIn("duplicate_pin_full_name", {error["code"] for error in report["validation"]["errors"]})
