from __future__ import annotations

import json
import unittest

import numpy as np

from floorset_benchmark.capacity_calibration import compare_calibrations
from floorset_benchmark.scaler import ScaleConfig, scale_case
from floorset_benchmark.single_converter import ConversionConfig, convert_record


def _record() -> dict[str, np.ndarray]:
    return {
        "area_target": np.array([100.0, 100.0], dtype=np.float32),
        "placement_constraints": np.array(
            [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=np.float32
        ),
        "b2b_connectivity": np.array(
            [[0, 1, 1.0], [0, 1, 20.0]], dtype=np.float32
        ),
        "p2b_connectivity": np.array([[0, 0, 0.5]], dtype=np.float32),
        "pins_pos": np.array([[-5.0, 5.0]], dtype=np.float32),
        "metrics": np.arange(8, dtype=np.float32),
        "fp_sol": np.array(
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[20, 0], [30, 0], [30, 10], [20, 10]],
            ],
            dtype=np.float32,
        ),
    }


class CapacityCalibrationTests(unittest.TestCase):
    def test_converter_preserves_weight_provenance_and_avoids_flat_widths(self) -> None:
        converted = convert_record(
            _record(), ConversionConfig(sample_id="capacity", source_commit="test")
        )
        pins = [pin for net in converted["pingroup"] for pin in net]
        self.assertTrue(all("source_connectivity_weight" in pin for pin in pins))
        self.assertGreater(len({round(float(pin["width"]), 12) for pin in pins}), 1)
        report = converted["capacity_calibration"]
        self.assertTrue(report["source_weight_to_pressure_monotonic"])
        self.assertEqual(report["packing"]["hard_overflow_count"], 0)

    def test_same_topology_comparison_has_capacity_effect_without_stress(self) -> None:
        converted = convert_record(
            _record(), ConversionConfig(sample_id="compare", source_commit="test")
        )
        scaled = scale_case(
            converted["block"],
            converted["pingroup"],
            ScaleConfig(
                200,
                0.60,
                case_id="capacity-1k",
                source_sample_id="fixture",
                target_pingroup_count=1000,
                width_policy="preserve",
            ),
        )
        comparison = compare_calibrations(
            scaled["block"], scaled["pingroup"], seed=7, auto_stress=True
        )
        baseline = comparison["baseline"]["report"]
        floorset = comparison["floorset_first"]["report"]
        self.assertEqual(
            json.dumps(comparison["baseline"]["block"], sort_keys=True),
            json.dumps(comparison["floorset_first"]["block"], sort_keys=True),
        )
        self.assertGreaterEqual(floorset["small_group_ratio"], 0.70)
        self.assertGreater(floorset["width_summary"]["max"], floorset["width_summary"]["min"])
        self.assertTrue(floorset["source_weight_to_pressure_monotonic"])
        self.assertTrue(comparison["effect"]["observable_capacity_effect"])
        self.assertIsNone(comparison["stress"])
        self.assertEqual(baseline["packing"]["hard_overflow_count"], 0)
        self.assertEqual(floorset["packing"]["hard_overflow_count"], 0)

    def test_scaled_floorset_first_policy_is_deterministic(self) -> None:
        converted = convert_record(
            _record(), ConversionConfig(sample_id="scale-policy", source_commit="test")
        )
        config = ScaleConfig(
            200,
            0.95,
            case_id="capacity-policy",
            source_sample_id="fixture",
            target_pingroup_count=1000,
            width_policy="floorset_first",
        )
        first = scale_case(converted["block"], converted["pingroup"], config)
        second = scale_case(converted["block"], converted["pingroup"], config)
        self.assertEqual(
            json.dumps(first["pingroup"], sort_keys=True),
            json.dumps(second["pingroup"], sort_keys=True),
        )
        report = first["capacity_calibration"]
        self.assertGreaterEqual(report["small_group_ratio"], 0.70)
        self.assertEqual(report["packing"]["hard_overflow_count"], 0)
        self.assertTrue(first["validation"]["valid"], first["validation"]["errors"])


if __name__ == "__main__":
    unittest.main()
