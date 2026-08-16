"""Tests for the FloorSet-Lite single-sample converter."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from floorset_benchmark.single_converter import (
    ConversionConfig,
    convert_record,
    validate_artifacts,
)
from floorset_benchmark.scaler import ScaleConfig, scale_case, validate_scaled_artifacts, write_scaled_case


class FloorSetSingleConverterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "area_target": np.array([100.0, 100.0], dtype=np.float32),
            "placement_constraints": np.array(
                [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=np.float32
            ),
            "b2b_connectivity": np.array([[0, 1, 2.0]], dtype=np.float32),
            "p2b_connectivity": np.array([[0, 0, 1.0]], dtype=np.float32),
            "pins_pos": np.array([[-5.0, 5.0]], dtype=np.float32),
            "metrics": np.arange(8, dtype=np.float32),
            "fp_sol": np.array(
                [
                    [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
                    [[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]],
                ],
                dtype=np.float32,
            ),
        }
        self.config = ConversionConfig(sample_id="fixture", source_commit="test")

    def test_convert_record_builds_valid_project_inputs(self) -> None:
        converted = convert_record(self.record, self.config)
        validation = validate_artifacts(converted["block"], converted["pingroup"])
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(converted["statistics"]["floorplan_block_count"], 2)
        self.assertEqual(converted["statistics"]["generated_net_count"], 2)
        self.assertEqual(converted["block"]["children"][0]["module_name"], "MIB_0001")
        self.assertEqual(converted["block"]["children"][1]["module_name"], "MIB_0001")

    def test_validation_rejects_missing_parent(self) -> None:
        converted = convert_record(self.record, self.config)
        converted["pingroup"][0][0]["parent_inst"] = "TOP.MISSING"
        validation = validate_artifacts(converted["block"], converted["pingroup"])
        self.assertFalse(validation["valid"])
        self.assertIn("missing_parent", {error["code"] for error in validation["errors"]})

    def test_repeated_external_pin_becomes_one_multi_endpoint_net(self) -> None:
        self.record["p2b_connectivity"] = np.array(
            [[0, 0, 1.0], [0, 1, 2.0]], dtype=np.float32
        )
        converted = convert_record(self.record, self.config)
        validation = validate_artifacts(converted["block"], converted["pingroup"])
        external_nets = [
            net
            for net in converted["pingroup"]
            if any(pin["parent_inst"] == "TOP.IO0000" for pin in net)
        ]
        self.assertTrue(validation["valid"], validation["errors"])
        self.assertEqual(len(external_nets), 1)
        self.assertEqual(len(external_nets[0]), 3)
        self.assertEqual(converted["statistics"]["p2b_edge_count"], 2)
        self.assertEqual(converted["statistics"]["p2b_net_count"], 1)

    def test_conversion_is_json_deterministic(self) -> None:
        first = convert_record(self.record, self.config)
        second = convert_record(self.record, self.config)
        first_text = json.dumps(first, sort_keys=True, separators=(",", ":"))
        second_text = json.dumps(second, sort_keys=True, separators=(",", ":"))
        self.assertEqual(first_text, second_text)

    def test_scaler_reaches_200_instances_and_exact_reuse_ratio(self) -> None:
        converted = convert_record(self.record, self.config)
        config = ScaleConfig(
            target_module_count=200,
            target_reuse_rate=0.60,
            case_id="smoke-200-r60",
            source_sample_id="fixture",
        )
        scaled = scale_case(converted["block"], converted["pingroup"], config)
        self.assertTrue(scaled["validation"]["valid"], scaled["validation"]["errors"])
        stats = scaled["statistics"]
        self.assertEqual(stats["module_instance_count"], 200)
        self.assertEqual(stats["reused_module_instance_count"], 120)
        self.assertEqual(stats["reuse_instance_ratio"], 0.60)
        self.assertEqual(stats["target_reused_module_instance_count"], 120)
        self.assertEqual(len(scaled["block"]["children"]), 200)

    def test_scaler_is_deterministic_and_rewrites_pin_references(self) -> None:
        converted = convert_record(self.record, self.config)
        config = ScaleConfig(200, 0.95, case_id="smoke-200-r95", source_sample_id="fixture")
        first = scale_case(converted["block"], converted["pingroup"], config)
        second = scale_case(converted["block"], converted["pingroup"], config)
        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )
        child_names = {child["name"] for child in first["block"]["children"]}
        self.assertTrue(all(pin["parent_inst"] in child_names for net in first["pingroup"] for pin in net))
        self.assertEqual(first["statistics"]["reused_module_instance_count"], 190)

    def test_scaler_rejects_target_smaller_than_source(self) -> None:
        converted = convert_record(self.record, self.config)
        with self.assertRaises(ValueError):
            scale_case(converted["block"], converted["pingroup"], ScaleConfig(1, 0.6))

    def test_validator_rejects_same_module_geometry_mismatch(self) -> None:
        converted = convert_record(self.record, self.config)
        block = converted["block"]
        block["children"][1]["vertex"][1][0] += 1.0
        validation = validate_scaled_artifacts(block, converted["pingroup"])
        self.assertFalse(validation["valid"])
        self.assertIn("module_geometry_mismatch", {error["code"] for error in validation["errors"]})

    def test_written_scaling_json_is_byte_deterministic(self) -> None:
        converted = convert_record(self.record, self.config)
        config = ScaleConfig(200, 0.60, case_id="json-repeat", source_sample_id="fixture")
        with TemporaryDirectory() as temp:
            first_dir = Path(temp) / "first"
            second_dir = Path(temp) / "second"
            write_scaled_case(converted["block"], converted["pingroup"], first_dir, config)
            write_scaled_case(converted["block"], converted["pingroup"], second_dir, config)
            for name in ("block.json", "pingroup.json", "manifest.json", "provenance.json", "validation.json", "lineage.json"):
                self.assertEqual((first_dir / name).read_bytes(), (second_dir / name).read_bytes())

    def test_seventy_six_templates_can_represent_200_r95(self) -> None:
        children = []
        for index in range(76):
            x = float(index * 20)
            children.append({
                "name": f"TOP.B{index:04d}", "module_name": f"M{index:04d}",
                "vertex": [[x, 0.0], [x + 10.0, 0.0], [x + 10.0, 10.0], [x, 10.0]],
                "children": [], "direction": 0, "color": "#000000",
            })
        block = {
            "name": "TOP", "module_name": "TOP",
            "vertex": [[0, 0], [1530, 0], [1530, 10], [0, 10]],
            "children": children,
        }
        scaled = scale_case(block, [], ScaleConfig(200, 0.95, case_id="76-to-200-r95"))
        self.assertTrue(scaled["validation"]["valid"], scaled["validation"]["errors"])
        self.assertEqual(scaled["statistics"]["reused_module_instance_count"], 190)
        self.assertEqual(scaled["statistics"]["reuse_instance_ratio"], 0.95)

    def test_cycle8_directions_preserve_area_and_edge_lengths(self) -> None:
        converted = convert_record(self.record, self.config)
        scaled = scale_case(converted["block"], converted["pingroup"], ScaleConfig(16, 0.5, case_id="cycle8"))
        directions = [child["direction"] for child in scaled["block"]["children"]]
        self.assertEqual(set(directions), set(range(8)))
        source_children = converted["block"]["children"]
        for index, child in enumerate(scaled["block"]["children"]):
            vertices = child["vertex"]
            edges = [
                ((vertices[(i + 1) % len(vertices)][0] - vertices[i][0]) ** 2
                 + (vertices[(i + 1) % len(vertices)][1] - vertices[i][1]) ** 2) ** 0.5
                for i in range(len(vertices))
            ]
            source = source_children[index % len(source_children)]["vertex"]
            source_edges = [
                ((source[(i + 1) % len(source)][0] - source[i][0]) ** 2
                 + (source[(i + 1) % len(source)][1] - source[i][1]) ** 2) ** 0.5
                for i in range(len(source))
            ]
            self.assertEqual(sorted(round(edge, 6) for edge in edges), sorted(round(edge, 6) for edge in source_edges))

    def test_wrong_direction_flag_rejects_non_square_homology(self) -> None:
        rectangle = {
            "name": "TOP", "module_name": "TOP", "vertex": [[0, 0], [20, 0], [20, 10], [0, 10]],
            "children": [
                {"name": "TOP.A", "module_name": "R", "direction": 0, "vertex": [[0, 0], [20, 0], [20, 10], [0, 10]], "children": []},
                {"name": "TOP.B", "module_name": "R", "direction": 0, "vertex": [[30, 0], [50, 0], [50, 10], [30, 10]], "children": []},
            ],
        }
        rectangle["children"][1]["direction"] = 2
        validation = validate_scaled_artifacts(rectangle, [])
        self.assertFalse(validation["valid"])
        self.assertIn("module_geometry_mismatch", {error["code"] for error in validation["errors"]})

    def test_non_r0_source_is_canonicalized_before_derived_directions(self) -> None:
        # Base rectangle 20x10, placed with source direction=R90 (2).
        source_polygon_a = [[10, 0], [10, 20], [0, 20], [0, 0]]
        source_polygon_b = [[40, 0], [40, 20], [30, 20], [30, 0]]
        block = {
            "name": "TOP", "module_name": "TOP", "vertex": [[0, 0], [50, 0], [50, 20], [0, 20]],
            "children": [
                {"name": "TOP.A", "module_name": "R", "direction": 2, "vertex": source_polygon_a, "children": []},
                {"name": "TOP.B", "module_name": "R", "direction": 2, "vertex": source_polygon_b, "children": []},
            ],
        }
        scaled = scale_case(block, [], ScaleConfig(16, 0.5, case_id="non-r0-source"))
        self.assertTrue(scaled["validation"]["valid"], scaled["validation"]["errors"])
        self.assertEqual(set(child["direction"] for child in scaled["block"]["children"]), set(range(8)))
        self.assertEqual(scaled["statistics"]["reused_module_instance_count"], 8)


if __name__ == "__main__":
    unittest.main()
