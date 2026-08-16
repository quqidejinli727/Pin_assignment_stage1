from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from PlaceDB import PlaceDB
from homology import HomologyManager

from floorset_benchmark.hierarchical_net_synthesizer import (
    ARCHETYPES,
    HierarchicalNetSynthesizer,
    NetSynthesisConfig,
    validate_hierarchical_nets,
)
from floorset_benchmark.hierarchical_suite import (
    HierarchicalCaseConfig,
    build_hierarchical_case,
)
from floorset_benchmark.hierarchy_composer import (
    HierarchyConfig,
    compose_hierarchy,
)
from floorset_benchmark.day6_suite import capacity_projection
from floorset_benchmark.single_converter import validate_artifacts


def _source_block() -> dict:
    children = []
    for index, (width, height) in enumerate(((10, 8), (12, 7), (9, 11), (14, 6))):
        x = index * 20
        children.append(
            {
                "name": f"TOP.B{index:04d}",
                "module_name": f"BLOCK_{index:04d}",
                "direction": 0,
                "color": "#000000",
                "vertex": [[x, 0], [x + width, 0], [x + width, height], [x, height]],
                "children": [],
            }
        )
    return {
        "name": "TOP",
        "module_name": "TOP",
        "direction": 0,
        "color": "#000000",
        "vertex": [[0, 0], [80, 0], [80, 20], [0, 20]],
        "children": children,
    }


class HierarchicalGeneratorTests(unittest.TestCase):
    def test_hierarchy_has_240_instances_and_exact_reuse_targets(self) -> None:
        for reuse, expected in ((0.60, 144), (0.95, 228)):
            result = compose_hierarchy(
                _source_block(), HierarchyConfig(f"hier-{reuse}", reuse)
            )
            report = result["report"]
            self.assertTrue(report["validation"]["valid"], report["validation"]["errors"])
            self.assertEqual(report["parent_count"], 12)
            self.assertEqual(report["leaf_count"], 228)
            self.assertEqual(report["nonroot_instance_count"], 240)
            self.assertEqual(report["reused_module_instance_count"], expected)
            self.assertEqual(report["reuse_instance_ratio"], reuse)
            self.assertTrue(all(not item["size_changed"] for item in result["lineage_modules"]))

    def test_synthesizer_hits_exact_count_and_all_archetypes(self) -> None:
        hierarchy = compose_hierarchy(
            _source_block(), HierarchyConfig("net-types", 0.60)
        )
        result = HierarchicalNetSynthesizer(
            hierarchy["block"],
            [],
            NetSynthesisConfig("net-types", 1000, "fixture"),
        ).synthesize()
        report = result["report"]
        self.assertEqual(report["pingroup_count"], 1000)
        self.assertTrue(report["validation"]["valid"], report["validation"]["errors"])
        self.assertTrue(all(report["net_type_counts"][name] > 0 for name in ARCHETYPES))
        self.assertEqual(report["validation"]["missing_successor_count"], 0)
        self.assertEqual(report["validation"]["successor_cycle_count"], 0)
        self.assertGreater(report["validation"]["gateway_net_count"], 0)

    def test_same_seed_is_deterministic_and_different_seed_changes_topology(self) -> None:
        hierarchy = compose_hierarchy(
            _source_block(), HierarchyConfig("determinism", 0.95)
        )
        config = NetSynthesisConfig("determinism", 1000, "fixture", base_seed=7)
        first = HierarchicalNetSynthesizer(hierarchy["block"], [], config).synthesize()
        second = HierarchicalNetSynthesizer(hierarchy["block"], [], config).synthesize()
        changed = HierarchicalNetSynthesizer(
            hierarchy["block"],
            [],
            NetSynthesisConfig("determinism", 1000, "fixture", base_seed=8),
        ).synthesize()
        self.assertEqual(
            json.dumps(first["pingroup"], sort_keys=True),
            json.dumps(second["pingroup"], sort_keys=True),
        )
        self.assertNotEqual(
            json.dumps(first["pingroup"], sort_keys=True),
            json.dumps(changed["pingroup"], sort_keys=True),
        )

    def test_validator_rejects_missing_successor_and_cycle(self) -> None:
        hierarchy = compose_hierarchy(
            _source_block(), HierarchyConfig("negative", 0.60)
        )
        synthesized = HierarchicalNetSynthesizer(
            hierarchy["block"], [], NetSynthesisConfig("negative", 1000, "fixture")
        ).synthesize()
        missing = deepcopy(synthesized["pingroup"])
        missing[0][0]["successors"] = ["TOP.MISSING.pin"]
        report = validate_hierarchical_nets(
            missing, synthesized["lineage_nets"], 1000
        )
        self.assertFalse(report["valid"])
        self.assertGreater(report["missing_successor_count"], 0)
        cycle = deepcopy(synthesized["pingroup"])
        cycle[0][1]["successors"] = [
            f"{cycle[0][0]['parent_inst']}.{cycle[0][0]['pingroup_name']}"
        ]
        report = validate_hierarchical_nets(
            cycle, synthesized["lineage_nets"], 1000
        )
        self.assertFalse(report["valid"])
        self.assertGreater(report["successor_cycle_count"], 0)

    def test_capacity_calibration_integrates_without_overflow(self) -> None:
        config = HierarchicalCaseConfig(
            case_id="integrated-1k",
            target_pingroup_count=1000,
            target_reuse_rate=0.60,
            source_sample_id="fixture",
        )
        first = build_hierarchical_case(_source_block(), [], config)
        second = build_hierarchical_case(_source_block(), [], config)
        self.assertTrue(first["validation"]["valid"], first["validation"]["errors"])
        self.assertGreaterEqual(first["capacity_report"]["small_group_ratio"], 0.75)
        self.assertEqual(first["capacity_report"]["packing"]["rejected_group_count"], 0)
        self.assertEqual(first["capacity_report"]["packing"]["hard_overflow_count"], 0)
        self.assertEqual(
            json.dumps(first["block"], sort_keys=True),
            json.dumps(second["block"], sort_keys=True),
        )
        self.assertEqual(
            json.dumps(first["pingroup"], sort_keys=True),
            json.dumps(second["pingroup"], sort_keys=True),
        )

    def test_reused_homology_instances_have_aligned_dangling_and_remapped_connections(self) -> None:
        hierarchy = compose_hierarchy(
            _source_block(), HierarchyConfig("reuse-semantics", 0.60)
        )
        result = HierarchicalNetSynthesizer(
            hierarchy["block"],
            [],
            NetSynthesisConfig("reuse-semantics", 1000, "fixture"),
        ).synthesize()
        report = result["reuse_connectivity_report"]
        self.assertGreater(report["pattern_template_counts"]["aligned_pair"], 0)
        self.assertGreater(report["pattern_template_counts"]["dangling_counterpart"], 0)
        self.assertGreater(report["pattern_template_counts"]["remapped_counterpart"], 0)
        self.assertEqual(
            report["expected_singleton_net_count"], report["actual_singleton_net_count"]
        )
        self.assertEqual(report["full_pin_multi_net_violation_count"], 0)
        self.assertEqual(report["homology_width_mismatch_count"], 0)
        self.assertGreater(report["remapped_endpoint_counts"]["local"], 0)
        self.assertGreater(report["remapped_endpoint_counts"]["cross_parent"], 0)
        self.assertGreater(report["remapped_endpoint_counts"]["gateway"], 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            block_path = Path(temp_dir) / "block.json"
            pingroup_path = Path(temp_dir) / "pingroup.json"
            block_path.write_text(json.dumps(hierarchy["block"]), encoding="utf-8")
            pingroup_path.write_text(json.dumps(result["pingroup"]), encoding="utf-8")
            placedb = PlaceDB(str(block_path), str(pingroup_path))
            homology = HomologyManager(placedb)
            aligned = next(
                record
                for record in result["lineage_nets"]
                if record.get("reuse_pattern") == "aligned_pair"
                and record.get("reuse_role") == "primary"
            )
            primary_net = result["pingroup"][aligned["derived_net_index"]]
            group_key = (
                f"{primary_net[0]['parent_module']}.{primary_net[0]['pingroup_name']}"
            )
            group = homology.pin_groups[group_key]
            related = homology.get_related_nets(group)
            self.assertEqual(len(group.pins), 2)
            self.assertEqual(len(related), 2)
            self.assertEqual(len({pin.parent_inst for pin in group.pins}), 2)

    def test_only_lineage_declared_singletons_are_accepted(self) -> None:
        hierarchy = compose_hierarchy(
            _source_block(), HierarchyConfig("singleton-contract", 0.60)
        )
        result = HierarchicalNetSynthesizer(
            hierarchy["block"],
            [],
            NetSynthesisConfig("singleton-contract", 1000, "fixture"),
        ).synthesize()
        singleton_indices = {
            index for index, net in enumerate(result["pingroup"]) if len(net) == 1
        }
        self.assertTrue(singleton_indices)
        strict = validate_artifacts(hierarchy["block"], result["pingroup"])
        self.assertFalse(strict["valid"])
        allowed = validate_artifacts(
            hierarchy["block"],
            result["pingroup"],
            allowed_singleton_net_indices=singleton_indices,
        )
        self.assertTrue(allowed["valid"], allowed["errors"])
        self.assertEqual(allowed["accepted_singleton_net_count"], len(singleton_indices))

        broken_lineage = deepcopy(result["lineage_nets"])
        singleton_index = min(singleton_indices)
        broken_lineage[singleton_index].pop("reason")
        validation = validate_hierarchical_nets(
            result["pingroup"], broken_lineage, 1000
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(
            any(error["code"] == "invalid_singleton_reason" for error in validation["errors"])
        )

    def test_shared_seed_namespace_aligns_phase_seeds_across_reuse_tiers(self) -> None:
        left_hierarchy = HierarchyConfig(
            "day5-r60", 0.60, seed_namespace="day5-20k"
        )
        right_hierarchy = HierarchyConfig(
            "day5-r95", 0.95, seed_namespace="day5-20k"
        )
        self.assertEqual(left_hierarchy.hierarchy_seed, right_hierarchy.hierarchy_seed)
        self.assertEqual(left_hierarchy.reuse_seed, right_hierarchy.reuse_seed)
        left_nets = NetSynthesisConfig(
            "day5-r60", 1000, "fixture", seed_namespace="day5-20k"
        )
        right_nets = NetSynthesisConfig(
            "day5-r95", 1000, "fixture", seed_namespace="day5-20k"
        )
        self.assertEqual(left_nets.topology_seed, right_nets.topology_seed)
        self.assertEqual(left_nets.width_seed, right_nets.width_seed)

    def test_day6_capacity_projection_checks_configured_and_all_small_bounds(self) -> None:
        report = {
            "packing": {
                "packing_utilization_limit": 0.90,
                "module_types": {
                    "A": {"capacity": 10.0},
                    "B": {"capacity": 20.0},
                },
            },
            "groups": {
                "A.small": {
                    "module_name": "A",
                    "width": 0.2,
                    "pressure": 0.02,
                    "bucket": "small",
                },
                "B.large": {
                    "module_name": "B",
                    "width": 1.0,
                    "pressure": 0.10,
                    "bucket": "large",
                },
            },
        }
        projection = capacity_projection(
            report, source_group_count=10, target_group_count=100
        )
        self.assertEqual(projection["linear_scale"], 10.0)
        self.assertGreater(
            projection["configured_minimum_max_module_utilization"],
            projection["all_small_absolute_optimistic_max_module_utilization"],
        )
        self.assertEqual(projection["configured_minimum_overloaded_module_count"], 0)


if __name__ == "__main__":
    unittest.main()
