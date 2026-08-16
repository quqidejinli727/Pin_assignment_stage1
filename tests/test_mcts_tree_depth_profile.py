from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from PlaceDB import PlaceDB
from homology import HomologyManager

from floorset_benchmark.hierarchical_net_synthesizer import (
    HierarchicalNetSynthesizer,
    NetSynthesisConfig,
    validate_hierarchical_nets,
)
from floorset_benchmark.hierarchy_composer import HierarchyConfig, compose_hierarchy
from floorset_benchmark.mcts_tree_depth_analysis import analyze_mcts_tree_depths


def _source_block() -> dict:
    return {
        "name": "TOP",
        "module_name": "TOP",
        "direction": 0,
        "color": "#000000",
        "vertex": [[0, 0], [80, 0], [80, 20], [0, 20]],
        "children": [
            {
                "name": f"TOP.B{index:04d}",
                "module_name": f"BLOCK_{index:04d}",
                "direction": 0,
                "color": "#000000",
                "vertex": [[index * 20, 0], [index * 20 + width, 0], [index * 20 + width, height], [index * 20, height]],
                "children": [],
            }
            for index, (width, height) in enumerate(((10, 8), (12, 7), (9, 11), (14, 6)))
        ],
    }


class MCTSTreeDepthProfileTests(unittest.TestCase):
    def test_mixed_reuse_clusters_preserve_exact_reuse_targets(self) -> None:
        for reuse, expected in ((0.60, 144), (0.95, 228)):
            result = compose_hierarchy(
                _source_block(),
                HierarchyConfig(
                    f"clusters-{reuse}",
                    reuse,
                    reuse_cluster_strategy="mixed2to10",
                ),
            )
            report = result["report"]
            self.assertEqual(report["reused_module_instance_count"], expected)
            sizes = {int(size) for size in report["reuse_cluster_size_histogram"]}
            self.assertTrue(sizes)
            self.assertTrue(all(2 <= size <= 10 for size in sizes))
            self.assertTrue(any(size > 2 for size in sizes))

    def _profile_fixture(self) -> tuple[dict, dict]:
        hierarchy = compose_hierarchy(
            _source_block(),
            HierarchyConfig(
                "tree-profile",
                0.95,
                reuse_cluster_strategy="mixed2to10",
            ),
        )
        synthesized = HierarchicalNetSynthesizer(
            hierarchy["block"],
            [],
            NetSynthesisConfig(
                "tree-profile",
                1000,
                "fixture",
                reuse_template_fraction=0.01,
                mcts_tree_depth_profile=True,
            ),
        ).synthesize()
        return hierarchy, synthesized

    def test_cross_net_homology_produces_three_group_runtime_tree(self) -> None:
        hierarchy, synthesized = self._profile_fixture()
        tree = next(
            item
            for item in synthesized["cross_net_homology_report"]["trees"]
            if item["target_effective_depth"] == 3
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            block_path = Path(temp_dir) / "block.json"
            pin_path = Path(temp_dir) / "pingroup.json"
            block_path.write_text(json.dumps(hierarchy["block"]), encoding="utf-8")
            pin_path.write_text(json.dumps(synthesized["pingroup"]), encoding="utf-8")
            placedb = PlaceDB(str(block_path), str(pin_path))
            homology = HomologyManager(placedb, use_fanout_reuse_for_sorting=False)
            hub = homology.pin_groups[tree["hub_homology_group"]]
            nets = homology.get_related_nets(hub)
            groups = homology.groups_for_pins(pin for net in nets for pin in net.pins)
            self.assertEqual(len(nets), 2)
            self.assertEqual(len(groups), 3)

    def test_aligned_two_net_deduplicates_both_homology_sides(self) -> None:
        hierarchy, synthesized = self._profile_fixture()
        tree = next(
            item
            for item in synthesized["cross_net_homology_report"]["trees"]
            if item["template"] == "aligned_two_net"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            block_path = Path(temp_dir) / "block.json"
            pin_path = Path(temp_dir) / "pingroup.json"
            block_path.write_text(json.dumps(hierarchy["block"]), encoding="utf-8")
            pin_path.write_text(json.dumps(synthesized["pingroup"]), encoding="utf-8")
            placedb = PlaceDB(str(block_path), str(pin_path))
            homology = HomologyManager(placedb, use_fanout_reuse_for_sorting=False)
            hub = homology.pin_groups[tree["hub_homology_group"]]
            nets = homology.get_related_nets(hub)
            groups = homology.groups_for_pins(pin for net in nets for pin in net.pins)
            self.assertEqual(len(nets), 2)
            self.assertEqual(len(groups), 2)

    def test_runtime_profile_and_capacity_deduction_semantics(self) -> None:
        hierarchy, synthesized = self._profile_fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            block_path = Path(temp_dir) / "block.json"
            pin_path = Path(temp_dir) / "pingroup.json"
            block_path.write_text(json.dumps(hierarchy["block"]), encoding="utf-8")
            pin_path.write_text(json.dumps(synthesized["pingroup"]), encoding="utf-8")
            report = analyze_mcts_tree_depths(block_path, pin_path)
        effective = report["effective_mcts_depth"]
        self.assertGreaterEqual(effective["buckets"]["depth_2_10"]["ratio"], 0.60)
        self.assertGreater(effective["buckets"]["depth_11_20"]["count"], 0)
        self.assertGreater(effective["buckets"]["depth_21_100"]["count"], 0)
        self.assertEqual(effective["buckets"]["depth_gt_100"]["count"], 0)
        self.assertEqual(
            report["capacity_deduction"]["deduction_count"],
            report["capacity_deduction"]["homology_group_count"],
        )
        self.assertFalse(report["batch_planner"]["consumed_by_assignment_solver"])

    def test_templates_have_real_gateway_branch_and_deferred_semantics(self) -> None:
        _, synthesized = self._profile_fixture()
        records = synthesized["lineage_nets"]
        gateway = [
            record
            for record in records
            if record.get("tree_template") == "hierarchical_gateway_tree"
        ]
        branched = [
            record
            for record in records
            if record.get("tree_template") == "multi_net_branched"
        ]
        deferred = [
            record
            for record in records
            if record.get("tree_template") == "deferred_homology_tree"
        ]
        self.assertTrue(any(record["gateway_path"] for record in gateway))
        self.assertGreaterEqual(len({record["derived_net_index"] for record in branched}), 2)
        self.assertTrue(any(record.get("net_kind") == "dangling_singleton" for record in deferred))

    def test_profile_is_deterministic_and_rejects_full_pin_reuse(self) -> None:
        _, first = self._profile_fixture()
        _, second = self._profile_fixture()
        self.assertEqual(
            json.dumps(first["pingroup"], sort_keys=True),
            json.dumps(second["pingroup"], sort_keys=True),
        )
        broken = deepcopy(first["pingroup"])
        broken[1].append(deepcopy(broken[0][0]))
        validation = validate_hierarchical_nets(
            broken, first["lineage_nets"], expected_pingroup_count=1000
        )
        self.assertFalse(validation["valid"])
        self.assertTrue(
            any(error["code"] == "duplicate_pin_full_name" for error in validation["errors"])
        )


if __name__ == "__main__":
    unittest.main()
