"""验证裁剪配置与接口格式导出结构。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assignment_solver import AssignmentSolver  # noqa: E402
from export_final_result import build_interface_result, write_config_record, write_interface_result  # noqa: E402
from segment_subdivision import percentile_edge_length, subdivision_specs  # noqa: E402


def build_small_case(directory: Path) -> tuple[Path, Path]:
    """生成能触发长边裁剪的两个复用矩形 block。"""
    block = {
        "name": "TOP",
        "module_name": "TOP",
        "direction": 0,
        "color": "#000000",
        "vertex": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "children": [
            {
                "name": "TOP.U_A0",
                "module_name": "A",
                "direction": 0,
                "color": "#000000",
                "vertex": [[0, 0], [30, 0], [30, 5], [0, 5]],
                "children": [],
            },
            {
                "name": "TOP.U_A1",
                "module_name": "A",
                "direction": 1,
                "color": "#000000",
                "vertex": [[70, 30], [40, 30], [40, 25], [70, 25]],
                "children": [],
            },
        ],
    }
    pingroup = [[
        {
            "parent_inst": "TOP.U_A0",
            "parent_module": "A",
            "pingroup_name": "p",
            "scope": [],
            "successors": [],
            "width": 1.0,
        },
        {
            "parent_inst": "TOP.U_A1",
            "parent_module": "A",
            "pingroup_name": "p",
            "scope": [],
            "successors": [],
            "width": 1.0,
        },
    ]]
    block_path = directory / "block.json"
    pingroup_path = directory / "pingroup.json"
    block_path.write_text(json.dumps(block), encoding="utf-8")
    pingroup_path.write_text(json.dumps(pingroup), encoding="utf-8")
    return block_path, pingroup_path


def main() -> None:
    """执行小用例并检查裁剪与接口输出中的关键一致性。"""
    with tempfile.TemporaryDirectory() as temporary_dir:
        block_path, pingroup_path = build_small_case(Path(temporary_dir))
        threshold = percentile_edge_length(block_path, 50)
        assert threshold > 0
        assert len(subdivision_specs([(0, 0), (30, 0), (30, 5), (0, 5)], 10)) > 4

        solver = AssignmentSolver(
            str(block_path),
            str(pingroup_path),
            simulations=8,
            enable_segment_subdivision=True,
            segment_length_percentile=25,
        )
        solver.solve()
        result = build_interface_result(solver.placedb, solver.homology, solver.segment_manager)

        assert len(solver.segment_manager.abstract_segments) > 4
        assert result["total_segments"] == len(result["segment_assignments"])
        assert result["total_segments"] == 1
        pins = []
        block_ids = {}
        for segment in result["segment_assignments"].values():
            assert "segment_info" in segment and "segment_insts" in segment
            segment_info = segment["segment_info"]
            expected_segment_direction = (
                1 if segment_info["y1"] == segment_info["y2"] else 0
            )
            assert segment["direction"] == expected_segment_direction
            assert segment_info["direction"] == expected_segment_direction
            assert any(
                instance["assigned_pins"] for instance in segment["segment_insts"].values()
            )
            for instance in segment["segment_insts"].values():
                expected_instance_direction = (
                    1 if instance["coordinates"][1] == instance["coordinates"][3] else 0
                )
                assert instance["direction"] == expected_instance_direction
                block_ids.setdefault(instance["block_name"], instance["block_id"])
                assert block_ids[instance["block_name"]] == instance["block_id"]
                pins.extend(instance["assigned_pins"])
        assert {pin["name"] for pin in pins} == set(solver.placedb.pin_dict)
        assert len({pin["net_id"] for pin in pins}) == 1
        assert len({pin["isomorphic_group_id"] for pin in pins}) == 1
        timestamp = "20260609010203000000"
        output_dir = Path(temporary_dir)
        result_path = write_interface_result(
            output_dir,
            solver.placedb,
            solver.homology,
            solver.segment_manager,
            timestamp=timestamp,
        )
        config_path = write_config_record(
            output_dir,
            {"simulations": 8},
            timestamp,
            result_path=result_path,
        )
        assert result_path.name == f"segment_assignments_{timestamp}.json"
        assert config_path.name == f"stage1_config_{timestamp}.json"
        config_record = json.loads(config_path.read_text(encoding="utf-8"))
        assert config_record["interface_result_path"] == str(result_path)
        print(f"Final result export validation passed: {result['total_segments']} segments exported.")


if __name__ == "__main__":
    main()
