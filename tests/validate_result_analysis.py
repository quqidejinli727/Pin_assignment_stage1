"""使用合成 pingroup 摆放验证 final_result 与 case 的比较分析逻辑。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_final_vs_case import (  # noqa: E402
    compare_nets,
    load_analysis_nets,
    metric_summary,
    write_analysis_report,
    write_selected_groups,
)


def pin(parent_inst: str, pingroup_name: str, scope: list, width: float = 1) -> dict:
    """生成最小可解析的 pingroup Pin 记录。"""
    return {
        "parent_inst": parent_inst,
        "parent_module": "UNIT",
        "pingroup_name": pingroup_name,
        "scope": scope,
        "successors": [],
        "width": width,
    }


def main() -> None:
    """验证 HPWL 计算、按 Pin 匹配、分组统计及文本输出。"""
    final = [
        [pin("TOP.A", "p", [[-10, 0], [10, 0]], 2), pin("TOP.B", "p", [[90, 0], [110, 0]], 3)],
        [pin("TOP.C", "p", [[-20, 0], [20, 0]]), pin("TOP.D", "p", [[480, 0], [520, 0]])],
        [pin("TOP.E", "p", [[0, 0], [20, 0]]), pin("TOP.F", "p", [[40, 0], [60, 0]])],
        [pin("TOP.S", "p", [])],
    ]
    # 故意调整 net 顺序，确保两份文件按 Pin 集合匹配而非仅依赖数组下标。
    case = [
        [pin("TOP.C", "p", [0, 0]), pin("TOP.D", "p", [100, 0])],
        [pin("TOP.A", "p", [0, 0]), pin("TOP.B", "p", [500, 0])],
        [pin("TOP.E", "p", []), pin("TOP.F", "p", [50, 0])],
        [pin("TOP.S", "p", [])],
    ]
    with tempfile.TemporaryDirectory() as temporary_dir:
        directory = Path(temporary_dir)
        final_path = directory / "final.json"
        case_path = directory / "case.json"
        final_path.write_text(json.dumps(final), encoding="utf-8")
        case_path.write_text(json.dumps(case), encoding="utf-8")
        final_nets, case_nets, skipped_case_pin_count = load_analysis_nets(
            final_path,
            case_path,
        )
        comparisons = compare_nets(
            final_nets,
            case_nets,
            group_width=300,
            final_feedthrough_by_id={0: 1.0, 1: 3.0},
            case_feedthrough_by_id={0: 2.0, 1: 2.0},
        )
        summary = metric_summary(comparisons)
        assert [item.final_hpwl for item in comparisons] == [100.0, 500.0, 0.0]
        assert skipped_case_pin_count == 2
        assert all("TOP.E.p" not in item.key and "TOP.F.p" not in item.key for item in comparisons)
        assert summary["optimized_net_count"] == 1
        assert summary["worsened_net_count"] == 1
        assert summary["final_average_feedthrough"] == 2.0
        assert {item.group for item in comparisons} == {"[-600, -300)", "[300, 600)", "= 0"}

        report = directory / "report.txt"
        selected = directory / "selected.txt"
        write_analysis_report(
            report,
            final_path,
            case_path,
            300,
            comparisons,
            skipped_case_pin_count=skipped_case_pin_count,
        )
        write_selected_groups(selected, comparisons, ["[-600, -300)"])
        report_text = report.read_text(encoding="utf-8")
        selected_text = selected.read_text(encoding="utf-8")
        assert "[[x1, y1], [x2, y2]] (analyzed at midpoint)" in report_text
        assert "final_average_feedthrough: 2.000000" in report_text
        assert "skipped_case_pin_count: 2" in report_text
        assert "TOP.A.p  width=2  reuse_count=7" in report_text
        assert "TOP.B.p  width=3  reuse_count=7" in report_text
        assert "TOP.S.p  width=1  reuse_count=7" in report_text
        assert "TOP.E.p" not in report_text and "TOP.F.p" not in report_text
        assert "TOP.A.p" in selected_text and "TOP.B.p" in selected_text
        print("Result analysis validation passed.")


if __name__ == "__main__":
    main()
