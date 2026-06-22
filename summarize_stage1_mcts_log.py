"""Summarize per-tree Stage1 MCTS timing logs.

Usage:
    python summarize_stage1_mcts_log.py path/to/stage1_mcts_*.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable


FIELD_PATTERN = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>\S+)")


def _parse_value(raw: str) -> Any:
    """Parse one log field value into int/float when possible."""
    try:
        if re.fullmatch(r"[-+]?\d+", raw):
            return int(raw)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", raw):
            return float(raw)
    except ValueError:
        pass
    return raw


def parse_mcts_tree_line(line: str) -> Dict[str, Any] | None:
    """Return parsed fields for one mcts_tree log line."""
    if "mcts_tree" not in line:
        return None
    fields: Dict[str, Any] = {}
    for match in FIELD_PATTERN.finditer(line):
        fields[match.group("key")] = _parse_value(match.group("value"))
    return fields if fields else None


def _sum_numeric(records: Iterable[Dict[str, Any]], key: str) -> float:
    """Sum one numeric field across parsed records."""
    total = 0.0
    for record in records:
        value = record.get(key, 0.0)
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def summarize_log(log_path: str | Path) -> Dict[str, Any]:
    """Summarize all mcts_tree records in a Stage1 log file."""
    path = Path(log_path)
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        record = parse_mcts_tree_line(line)
        if record is not None:
            records.append(record)

    timing_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if key.endswith("_s") and isinstance(value, (int, float))
        }
    )
    count_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if key != "round"
            and not key.endswith("_s")
            and isinstance(value, (int, float))
        }
    )

    timing_totals = {key: _sum_numeric(records, key) for key in timing_keys}
    count_totals = {key: int(_sum_numeric(records, key)) for key in count_keys}
    first_round = records[0].get("round") if records else None
    last_round = records[-1].get("round") if records else None

    summary = {
        "log_path": str(path),
        "mcts_tree_count": len(records),
        "first_round": first_round,
        "last_round": last_round,
        "total_logged_tree_wall_s": timing_totals.get("search_wall_s", 0.0),
        "total_pre_first_mcts_s": timing_totals.get("pre_first_mcts_s", 0.0),
        "total_tree_build_s": timing_totals.get("tree_build_s", 0.0),
        "total_commit_s": timing_totals.get("commit_s", 0.0),
        "total_reward_s": timing_totals.get("reward_total_s", 0.0),
        "total_reward_hpwl_s": timing_totals.get("reward_hpwl_s", 0.0),
        "total_reward_ft_s": timing_totals.get("reward_ft_s", 0.0),
        "total_mcts_main_s": timing_totals.get("mcts_main_s", 0.0),
        "total_child_generation_s": timing_totals.get("child_generation_s", 0.0),
        "total_child_actions_s": timing_totals.get("child_actions_s", 0.0),
        "total_child_feasible_s": timing_totals.get("child_feasible_s", 0.0),
        "total_child_pruning_s": timing_totals.get("child_pruning_s", 0.0),
        "total_child_create_s": timing_totals.get("child_create_s", 0.0),
        "total_child_usage_clone_s": timing_totals.get(
            "child_usage_clone_s", 0.0
        ),
        "total_child_assignment_copy_s": timing_totals.get(
            "child_assignment_copy_s", 0.0
        ),
        "timing_totals_s": timing_totals,
        "count_totals": count_totals,
    }
    summary["estimated_stage1_timed_total_s"] = (
        summary["total_pre_first_mcts_s"]
        + summary["total_tree_build_s"]
        + summary["total_logged_tree_wall_s"]
        + summary["total_commit_s"]
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize Stage1 per-MCTS-tree timing log totals."
    )
    parser.add_argument("log_path", help="Path to stage1_mcts_*.log.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. If omitted, only prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    summary = summarize_log(args.log_path)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
