"""Command-line entry point for MCTS pin assignment."""

from __future__ import annotations

import argparse
import json

from assignment_solver import AssignmentSolver


def parse_args() -> argparse.Namespace:
    """解析命令行参数，包括输入 JSON、输出路径和 MCTS 参数。"""
    parser = argparse.ArgumentParser(description="Run MCTS pin-to-segment assignment.")
    parser.add_argument("--block", required=True, help="Path to block JSON file.")
    parser.add_argument("--pingroup", required=True, help="Path to pingroup JSON file.")
    parser.add_argument("--output", required=True, help="Path to output assignment JSON.")
    parser.add_argument("--simulations", type=int, default=None, help="MCTS simulations per local tree.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--strict-capacity",
        action="store_true",
        help="Disable final overflow fallback for groups wider than every candidate segment.",
    )
    return parser.parse_args()


def main() -> None:
    """命令行主函数：创建求解器、执行分配并打印 summary。"""
    args = parse_args()
    solver = AssignmentSolver(
        block_json_path=args.block,
        pingroup_json_path=args.pingroup,
        simulations=args.simulations,
        random_seed=args.seed,
        allow_overflow_fallback=not args.strict_capacity,
    )
    result = solver.write_output(args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
