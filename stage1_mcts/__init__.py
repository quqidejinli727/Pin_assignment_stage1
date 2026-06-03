"""Stage 1: MCTS segment assignment.

Provides the ``run_mcts`` entry point used by the top-level PinAssignFlow
``run.py`` script.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_STAGE_DIR)
_CODE_DIR = _STAGE_DIR if os.path.exists(os.path.join(_STAGE_DIR, "assignment_solver.py")) else _PROJECT_ROOT
if _CODE_DIR in sys.path:
    sys.path.remove(_CODE_DIR)
sys.path.insert(0, _CODE_DIR)

# Stage packages may contain modules with overlapping names. Clear Stage 1
# modules so this package always reloads them from the selected Stage 1 code dir.
_STAGE1_MODULES = [
    "PlaceDB",
    "assignment_solver",
    "export_final_result",
    "geometry_utils",
    "homology",
    "mcts",
    "scoring",
    "segment",
    "segment_subdivision",
]
for _mod in _STAGE1_MODULES:
    sys.modules.pop(_mod, None)


def run_mcts(
    block_json: str,
    pingroup_json: str,
    output_dir: str,
    num_simulations: int = 1000,
    time_limit: float | None = None,
) -> str:
    """Run MCTS segment assignment.

    Args:
        block_json: Path to block.json from the benchmark case.
        pingroup_json: Path to pingroup.json from the benchmark case.
        output_dir: Directory for Stage 1 outputs.
        num_simulations: MCTS base simulation count.
        time_limit: Accepted for run.py compatibility; the current solver uses
            simulation count rather than a wall-clock limit.

    Returns:
        Path to the generated ``segment_assignments_*.json`` file.
    """
    from assignment_solver import AssignmentSolver
    from export_final_result import write_interface_result

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if time_limit is not None:
        logger.info(
            "Stage 1 time_limit=%.1fs accepted for compatibility; "
            "current MCTS uses num_simulations.",
            time_limit,
        )

    solver = AssignmentSolver(
        block_json_path=str(Path(block_json).resolve()),
        pingroup_json_path=str(Path(pingroup_json).resolve()),
        simulations=num_simulations,
    )
    assignment_result = solver.solve()
    assignment_path = output_path / "stage1_assignment.json"
    assignment_path.write_text(
        json.dumps(assignment_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(
        write_interface_result(
            output_path,
            solver.placedb,
            solver.homology,
            solver.segment_manager,
        )
    )
