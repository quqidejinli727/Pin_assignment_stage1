"""Stage 1: MCTS segment assignment.

Provides the ``run_mcts`` entry point used by the top-level PinAssignFlow
``run.py`` script.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

logger = logging.getLogger(__name__)

_STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_STAGE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Stage packages may contain modules with overlapping names. Clear Stage 1
# modules so this package always reloads them from the current project root.
_STAGE1_MODULES = [
    "PlaceDB",
    "assignment_solver",
    "config",
    "export_final_result",
    "geometry_utils",
    "homology",
    "main",
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
    from config import DEFAULT_CONFIG
    from main import run_pipline

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if time_limit is not None:
        logger.info(
            "Stage 1 time_limit=%.1fs accepted for compatibility; "
            "current MCTS uses num_simulations.",
            time_limit,
        )

    config = replace(
        DEFAULT_CONFIG,
        block_json_path=Path(block_json).resolve(),
        pingroup_json_path=Path(pingroup_json).resolve(),
        assignment_output_path=output_path / "stage1_assignment.json",
        results_root=output_path / "stage1_run_results",
        interface_result_dir=output_path,
        simulations=num_simulations,
        enable_feedthrough=False,
        export_interface_result=True,
    )
    return str(run_pipline(config))
