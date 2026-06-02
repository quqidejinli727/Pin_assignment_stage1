"""Stage 1 adapter used by the top-level PinAssignFlow ``run.py`` script."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from config import DEFAULT_CONFIG
from main import run_pipline


def run_mcts(
    block_json: str,
    pingroup_json: str,
    output_dir: str,
    num_simulations: int = 1000,
    time_limit: float | None = None,
) -> str:
    """Run MCTS segment assignment with the interface expected by ``run.py``.

    ``time_limit`` is accepted for compatibility with the top-level pipeline.
    The current Stage 1 implementation is controlled by simulation count.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if time_limit is not None:
        logging.info(
            "Stage 1 time_limit=%.1fs accepted for compatibility; "
            "current MCTS uses num_simulations.",
            time_limit,
        )

    config = replace(
        DEFAULT_CONFIG,
        block_json_path=Path(block_json),
        pingroup_json_path=Path(pingroup_json),
        assignment_output_path=output_path / "stage1_assignment.json",
        results_root=output_path / "stage1_run_results",
        interface_result_dir=output_path,
        simulations=num_simulations,
        enable_feedthrough=False,
        export_interface_result=True,
    )
    return str(run_pipline(config))
