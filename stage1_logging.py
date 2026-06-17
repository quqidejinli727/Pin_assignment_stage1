"""File logging helpers for Stage 1 runs."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def configure_stage1_file_logging(output_dir: str | Path, prefix: str = "stage1_mcts") -> Path:
    """Attach a timestamped file handler for Stage 1 assignment logs."""
    log_dir = Path(output_dir) / "stage1_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    log_path = log_dir / f"{prefix}_{timestamp}.log"

    logger = logging.getLogger("assignment_solver")
    for handler in list(logger.handlers):
        if getattr(handler, "_stage1_file_handler", False):
            logger.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handler._stage1_file_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.info("stage1_log_path=%s", log_path)
    return log_path


def close_stage1_file_logging() -> None:
    """Close Stage 1 file handlers so Windows can release the log file."""
    logger = logging.getLogger("assignment_solver")
    for handler in list(logger.handlers):
        if getattr(handler, "_stage1_file_handler", False):
            logger.removeHandler(handler)
            handler.close()
