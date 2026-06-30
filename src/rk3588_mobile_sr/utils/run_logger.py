"""Loguru-based training run logger (rank 0 file + optional TTY console)."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}"


def setup_run_logger(
    save_dir: Path | str,
    rank: int,
    *,
    level: str = "INFO",
) -> None:
    """Configure loguru for a training run.

    Rank 0 writes to ``{save_dir}/train.log`` (truncated each run) and mirrors
  to stderr when it is a TTY. Other ranks remove all handlers so nothing is
    emitted.
    """
    logger.remove()
    if rank != 0:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        save_dir / "train.log",
        format=_LOG_FORMAT,
        level=level,
        mode="w",
        encoding="utf-8",
        enqueue=True,
    )
    if sys.stderr.isatty():
        logger.add(
            sys.stderr,
            format=_LOG_FORMAT,
            level=level,
            colorize=True,
            enqueue=True,
        )

    logger.info("=== training run started ===")


__all__ = ["logger", "setup_run_logger"]
