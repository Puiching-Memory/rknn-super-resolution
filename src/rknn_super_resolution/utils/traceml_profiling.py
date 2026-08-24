"""TraceML training performance diagnostics helpers."""

from __future__ import annotations

import argparse
import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch.nn as nn

from rknn_super_resolution.utils.run_logger import logger
from rknn_super_resolution.utils.swanlab_logging import log_metrics

_active = False


def add_traceml_args(parser: argparse.ArgumentParser) -> None:
    """Register TraceML CLI flags on a training argument parser."""
    parser.add_argument(
        "--traceml",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable TraceML step tracing (default: on when launched via `traceml run`)",
    )


def _traceml_requested(args: argparse.Namespace) -> bool:
    explicit = getattr(args, "traceml", None)
    if explicit is not None:
        return explicit
    return bool(os.environ.get("TRACEML_SESSION_ID", "").strip())


def setup_traceml(args: argparse.Namespace, *, rank: int) -> None:
    """Initialize TraceML instrumentation when enabled for this run."""
    global _active
    _active = False
    if not _traceml_requested(args):
        return

    import traceml_ai as traceml

    traceml.init(mode="auto")
    _active = True
    if rank == 0:
        session_id = os.environ.get("TRACEML_SESSION_ID", "").strip()
        if session_id:
            logs_dir = os.environ.get("TRACEML_LOGS_DIR", "logs")
            logger.info("TraceML enabled (session={}, logs={}).", session_id, logs_dir)
        else:
            logger.info(
                "TraceML instrumentation enabled. Launch with "
                "`traceml run ...` to write final_summary.json at end of run."
            )


def trace_training_step(model: nn.Module) -> AbstractContextManager[None]:
    """Return a TraceML step boundary context manager, or a no-op when disabled."""
    if not _active:
        return nullcontext()
    import traceml_ai as traceml

    return traceml.trace_step(model)


def finish_traceml(*, rank: int) -> None:
    """Publish TraceML summary metrics and print the text report on rank 0."""
    if not _active or rank != 0:
        return

    import traceml_ai as traceml

    try:
        summary = traceml.summary(print_text=True, rank0_only=True)
    except RuntimeError as exc:
        logger.warning("TraceML summary unavailable: {}", exc)
        return

    if not summary:
        return

    flat_metrics = _flatten_summary(summary)
    if flat_metrics:
        log_metrics(flat_metrics)


def _flatten_summary(value: Any, prefix: str = "traceml") -> dict[str, float | int | str]:
    """Flatten nested TraceML summary dicts for experiment trackers."""
    metrics: dict[str, float | int | str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else key
            metrics.update(_flatten_summary(child, child_prefix))
    elif isinstance(value, bool):
        metrics[prefix] = int(value)
    elif isinstance(value, (int, float, str)):
        metrics[prefix] = value
    return metrics
