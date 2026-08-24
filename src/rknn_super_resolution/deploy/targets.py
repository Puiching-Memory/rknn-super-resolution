"""RKNN deployment target metadata."""

from __future__ import annotations

TESTED_RKNN_TARGETS = ("rk3576", "rk3588", "rv1126b")


def normalize_rknn_target(value: str) -> str:
    """Normalize a non-empty RKNN target name without restricting the toolkit."""
    target = value.strip().lower()
    if not target:
        raise ValueError("RKNN target cannot be empty")
    return target
