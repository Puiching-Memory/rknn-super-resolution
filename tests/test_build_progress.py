"""Tests for build progress reporting."""

from __future__ import annotations

from rk3588_mobile_sr.data_pipeline.build_progress import (
    BuildCounts,
    BuildTargets,
    format_progress_line,
)


def test_format_progress_line_includes_bar_and_counts() -> None:
    line = format_progress_line(
        counts=BuildCounts(mezzanine=400, scaled=3000, codec=50000),
        targets=BuildTargets(mezzanine=816, scaled=6528, codec=97920),
        started_at=0.0,
        now=100.0,
    )
    assert "build" in line
    assert ("█" in line or "░" in line)
    assert "mezz  400/816" in line
    assert "scaled  3000/6528" in line
    assert "codec  50000/97920" in line
    assert "elapsed 1:40" in line
