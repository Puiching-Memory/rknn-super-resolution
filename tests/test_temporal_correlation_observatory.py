"""Tests for real-clip temporal correlation statistics."""

import pytest
import torch

from rknn_super_resolution.dev.temporal_correlation_observatory import analyze_bank


def test_analyze_bank_reports_improvement_and_maps() -> None:
    current = torch.rand(2, 1, 32, 32)
    history = current[:, None].repeat(1, 3, 1, 1, 1)
    metrics, maps = analyze_bank(
        current,
        history,
        ((0.0, 0.0), (-1.0, 0.0), (1.0, 0.0)),
        block_size=8,
    )
    assert metrics["best_mse"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["noncenter_ratio"] == 0.0
    assert maps["selection"].shape == (4, 4)
    assert maps["confidence"].shape == (4, 4)
