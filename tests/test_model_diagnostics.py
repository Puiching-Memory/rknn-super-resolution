"""Tests for Phase-RLFN training diagnostics."""

import torch

from rknn_super_resolution.models import PhaseRLFNSR
from rknn_super_resolution.utils.model_diagnostics import (
    ForwardDiagnosticsTracker,
    check_deploy_consistency,
    collect_grad_norms,
    collect_param_norms,
)


def test_forward_tracker_reports_skip_and_clip_stats() -> None:
    model = PhaseRLFNSR(num_blocks=2, num_channels=8)
    tracker = ForwardDiagnosticsTracker(model)
    try:
        model(torch.rand(1, 3, 16, 16) * 255.0)
        stats = tracker.read_stats()
        assert "model/skip_ratio" in stats
        assert "model/clip_sat_low" in stats
        assert "model/residual_abs_mean" in stats
    finally:
        tracker.close()


def test_grad_and_weight_norm_groups() -> None:
    model = PhaseRLFNSR(num_blocks=2, num_channels=8)
    model(torch.rand(1, 3, 16, 16) * 255.0).mean().backward()
    assert "grad_norm/stem" in collect_grad_norms(model)
    assert "grad_norm/residual_head" in collect_grad_norms(model)
    assert "weight_norm/block_0" in collect_param_norms(model, prefix="weight_norm")


def test_deploy_consistency_is_exact_with_codec_input() -> None:
    model = PhaseRLFNSR(num_blocks=1, num_channels=8).eval()
    model_input = (
        torch.rand(1, 3, 16, 16) * 255.0,
        torch.randn(1, 96, 2, 2),
    )
    loader = [(model_input, torch.rand(1, 3, 48, 48) * 255.0)]
    metrics = check_deploy_consistency(model, loader, torch.device("cpu"), max_batches=1)
    assert metrics["deploy/max_abs_diff"] == 0.0
