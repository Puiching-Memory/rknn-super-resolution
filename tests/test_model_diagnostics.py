"""Tests for model training diagnostics."""

import torch

from rk3588_mobile_sr.models.mobileone_sr import MobileOneSR
from rk3588_mobile_sr.utils.model_diagnostics import (
    ForwardDiagnosticsTracker,
    check_deploy_consistency,
    collect_grad_norms,
    collect_param_norms,
)


def test_forward_tracker_reports_skip_and_clip_stats():
    model = MobileOneSR(num_blocks=2, num_channels=8)
    tracker = ForwardDiagnosticsTracker(model)
    try:
        x = torch.rand(1, 3, 32, 32) * 255.0
        model(x)
        stats = tracker.read_stats()
        assert "model/skip_ratio" in stats
        assert "model/clip_sat_low" in stats
        assert "model/clip_sat_high" in stats
        assert stats["model/skip_ratio"] >= 0.0
    finally:
        tracker.close()


def test_grad_and_weight_norm_groups():
    model = MobileOneSR(num_blocks=2, num_channels=8)
    x = torch.rand(1, 3, 32, 32) * 255.0
    out = model(x)
    out.mean().backward()

    grad_norms = collect_grad_norms(model)
    weight_norms = collect_param_norms(model, prefix="weight_norm")
    assert "grad_norm/stem" in grad_norms
    assert "grad_norm/out_conv" in grad_norms
    assert "weight_norm/body_0" in weight_norms


def test_deploy_consistency_is_near_zero():
    model = MobileOneSR(num_blocks=2, num_channels=8)
    model.eval()
    loader = [(torch.rand(1, 3, 32, 32) * 255.0, torch.rand(1, 3, 96, 96) * 255.0)]
    metrics = check_deploy_consistency(model, loader, torch.device("cpu"), max_batches=1)
    assert metrics["deploy/max_abs_diff"] < 1e-3
    assert metrics["deploy/psnr_train_vs_deploy"] > 80.0
