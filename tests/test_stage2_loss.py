"""Tests for Stage-2 combined loss."""

import torch

from rk3588_mobile_sr.losses import Stage2Loss


def test_stage2_loss_components_sum_to_total():
    criterion = Stage2Loss(lambda_dct=0.02, lambda_dists=0.05, lambda_kd=0.03)
    pred = torch.rand(2, 3, 32, 32, requires_grad=True) * 255.0
    hr = torch.rand(2, 3, 32, 32) * 255.0
    teacher = torch.rand(2, 3, 32, 32) * 255.0

    out = criterion(pred, hr, teacher)
    expected = out.charbonnier + 0.02 * out.dct + 0.05 * out.dists + 0.03 * out.kd
    assert torch.allclose(out.total, expected)

    logged = out.log_dict()
    assert "train/loss_charbonnier" in logged
    assert "train/loss_dists" in logged
    assert "train/loss_dists_weighted" in logged
    assert "train/loss_dct_weighted" in logged
    assert "train/loss_kd_weighted" in logged
    assert logged["train/loss_total"] == float(out.total.detach())
