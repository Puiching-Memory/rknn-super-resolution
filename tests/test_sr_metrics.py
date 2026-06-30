"""Tests for SR validation metrics."""

import torch

from rk3588_mobile_sr.utils.sr_metrics import (
    ValidationMetrics,
    batch_psnr,
    batch_y_psnr,
    shave_borders,
)


def test_identical_tensors_have_high_psnr():
    x = torch.ones(2, 3, 32, 32) * 128.0
    psnr = batch_psnr(x, x, shave=0)
    assert torch.all(psnr > 99.0)


def test_shave_borders_reduces_spatial_size():
    x = torch.zeros(1, 3, 16, 16)
    shaved = shave_borders(x, shave=3)
    assert shaved.shape == (1, 3, 10, 10)


def test_validation_metrics_to_log_dict_includes_fixed():
    metrics = ValidationMetrics(
        psnr=30.0,
        y_psnr=31.0,
        ssim=0.9,
        l1=2.5,
        psnr_min=28.0,
        psnr_p10=29.0,
        psnr_p50=30.0,
        psnr_p90=31.0,
        fixed_psnr={0: 29.5, 1: 30.2},
    )
    logged = metrics.to_log_dict()
    assert logged["val/psnr"] == 30.0
    assert logged["val/fixed_0_psnr"] == 29.5
    assert logged["val/fixed_1_psnr"] == 30.2


def test_y_psnr_runs_on_batch():
    pred = torch.rand(2, 3, 24, 24) * 255.0
    target = pred.clone()
    y_psnr = batch_y_psnr(pred, target, shave=3)
    assert y_psnr.shape == (2,)
    assert torch.all(y_psnr > 99.0)
