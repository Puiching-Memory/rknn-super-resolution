"""Tests for SR validation metrics."""

import torch

from rknn_super_resolution.utils.sr_metrics import (
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


def test_validation_metrics_to_log_dict():
    metrics = ValidationMetrics(
        psnr=30.0,
        y_psnr=31.0,
        ssim=0.9,
        l1=2.5,
        psnr_min=28.0,
        psnr_p10=29.0,
        psnr_p50=30.0,
        psnr_p90=31.0,
        vmaf=72.5,
    )
    logged = metrics.to_log_dict()
    assert logged["val/psnr"] == 30.0
    assert logged["val/vmaf"] == 72.5


def test_y_psnr_runs_on_batch():
    pred = torch.rand(2, 3, 24, 24) * 255.0
    target = pred.clone()
    y_psnr = batch_y_psnr(pred, target, shave=3)
    assert y_psnr.shape == (2,)
    assert torch.all(y_psnr > 99.0)


def test_validate_ddp_extended_yuv_uses_luma_and_skips_vmaf():
    import torch.nn as nn

    from rknn_super_resolution.data.yuv_utils import rgb_to_yuv444
    from rknn_super_resolution.utils.sr_metrics import validate_ddp_extended

    class _Passthrough(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("_device", torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class _Loader:
        dataset = [0]
        sampler = None

        def __init__(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
            self._batch = batch

        def __iter__(self):
            yield self._batch

        def __len__(self) -> int:
            return 1

    class _ForwardGuard(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module = _Passthrough()
            self.forward_calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return self.module(x)

    rgb = torch.zeros(1, 3, 24, 24)
    rgb[:, 0] = 200.0
    yuv = rgb_to_yuv444(rgb)
    model = _ForwardGuard()
    score, metrics = validate_ddp_extended(
        model,
        _Loader((yuv, yuv)),
        rank=0,
        world_size=1,
        scale=3,
        compute_vmaf=False,
        colorspace="yuv",
    )
    assert metrics is not None
    assert score == metrics.psnr
    assert metrics.psnr > 99.0
    assert metrics.y_psnr > 99.0
    assert metrics.vmaf is None
    assert model.forward_calls == 1
