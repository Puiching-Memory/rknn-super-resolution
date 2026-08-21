"""Tests for RGB/YUV conversion utilities."""

import numpy as np
import torch

from rk3588_mobile_sr.data.yuv_utils import (
    colorspace_roundtrip_rgb,
    maybe_rgb_to_colorspace,
    rgb_to_yuv444,
    yuv444_to_rgb,
)


def test_rgb_yuv_roundtrip():
    rgb = torch.zeros(1, 3, 8, 8)
    rgb[:, 0] = 255.0
    rgb[:, 2] = 128.0
    yuv = rgb_to_yuv444(rgb)
    back = yuv444_to_rgb(yuv)
    assert yuv.shape == rgb.shape
    assert torch.allclose(back, rgb, atol=1.5)


def test_bt709_primary_coefficients():
    rgb = torch.zeros(3, 1, 1)
    rgb[0] = 255.0
    yuv = rgb_to_yuv444(rgb)
    assert torch.allclose(yuv[0], torch.tensor([[0.2126 * 255.0]]), atol=1e-4)
    assert torch.allclose(yuv[1], torch.tensor([[127.5 - 0.5 * 0.2126 * 255.0 / 0.9278]]), atol=1e-4)


def test_yuv444_to_rgb_chw():
    rgb = torch.zeros(3, 8, 8)
    rgb[0] = 255.0
    yuv = rgb_to_yuv444(rgb.unsqueeze(0)).squeeze(0)
    back = yuv444_to_rgb(yuv)
    assert back.shape == rgb.shape
    assert torch.allclose(back, rgb, atol=1.5)


def test_maybe_rgb_to_colorspace_yuv():
    rgb = torch.rand(2, 3, 16, 16) * 255.0
    yuv = maybe_rgb_to_colorspace(rgb, "yuv")
    assert yuv.shape == rgb.shape
    assert yuv.min() >= 0.0
    assert yuv.max() <= 255.0


def test_maybe_rgb_to_colorspace_chw():
    rgb = torch.rand(3, 16, 16) * 255.0
    yuv = maybe_rgb_to_colorspace(rgb, "yuv")
    assert yuv.shape == rgb.shape


def test_colorspace_roundtrip_rgb_is_identity_for_rgb():
    rgb = torch.rand(3, 16, 16) * 255.0
    back = colorspace_roundtrip_rgb(rgb, "rgb")
    assert torch.allclose(back, rgb)


def test_colorspace_roundtrip_rgb_yuv_recovers_rgb():
    rgb = torch.rand(1, 3, 16, 16) * 255.0
    back = colorspace_roundtrip_rgb(rgb, "yuv")
    assert torch.allclose(back, rgb, atol=1.5)


def test_mlvc_torch_and_numpy_converters_match_yuv_utils():
    from rk3588_mobile_sr.data.mlvc_loader import rgb_to_mlvc_ycbcr as loader_rgb_to_ycbcr
    from rk3588_mobile_sr.deploy.rknn_eval import rgb_to_mlvc_ycbcr as numpy_rgb_to_ycbcr

    rgb = torch.rand(1, 3, 8, 12) * 255.0
    yuv_utils = rgb_to_yuv444(rgb)
    yuv_loader = loader_rgb_to_ycbcr(rgb / 255.0) * 255.0
    yuv_numpy = numpy_rgb_to_ycbcr(rgb[0].permute(1, 2, 0).numpy())

    assert torch.allclose(yuv_utils, yuv_loader, atol=1e-3)
    assert np.allclose(yuv_utils[0].permute(1, 2, 0).numpy(), yuv_numpy, atol=1e-3)
