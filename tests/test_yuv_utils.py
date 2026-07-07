"""Tests for RGB/YUV conversion utilities."""

import torch

from rk3588_mobile_sr.data.yuv_utils import (
    colorspace_roundtrip_rgb,
    maybe_rgb_to_colorspace,
    rgb_to_model_colorspace,
    rgb_to_yuv444,
    simulate_nv12_roundtrip_rgb,
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


def test_simulate_nv12_reduces_chroma_resolution():
    rgb = torch.rand(1, 3, 16, 16) * 255.0
    nv12_rgb = simulate_nv12_roundtrip_rgb(rgb)
    assert nv12_rgb.shape == rgb.shape
    assert not torch.allclose(nv12_rgb, rgb, atol=0.01)


def test_rgb_to_model_colorspace_nv12_then_yuv():
    rgb = torch.rand(2, 3, 16, 16) * 255.0
    yuv = rgb_to_model_colorspace(rgb, colorspace="yuv", nv12_simulate=True)
    assert yuv.shape == rgb.shape
    yuv_plain = rgb_to_model_colorspace(rgb, colorspace="yuv", nv12_simulate=False)
    assert not torch.allclose(yuv, yuv_plain, atol=0.01)
