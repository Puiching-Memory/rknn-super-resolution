"""Tests for MambaIRv2Light teacher integration."""

from __future__ import annotations

import importlib.util

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mamba_ssm") is None,
    reason="mamba-ssm not installed",
)


def test_build_mambairv2_light_forward():
    from rk3588_mobile_sr.models.mambairv2_light import build_mambairv2_light

    model = build_mambairv2_light(upscale=3).eval()
    if not torch.cuda.is_available():
        pytest.skip("mamba selective_scan requires CUDA")
    model = model.cuda()
    x = torch.rand(1, 3, 48, 48, device="cuda")
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 144, 144)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_teacher_wrapper_output_range():
    from rk3588_mobile_sr.models.mambairv2_light import build_mambairv2_light
    from rk3588_mobile_sr.models.teacher_wrapper import TeacherWrapper

    model = build_mambairv2_light(upscale=3).eval()
    if not torch.cuda.is_available():
        pytest.skip("mamba selective_scan requires CUDA")
    model = model.cuda()
    teacher = TeacherWrapper(model, scale=3)
    lr = torch.rand(1, 3, 32, 32, device="cuda") * 255.0
    with torch.no_grad():
        out = teacher(lr)
    assert out.shape == (1, 3, 96, 96)
    assert out.min() >= 0.0
    assert out.max() <= 255.0
