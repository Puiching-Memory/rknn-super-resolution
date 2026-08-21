"""Tests for QAT stem fusion and checkpoint filtering."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from rk3588_mobile_sr.models.mobileone_sr import MobileOneSR
from rk3588_mobile_sr.models.qat_utils import _filter_qat_state_dict, fuse_stem


def test_fuse_stem_keeps_channels_and_matches_unfused_eval():
    model = MobileOneSR(num_channels=8, num_blocks=1, scale=3, phase_factor=2)
    model.eval()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        before = model(x)

    fused = fuse_stem(copy.deepcopy(model))
    assert len(fused.stem) == 2
    assert isinstance(fused.stem[0], nn.Conv2d)
    assert fused.stem[0].bias is not None
    assert fused.stem[0].in_channels == model.core_in_channels
    assert fused.stem[0].out_channels == 8

    with torch.no_grad():
        after = fused(x)
    assert after.shape == before.shape
    assert torch.allclose(before, after, atol=1e-4, rtol=1e-5)


def test_qat_example_input_is_training_layout_not_npu_core():
    model = MobileOneSR(num_channels=8, num_blocks=1, scale=3, phase_factor=2)
    assert model.core_in_channels == 12
    example = torch.randn(1, 3, 16, 16)
    assert example.shape[1] != model.core_in_channels
    with torch.no_grad():
        out = model(example)
    assert out.shape == (1, 3, 48, 48)


def test_filter_qat_state_dict_drops_observers():
    state = {
        "stem.0.weight": torch.ones(1),
        "stem.0.activation_post_process.scale": torch.ones(1),
        "body.0.fake_quant.scale": torch.ones(1),
        "out_conv.weight": torch.ones(1),
        "out_conv.observer.min_val": torch.zeros(1),
    }
    filtered = _filter_qat_state_dict(state)
    assert set(filtered) == {"stem.0.weight", "out_conv.weight"}
