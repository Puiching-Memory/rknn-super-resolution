"""Tests for core-only QAT preparation."""

import torch

from rknn_super_resolution.models import PhaseRLFNSR
from rknn_super_resolution.models.qat_utils import prepare_model_for_qat


def test_qat_prepares_only_residual_core() -> None:
    model = PhaseRLFNSR(num_channels=8, num_blocks=1, codec_project_channels=4)
    phases = torch.randn(1, 12, 8, 8)
    codec = torch.randn(1, 96, 2, 2)
    prepared = prepare_model_for_qat(model, example_inputs=(phases, codec))
    assert isinstance(prepared.core, torch.fx.GraphModule)
    assert isinstance(prepared.input_unshuffle, torch.nn.PixelUnshuffle)
    assert isinstance(prepared.output_shuffle, torch.nn.PixelShuffle)
    with torch.no_grad():
        output = prepared(torch.rand(1, 3, 16, 16) * 255.0, codec)
    assert output.shape == (1, 3, 48, 48)
