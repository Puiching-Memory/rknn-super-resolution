"""Tests for the BN-free Phase-RLFN architecture."""

import torch

from rk3588_mobile_sr.models import PhaseRLFNSR, ResidualLocalBlock


def test_phase_rlfn_starts_as_exact_bicubic() -> None:
    model = PhaseRLFNSR(num_channels=8, num_blocks=2).eval()
    current = torch.rand(1, 3, 16, 20) * 255.0
    with torch.no_grad():
        output = model(current)
    assert torch.equal(output, model.bicubic_base(current))
    assert output.shape == (1, 3, 48, 60)


def test_codec_adapter_is_optional_and_zero_initialized() -> None:
    model = PhaseRLFNSR(num_channels=8, num_blocks=1).eval()
    current = torch.rand(1, 3, 16, 20) * 255.0
    codec = torch.randn(1, 96, 2, 3)
    with torch.no_grad():
        fallback = model(current)
        codec_aware = model(current, codec)
    assert torch.equal(fallback, codec_aware)


def test_zero_codec_is_exact_fallback_after_adapter_learns() -> None:
    model = PhaseRLFNSR(num_channels=8, num_blocks=1).eval()
    model.core.codec_fuse.weight.data.normal_()
    current = torch.rand(1, 3, 16, 20) * 255.0
    zero_codec = torch.zeros(1, 96, 2, 3)
    with torch.no_grad():
        assert torch.equal(model(current), model(current, zero_codec))


def test_residual_local_block_preserves_shape_and_has_no_bn() -> None:
    block = ResidualLocalBlock(8)
    output = block(torch.randn(1, 8, 10, 12))
    assert output.shape == (1, 8, 10, 12)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in block.modules())


def test_core_contract_and_deploy_noop() -> None:
    model = PhaseRLFNSR(num_channels=8, num_blocks=2).eval()
    current = torch.rand(1, 3, 16, 20) * 255.0
    phases = torch.nn.functional.pixel_unshuffle(current, 2)
    codec = torch.randn(1, 96, 2, 3)
    with torch.no_grad():
        before = model.forward_core(phases, codec)
        model.switch_to_deploy()
        after = model.forward_core(phases, codec)
    assert phases.shape == (1, 12, 8, 10)
    assert before.shape == (1, 108, 8, 10)
    assert torch.equal(before, after)
