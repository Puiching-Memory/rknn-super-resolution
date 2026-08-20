"""Tests for model modules."""

import torch

from rk3588_mobile_sr.models import MobileOneBlock, MobileOneSR


def test_mobileone_sr_forward_shape():
    """MobileOneSR forward should produce 3x upsampled output."""
    model = MobileOneSR(num_channels=16, num_blocks=2, scale=3)
    model.eval()

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3, 192, 192)
    assert torch.isfinite(out).all()
    assert out.min() >= 0.0
    assert out.max() <= 255.0


def test_mobileone_sr_output_range():
    """Output should be clipped to [0, 255] via Hardtanh."""
    model = MobileOneSR(num_channels=8, num_blocks=1, scale=3)
    model.eval()

    # Large negative/positive input to exercise clipping.
    x = torch.full((1, 3, 8, 8), 1e4)
    with torch.no_grad():
        out = model(x)

    assert out.min() >= 0.0
    assert out.max() <= 255.0


def test_mobileone_block_train_forward():
    """MobileOneBlock should produce correct shape in train mode."""
    block = MobileOneBlock(16, 16, num_conv_branches=2, inference_mode=False)
    block.eval()

    x = torch.randn(1, 16, 32, 32)
    with torch.no_grad():
        out = block(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mobileone_block_deploy_forward():
    """MobileOneBlock should work after switching to deploy/inference mode."""
    block = MobileOneBlock(8, 8, num_conv_branches=2, inference_mode=False, negative_slope=0.1)
    block.eval()

    x = torch.randn(1, 8, 16, 16)
    with torch.no_grad():
        train_out = block(x)

    block.reparameterize()
    assert block.inference_mode is True

    with torch.no_grad():
        deploy_out = block(x)

    assert deploy_out.shape == train_out.shape
    assert torch.allclose(deploy_out, train_out, atol=1e-5)


def test_mobileone_sr_switch_to_deploy():
    """MobileOneSR switch_to_deploy should convert all MobileOne blocks."""
    model = MobileOneSR(num_channels=8, num_blocks=2, scale=3)
    model.eval()

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        train_out = model(x)

    model.switch_to_deploy()

    deploy_blocks = [m for m in model.modules() if isinstance(m, MobileOneBlock)]
    assert all(m.inference_mode for m in deploy_blocks)

    with torch.no_grad():
        deploy_out = model(x)

    assert deploy_out.shape == train_out.shape


def test_mobileone_sr_core_contract():
    model = MobileOneSR(num_channels=8, num_blocks=2, scale=3, phase_factor=2)
    model.eval()
    lr = torch.randn(1, 3, 16, 20)

    with torch.no_grad():
        packed = torch.nn.functional.pixel_unshuffle(lr, 2)
        core = model.forward_core(packed)
        full = model(lr)
        reconstructed = torch.nn.functional.pixel_shuffle(core, 6)

    assert packed.shape == (1, 12, 8, 10)
    assert core.shape == (1, 108, 8, 10)
    assert full.shape == (1, 3, 48, 60)
    assert torch.equal(full, reconstructed)


def test_mobileone_sr_phase_switch_to_deploy_matches():
    model = MobileOneSR(num_channels=8, num_blocks=2)
    model.eval()
    lr = torch.randn(1, 3, 12, 16)
    with torch.no_grad():
        before = model(lr)

    model.switch_to_deploy()
    with torch.no_grad():
        after = model(lr)

    assert torch.allclose(before, after, atol=1e-4, rtol=1e-5)
