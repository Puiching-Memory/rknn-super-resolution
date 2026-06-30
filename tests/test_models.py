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
    block = MobileOneBlock(8, 8, num_conv_branches=2, inference_mode=False)
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
