"""Tests for loss modules."""

import pytest
import torch

from rk3588_mobile_sr.losses import CharbonnierLoss, DCTLoss


def test_charbonnier_loss_positive():
    """Charbonnier loss should be positive for non-identical inputs."""
    criterion = CharbonnierLoss(eps=1e-6)
    pred = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)

    loss = criterion(pred, target)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_charbonnier_loss_zero_for_equal():
    """Charbonnier loss should approach zero when inputs match."""
    criterion = CharbonnierLoss(eps=1e-6)
    x = torch.randn(2, 3, 16, 16)

    loss = criterion(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_dct_loss_shape_and_sign():
    """DCTLoss should return a finite scalar."""
    criterion = DCTLoss(block_size=8)
    pred = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)

    loss = criterion(pred, target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
