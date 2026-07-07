"""Tests for pyiqa perceptual metric helpers."""

import torch

from rk3588_mobile_sr.utils.pyiqa_metric import (
    batch_perceptual_metric,
    normalize_for_pyiqa,
)


def test_normalize_for_pyiqa():
    rgb = torch.tensor([[[[0.0, 255.0]]]])
    out = normalize_for_pyiqa(rgb)
    assert torch.allclose(out, torch.tensor([[[[0.0, 1.0]]]]))


def test_batch_perceptual_metric_returns_per_sample_scores():
    device = torch.device("cpu")
    pred = torch.rand(2, 3, 32, 32) * 255.0
    target = torch.rand(2, 3, 32, 32) * 255.0
    scores = batch_perceptual_metric(pred, target, device=device)
    assert scores.shape == (2,)
    assert torch.all(scores >= 0.0)
