"""pyiqa-backed full-reference perceptual metrics (default: DISTS)."""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_PERCEPTUAL_METRIC = "dists"

_cache: dict[tuple[str, str, bool], nn.Module] = {}


def normalize_for_pyiqa(tensor: torch.Tensor) -> torch.Tensor:
    """Map BCHW RGB in [0, 255] to [0, 1] for pyiqa."""
    return tensor / 255.0


def get_pyiqa_metric(
    *,
    metric: str = DEFAULT_PERCEPTUAL_METRIC,
    device: torch.device | str,
    as_loss: bool = False,
) -> nn.Module:
    """Return a cached pyiqa metric on ``device``."""
    key = (metric.lower(), str(device), as_loss)
    if key not in _cache:
        import pyiqa

        model = pyiqa.create_metric(metric, device=device, as_loss=as_loss)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _cache[key] = model
    return _cache[key]


def _per_sample_scores(
    model: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute per-image scores for BCHW tensors in [0, 1]."""
    batch_size = pred.shape[0]
    if batch_size == 1:
        score = model(pred, target)
        return score.reshape(1)

    scores = model(pred, target)
    if scores.numel() == batch_size:
        return scores.reshape(batch_size)

    return torch.stack(
        [model(pred[i : i + 1], target[i : i + 1]).reshape(()) for i in range(batch_size)]
    )


@torch.no_grad()
def batch_perceptual_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    metric: str = DEFAULT_PERCEPTUAL_METRIC,
    shave: int = 0,
) -> torch.Tensor:
    """Per-sample perceptual distance for BCHW RGB in [0, 255]. Lower is better."""
    from rk3588_mobile_sr.utils.sr_metrics import shave_borders

    pred = shave_borders(pred, shave)
    target = shave_borders(target, shave)
    model = get_pyiqa_metric(metric=metric, device=device, as_loss=False)
    return _per_sample_scores(
        model,
        normalize_for_pyiqa(pred),
        normalize_for_pyiqa(target),
    )


class PyIQAPerceptualLoss(nn.Module):
    """Differentiable pyiqa FR metric for RGB tensors in [0, 255]."""

    def __init__(self, *, metric: str = DEFAULT_PERCEPTUAL_METRIC) -> None:
        super().__init__()
        self.metric = metric
        self._model: nn.Module | None = None
        self._device: torch.device | None = None

    def _get_model(self, device: torch.device) -> nn.Module:
        if self._model is None or self._device != device:
            self._model = get_pyiqa_metric(metric=self.metric, device=device, as_loss=True)
            self._device = device
        return self._model

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        model = self._get_model(pred.device)
        pred_n = normalize_for_pyiqa(pred)
        target_n = normalize_for_pyiqa(target)
        scores = _per_sample_scores(model, pred_n, target_n)
        return scores.mean()
