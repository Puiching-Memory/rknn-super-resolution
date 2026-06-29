"""LPIPS perceptual metric for SR validation."""

from __future__ import annotations

import torch
import torch.nn as nn

_lpips_models: dict[tuple[str, str], nn.Module] = {}


def get_lpips_model(*, net: str = "alex", device: torch.device | str) -> nn.Module:
    """Return a cached LPIPS model on ``device``."""
    import lpips

    key = (net, str(device))
    if key not in _lpips_models:
        model = lpips.LPIPS(net=net, verbose=False).to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _lpips_models[key] = model
    return _lpips_models[key]


def normalize_for_lpips(tensor: torch.Tensor) -> torch.Tensor:
    """Map BCHW RGB in [0, 255] to [-1, 1] for LPIPS."""
    return tensor / 127.5 - 1.0


@torch.no_grad()
def batch_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    net: str = "alex",
    shave: int = 0,
) -> torch.Tensor:
    """Per-sample LPIPS for BCHW RGB tensors in [0, 255]. Lower is better."""
    from utils.sr_metrics import shave_borders

    pred = shave_borders(pred, shave)
    target = shave_borders(target, shave)
    model = get_lpips_model(net=net, device=device)
    scores = model(normalize_for_lpips(pred), normalize_for_lpips(target))
    return scores.view(scores.shape[0])
