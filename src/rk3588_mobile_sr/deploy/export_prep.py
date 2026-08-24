"""Prepare the BN-free Phase-RLFN core for ONNX export."""

from __future__ import annotations

import torch.nn as nn

from rk3588_mobile_sr.models import PhaseRLFNSR


def clip_deploy_weights(model: nn.Module, clip_min: float, clip_max: float) -> None:
    """Clamp Conv/Linear weights to the QAT training range."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data.clamp_(clip_min, clip_max)


def fused_weight_report(model: PhaseRLFNSR) -> dict[str, float]:
    """Return max absolute weights for each exported core component."""
    core = model.core
    report = {"stem": float(core.stem.weight.abs().max().item())}
    for index, block in enumerate(core.blocks):
        report[f"blocks.{index}.conv1"] = float(block.conv1.weight.abs().max().item())
        report[f"blocks.{index}.conv2"] = float(block.conv2.weight.abs().max().item())
    for name in ("feature_fuse", "codec_expand", "codec_fuse", "residual_head"):
        module = getattr(core, name)
        report[name] = float(module.weight.abs().max().item())
    return report


def prepare_float_for_export(
    model: PhaseRLFNSR,
    *,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> dict[str, float]:
    """Finalize the native deploy graph and optionally clip its weights."""
    model.switch_to_deploy()
    if clip_min is not None and clip_max is not None:
        clip_deploy_weights(model, clip_min, clip_max)
    model.eval()
    return fused_weight_report(model)
