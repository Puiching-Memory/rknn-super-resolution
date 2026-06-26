"""QAT helpers: fuse, prepare, convert for RK3588 deploy graph."""

from typing import Any

import torch
import torch.nn as nn
from torch.ao.quantization import (
    QConfigMapping,
    default_observer,
    default_weight_observer,
    get_default_qat_qconfig,
)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

from .mobileone_sr import MobileOneSR


def fuse_stem(model: MobileOneSR) -> MobileOneSR:
    """Fuse stem Conv+BN+ReLU into Conv+ReLU for FX QAT."""
    conv, bn, relu = model.stem[0], model.stem[1], model.stem[2]
    std = (bn.running_var + bn.eps).sqrt()
    weight = conv.weight * (bn.weight / std).view(-1, 1, 1, 1)
    bias = bn.bias - bn.running_mean * bn.weight / std
    if conv.bias is not None:
        bias = bias + conv.bias

    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True,
    )
    fused.weight.data = weight
    fused.bias.data = bias
    model.stem = nn.Sequential(fused, relu)
    return model


def get_qconfig(backend: str = "qnnpack", act_quant_min: int = 0, act_quant_max: int = 255):
    """Build a QAT qconfig tuned for [0,255] images."""
    if backend == "qnnpack":
        return get_default_qat_qconfig("qnnpack")

    act_observer = default_observer.with_args(quant_min=act_quant_min, quant_max=act_quant_max)
    weight_observer = default_weight_observer
    qconfig = torch.ao.quantization.QConfig(
        activation=act_observer,
        weight=weight_observer,
    )
    return qconfig


def prepare_model_for_qat(
    model: MobileOneSR,
    backend: str = "qnnpack",
    example_inputs: tuple[torch.Tensor, ...] = None,
) -> torch.fx.GraphModule:
    """Switch to deploy, fuse stem, then prepare FX QAT."""
    model.switch_to_deploy()
    model = fuse_stem(model)
    model.train()

    qconfig = get_qconfig(backend)
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    if example_inputs is None:
        example_inputs = (torch.randn(1, 3, 360, 640),)

    prepared = prepare_qat_fx(model, qconfig_mapping, example_inputs)
    return prepared


def convert_qat_model(prepared_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Convert a prepared QAT model to quantized (int8) model."""
    prepared_model.eval()
    return convert_fx(prepared_model)


def bn_recalibrate(model: nn.Module, loader: Any, device: torch.device, batches: int = 64) -> None:
    """Forward-only mini-batch recalibration of BN running statistics."""
    model.train()
    with torch.no_grad():
        for idx, (lr, _) in enumerate(loader):
            if idx >= batches:
                break
            lr = lr.to(device)
            _ = model(lr)
