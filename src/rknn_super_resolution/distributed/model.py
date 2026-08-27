"""Compile and DDP wrapping for the BN-free training model."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from rknn_super_resolution.distributed.context import DistributedContext


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


def is_compiled_module(module: nn.Module) -> bool:
    return hasattr(unwrap_model(module), "_orig_mod") or hasattr(module, "_orig_mod")


def wrap_training_model(
    model: nn.Module,
    ctx: DistributedContext,
    *,
    compile_model: bool = True,
) -> nn.Module:
    """Compile, then wrap with DDP using the local CUDA device."""
    if compile_model:
        model = torch.compile(model)
    if ctx.world_size > 1:
        local_rank = ctx.device.index
        if ctx.device.type != "cuda" or local_rank is None:
            raise RuntimeError("DDP wrapping expects a CUDA device index")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model
