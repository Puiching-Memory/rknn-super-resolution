"""Model wrapping for distributed training."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from rk3588_mobile_sr.distributed.context import DistributedContext


class SyncBnPolicy(Enum):
    """When to convert BatchNorm to SyncBatchNorm."""

    NEVER = "never"
    IF_FLAG = "if_flag"
    ALWAYS = "always"


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


def is_compiled_module(module: nn.Module) -> bool:
    return hasattr(unwrap_model(module), "_orig_mod") or hasattr(module, "_orig_mod")


def wrap_training_model(
    model: nn.Module,
    ctx: DistributedContext,
    *,
    compile_model: bool = True,
    sync_bn: SyncBnPolicy = SyncBnPolicy.NEVER,
    sync_bn_flag: bool = False,
) -> nn.Module:
    """Apply compile → SyncBN → DDP in the canonical order."""
    if compile_model:
        model = torch.compile(model)

    use_sync_bn = sync_bn == SyncBnPolicy.ALWAYS or (
        sync_bn == SyncBnPolicy.IF_FLAG and sync_bn_flag
    )
    if use_sync_bn and ctx.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if ctx.world_size > 1:
        model = DDP(model, device_ids=[ctx.rank])

    return model
