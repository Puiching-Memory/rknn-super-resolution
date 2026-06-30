"""Shared training framework for MobileOneSR stages."""

from __future__ import annotations

import argparse
import socket
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rk3588_mobile_sr.data.div2k_dali import build_dali_train_loader
from rk3588_mobile_sr.data.div2k_lmdb import build_lmdb_train_loader
from rk3588_mobile_sr.data.div2k_loader import build_dataloader
from rk3588_mobile_sr.models.mobileone_sr import MobileOneSR
from rk3588_mobile_sr.utils.traceml_profiling import add_traceml_args


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add training arguments shared by all stages."""
    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--lr_dir", type=str, default=None)
    parser.add_argument("--val_hr_dir", type=str, default=None)
    parser.add_argument("--val_lr_dir", type=str, default=None)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--num_channels", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--num_conv_branches", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16, help="per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--use_dali",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use NVIDIA DALI for GPU-accelerated data loading",
    )
    parser.add_argument(
        "--lmdb_dir",
        type=str,
        default=None,
        help="prebuilt LMDB patch cache (takes priority over DALI when set)",
    )
    parser.add_argument(
        "--lmdb_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply flip/transpose augment when reading LMDB patches (online, each epoch)",
    )
    parser.add_argument(
        "--dali_num_threads",
        type=int,
        default=8,
        help="CPU threads per DALI pipeline",
    )
    parser.add_argument(
        "--dali_min_steps_per_epoch",
        type=int,
        default=512,
        help="repeat file list so each DDP rank has at least this many batches per epoch",
    )
    parser.add_argument(
        "--dali_prefetch_queue_depth",
        type=int,
        default=4,
        help="DALI reader prefetch queue depth",
    )
    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=4,
        help="background batches to prefetch ahead of the training step",
    )
    parser.add_argument(
        "--sync_bn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use SyncBatchNorm across DDP ranks (usually unnecessary with large per-GPU batch)",
    )
    parser.add_argument(
        "--samples_per_image",
        type=int,
        default=1,
        help="random patches sampled per image each epoch (DALI train loader)",
    )
    parser.add_argument("--swanlab_project", type=str, default="rk3588-mobile-sr")
    parser.add_argument(
        "--swanlab_experiment",
        type=str,
        default=None,
        help="SwanLab experiment name; defaults to the training stage script name",
    )
    parser.add_argument(
        "--no_swanlab",
        action="store_true",
        help="disable SwanLab experiment logging",
    )
    parser.add_argument(
        "--vis_samples",
        type=int,
        default=4,
        help="number of validation image panels to upload to SwanLab",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="disable SwanLab validation image logging",
    )
    parser.add_argument(
        "--vis_max_size",
        type=int,
        default=768,
        help="max side length of uploaded validation images",
    )
    parser.add_argument(
        "--no_model_diag",
        action="store_true",
        help="disable model diagnostics (grad norms, clip saturation, deploy check)",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="log training loss every N batches within an epoch (0 = epoch end only)",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="automatic mixed precision on CUDA (bf16 when supported, else fp16)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the model before DDP wrapping",
    )
    add_traceml_args(parser)
    return parser


@dataclass
class TrainAccel:
    """AMP settings shared by training loops."""

    enabled: bool
    dtype: torch.dtype
    scaler: torch.amp.GradScaler | None


def resolve_amp_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_train_accel(args: argparse.Namespace) -> TrainAccel:
    if not getattr(args, "amp", True):
        return TrainAccel(enabled=False, dtype=torch.float32, scaler=None)
    dtype = resolve_amp_dtype()
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    return TrainAccel(enabled=True, dtype=dtype, scaler=scaler)


def amp_autocast(accel: TrainAccel):
    if accel.enabled:
        return torch.amp.autocast("cuda", dtype=accel.dtype)
    return nullcontext()


def run_backward(
    loss: torch.Tensor,
    optimizer: optim.Optimizer,
    accel: TrainAccel,
) -> None:
    if accel.scaler is not None:
        accel.scaler.scale(loss).backward()
        accel.scaler.step(optimizer)
        accel.scaler.update()
    else:
        loss.backward()
        optimizer.step()


def maybe_sync_batchnorm(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if getattr(args, "sync_bn", False):
        return nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def maybe_compile(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if not getattr(args, "compile", True):
        return model
    return torch.compile(model)


def require_cuda() -> None:
    """Ensure a CUDA-capable PyTorch build and visible GPU are available."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but unavailable. Run `uv sync` to install cu130 PyTorch."
        )


def setup_device(args: argparse.Namespace) -> torch.device:
    """Resolve the CUDA device from CLI args."""
    require_cuda()
    device_name = args.device
    if device_name == "cuda":
        return torch.device("cuda")
    if device_name.startswith("cuda:"):
        return torch.device(device_name)
    raise ValueError(f"Only CUDA devices are supported, got {device_name!r}")


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip DDP ``module.`` and ``torch.compile`` ``_orig_mod.`` prefixes."""
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        name = key
        if name.startswith("_orig_mod."):
            name = name.removeprefix("_orig_mod.")
        if name.startswith("module."):
            name = name.removeprefix("module.")
        normalized[name] = value
    return normalized


def build_model(
    args: argparse.Namespace,
    device: torch.device,
    *,
    inference_mode: bool = False,
    weight_path: str | None = None,
    strict_load: bool = True,
) -> MobileOneSR:
    """Build a MobileOneSR model and optionally load pretrained weights."""
    model = MobileOneSR(
        scale=args.scale,
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        num_conv_branches=args.num_conv_branches,
        inference_mode=inference_mode,
    ).to(device)
    if weight_path is not None:
        raw = torch.load(weight_path, map_location=device, weights_only=False)
        if isinstance(raw, dict) and "state_dict" in raw:
            state_dict = _normalize_state_dict(raw["state_dict"])
        elif isinstance(raw, dict):
            state_dict = _normalize_state_dict(raw)
        else:
            raise TypeError(f"Unsupported checkpoint format in {weight_path}")
        model.load_state_dict(state_dict, strict=strict_load)
    return model


def build_loaders(
    args: argparse.Namespace,
    device: torch.device,
    *,
    train_aug: bool = True,
    val_bs: int = 1,
    distributed: bool = False,
    rank: int | None = None,
    world_size: int | None = None,
) -> tuple[DataLoader | object, DistributedSampler | object | None, DataLoader | None]:
    """Build train/validation dataloaders."""
    if distributed and dist.is_initialized():
        rank = dist.get_rank() if rank is None else rank
        world_size = dist.get_world_size() if world_size is None else world_size
    else:
        rank = 0 if rank is None else rank
        world_size = 1 if world_size is None else world_size

    use_lmdb = bool(getattr(args, "lmdb_dir", None)) and train_aug
    use_dali = getattr(args, "use_dali", False) and train_aug and not use_lmdb
    if use_lmdb:
        train_loader, train_sampler = build_lmdb_train_loader(
            args.lmdb_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=args.lmdb_augment,
            distributed=distributed,
        )
    elif use_dali:
        if args.lr_dir is None:
            raise ValueError("--lr_dir is required when --use_dali is enabled")
        train_loader = build_dali_train_loader(
            hr_dir=args.hr_dir,
            lr_dir=args.lr_dir,
            scale=args.scale,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            device_id=rank,
            shard_id=rank,
            num_shards=world_size,
            num_threads=args.dali_num_threads,
            augment=True,
            samples_per_image=getattr(args, "samples_per_image", 1),
            min_steps_per_epoch=getattr(args, "dali_min_steps_per_epoch", 512),
            prefetch_queue_depth=getattr(args, "dali_prefetch_queue_depth", 4),
        )
        train_sampler = None
    else:
        train_loader, train_sampler = build_dataloader(
            hr_dir=args.hr_dir,
            lr_dir=args.lr_dir,
            scale=args.scale,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=train_aug,
            distributed=distributed,
        )

    val_loader = None
    if args.val_hr_dir:
        val_loader, _ = build_dataloader(
            hr_dir=args.val_hr_dir,
            lr_dir=args.val_lr_dir,
            scale=args.scale,
            patch_size=args.patch_size,
            batch_size=val_bs,
            num_workers=args.num_workers,
            augment=False,
            distributed=distributed,
        )

    return train_loader, train_sampler, val_loader


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def setup_ddp(
    rank: int | None = None,
    world_size: int | None = None,
    *,
    device_id: torch.device | None = None,
) -> int:
    """Initialize NCCL process group; prefer :func:`distributed_session` in new code."""
    from rk3588_mobile_sr.distributed.context import _init_process_group

    ctx = _init_process_group(rank, world_size, device_id=device_id)
    return ctx.rank


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint_dict(
    state_dict: dict,
    path: Path,
    *,
    is_best: bool = False,
    best_metric_tracker: dict | None = None,
) -> bool:
    """Save checkpoint if is_best passes the best metric tracker."""
    if is_best and best_metric_tracker is not None:
        metric = best_metric_tracker.get("metric")
        if metric is None or metric <= best_metric_tracker["value"]:
            return False
        best_metric_tracker["value"] = metric
    torch.save(state_dict, path)
    return True


def make_optimizer(model: nn.Module, lr: float) -> optim.Adam:
    return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
