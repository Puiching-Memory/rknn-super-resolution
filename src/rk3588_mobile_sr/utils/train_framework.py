"""Shared training framework for MobileOneSR stages."""

from __future__ import annotations

import argparse
import socket
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from rk3588_mobile_sr.config import load_config
from rk3588_mobile_sr.data.train_loader import (
    CodecTrainLoader,
    build_codec_train_loader,
    data_settings_from_args,
)
from rk3588_mobile_sr.data.val_loader import build_val_loader
from rk3588_mobile_sr.models.mobileone_sr import MobileOneSR
from rk3588_mobile_sr.utils.traceml_profiling import add_traceml_args


def resolve_colorspace(args: argparse.Namespace) -> str:
    """Resolve train/val colorspace from CLI or YAML (default yuv)."""
    explicit = getattr(args, "colorspace", None)
    if explicit:
        return explicit
    return load_config(getattr(args, "config", None)).data.colorspace


def resolve_prefetch_batches(args: argparse.Namespace) -> int:
    if getattr(args, "prefetch_batches", None) is not None:
        return args.prefetch_batches
    return load_config(getattr(args, "config", None)).data.prefetch_batches


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add training arguments shared by all stages."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config path (default: bundled mobileone_sr_x3.yaml)",
    )
    parser.add_argument(
        "--codec_manifest",
        type=str,
        default=None,
        help="offline codec clip manifest JSONL (default: data.codec_manifest from YAML)",
    )
    parser.add_argument(
        "--val_manifest",
        type=str,
        default=None,
        help="fixed validation manifest JSONL",
    )
    parser.add_argument(
        "--decode",
        type=str,
        default=None,
        choices=["auto", "raw"],
        help="frame read backend: auto or raw (offline LR .npy + source YUV)",
    )
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--num_channels", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--num_conv_branches", type=int, default=4)
    parser.add_argument("--negative_slope", type=float, default=0.1)
    parser.add_argument(
        "--colorspace",
        type=str,
        default=None,
        choices=["rgb", "yuv"],
        help="train/val tensor layout: rgb or yuv444 (BT.601, [0,255])",
    )
    parser.add_argument(
        "--no_nv12_simulate",
        action="store_true",
        help="disable NV12 4:2:0 chroma subsample simulation before YUV conversion",
    )
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16, help="per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=None,
        help="background batches to prefetch ahead of the training step",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--sync_bn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use SyncBatchNorm across DDP ranks (usually unnecessary with large per-GPU batch)",
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
        "--swanlab_run_id",
        type=str,
        default=None,
        help="SwanLab run id for resume (default: load from checkpoint/swanlab_run.json)",
    )
    parser.add_argument(
        "--vis_samples",
        type=int,
        default=8,
        help="number of validation / data-preview panels (diverse codec×bitrate)",
    )
    parser.add_argument(
        "--no_data_preview",
        action="store_true",
        help="skip pre-training codec downsample / colorspace preview",
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


def _training_module_for_state_dict(model: nn.Module) -> nn.Module:
    """Return the innermost trainable module for checkpoint load/save."""
    from rk3588_mobile_sr.distributed.model import unwrap_model

    inner = unwrap_model(model)
    if hasattr(inner, "_orig_mod"):
        return inner._orig_mod
    return inner


def training_module_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Serialize model weights without DDP/compile prefixes."""
    return _training_module_for_state_dict(model).state_dict()


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


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    *,
    train_accel: TrainAccel | None = None,
    device: torch.device | None = None,
) -> int:
    """Restore model/optimizer/scheduler (and AMP scaler) from a stage checkpoint."""
    raw = torch.load(path, map_location=device or "cpu", weights_only=False)
    if not isinstance(raw, dict) or "state_dict" not in raw:
        raise TypeError(f"Expected full training checkpoint in {path}")
    unwrap = _training_module_for_state_dict(model)
    unwrap.load_state_dict(_normalize_state_dict(raw["state_dict"]), strict=True)
    optimizer.load_state_dict(raw["optimizer"])
    scheduler.load_state_dict(raw["scheduler"])
    if train_accel is not None and train_accel.scaler is not None and "scaler" in raw:
        train_accel.scaler.load_state_dict(raw["scaler"])
    return int(raw.get("step", 0))


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
        negative_slope=getattr(args, "negative_slope", 0.1),
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
) -> tuple[CodecTrainLoader, None, DataLoader | None]:
    """Build canvas codec train/validation loaders."""
    del device
    if distributed and dist.is_initialized():
        rank = dist.get_rank() if rank is None else rank
    else:
        rank = 0 if rank is None else rank

    world_size = dist.get_world_size() if distributed and dist.is_initialized() else 1
    settings = replace(data_settings_from_args(args), augment=train_aug)

    train_bundle = build_codec_train_loader(
        settings,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        seed=42,
        device_id=rank if distributed else 0,
    )

    val_loader = None
    app_cfg = load_config(getattr(args, "config", None))
    val_manifest = getattr(args, "val_manifest", None) or app_cfg.data.val_manifest
    if val_manifest:
        val_loader, _ = build_val_loader(
            settings,
            val_manifest=val_manifest,
            batch_size=val_bs,
            num_workers=2,
            distributed=distributed,
            rank=rank,
        )

    return train_bundle, None, val_loader


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
