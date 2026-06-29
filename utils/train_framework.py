"""Shared training framework for MobileOneSR stages."""

from __future__ import annotations

import argparse
import os
import socket
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.div2k_dali import build_dali_train_loader
from data.div2k_lmdb import build_lmdb_train_loader
from data.div2k_loader import build_dataloader
from models.mobileone_sr import MobileOneSR
from utils.model_diagnostics import (
    ForwardDiagnosticsTracker,
    check_deploy_consistency,
    collect_training_diagnostics,
)
from utils.sr_metrics import validate_ddp, validate_ddp_extended
from utils.swanlab_logging import log_metrics, log_validation_sr_images
from utils.traceml_profiling import add_traceml_args, trace_training_step


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
        state_dict = torch.load(weight_path, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
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
    """Initialize the default NCCL process group.

    When running under ``torchrun`` the environment variables ``RANK``,
    ``WORLD_SIZE``, ``MASTER_ADDR`` and ``MASTER_PORT`` are already set, so we
    simply call ``init_process_group`` using those values.  For single-process
    entry points (e.g. plain ``python train_stage*.py`` without a launcher) the
    function can be called with explicit ``rank``/``world_size`` arguments.

    Returns the rank of the current process.
    """
    rank_env = int(os.environ["RANK"]) if "RANK" in os.environ else None
    world_size_env = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else None

    if rank is not None and rank_env is not None and rank != rank_env:
        raise RuntimeError(
            f"Explicit rank {rank} does not match torchrun-provided RANK={rank_env}"
        )
    if world_size is not None and world_size_env is not None and world_size != world_size_env:
        raise RuntimeError(
            f"Explicit world_size {world_size} does not match torchrun-provided "
            f"WORLD_SIZE={world_size_env}"
        )

    rank = rank_env if rank_env is not None else rank
    world_size = world_size_env if world_size_env is not None else world_size
    if rank is None or world_size is None:
        raise RuntimeError(
            "rank and world_size must be provided either by torchrun env vars or "
            "as arguments to setup_ddp()"
        )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    if "MASTER_PORT" not in os.environ:
        raise RuntimeError(
            "MASTER_PORT is not set. When using torchrun it is provided by the launcher."
        )

    if device_id is None:
        device_id = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device_id)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=device_id,
    )
    return rank


def _average_metrics_across_ranks(
    metrics: dict[str, float],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, float]:
    if world_size <= 1:
        return metrics
    reduced: dict[str, float] = {}
    for key, value in metrics.items():
        tensor = torch.tensor([value], device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        reduced[key] = float(tensor.item())
    return reduced


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def train_epochs_ddp(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    device: torch.device,
    rank: int,
    world_size: int,
    *,
    epochs: int,
    val_loader: DataLoader | None = None,
    save_dir: Path | None = None,
    val_every: int = 10,
    save_every: int = 50,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
    epoch_start: Callable[[int], None] | None = None,
    post_step: Callable[[nn.Module], None] | None = None,
    save_best_extra: Callable[[Path], None] | None = None,
    best_metric_tracker: dict | None = None,
    vis_samples: int = 4,
    log_images: bool = True,
    vis_max_size: int = 768,
    train_accel: TrainAccel | None = None,
    val_scale: int = 3,
    extended_val: bool = False,
    val_lpips: bool = False,
    model_diag: bool = True,
) -> float:
    """Generic DDP epoch-based training loop.

    ``model`` is expected to be a ``DistributedDataParallel`` wrapper.  The
    underlying ``model.module`` is passed to ``loss_fn`` and ``post_step`` so
    that callbacks do not need to handle DDP themselves.
    """
    if best_metric_tracker is None:
        best_metric_tracker = {"value": -1.0}
    if train_accel is None:
        train_accel = TrainAccel(enabled=False, dtype=torch.float32, scaler=None)
    if save_dir is not None:
        save_dir = Path(save_dir)
        if rank == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
        if world_size > 1:
            dist.barrier()

    unwrap = getattr(model, "module", model)
    is_rank0 = rank == 0

    progress = None
    if is_rank0:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
        )
        progress.start()
        epoch_task = progress.add_task("Epochs", total=epochs)
        batch_task = progress.add_task("Batches", visible=False)

    best_psnr = -1.0
    diag_tracker = ForwardDiagnosticsTracker(model) if model_diag else None
    try:
        for epoch in range(1, epochs + 1):
            if epoch_start is not None:
                epoch_start(epoch)

            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            total_loss = 0.0
            component_sums: dict[str, float] = defaultdict(float)
            if is_rank0:
                progress.reset(batch_task, total=len(train_loader), visible=True)
                progress.update(batch_task, description=f"Epoch {epoch}")

            for lr, hr in train_loader:
                with trace_training_step(model):
                    lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with amp_autocast(train_accel):
                        loss_result = loss_fn(unwrap, lr, hr)
                    if isinstance(loss_result, tuple):
                        loss, step_metrics = loss_result
                        if isinstance(step_metrics, dict):
                            for key, value in step_metrics.items():
                                component_sums[key] += float(value)
                    else:
                        loss = loss_result
                    run_backward(loss, optimizer, train_accel)
                    if post_step is not None:
                        post_step(unwrap)
                total_loss += loss.item()
                if is_rank0:
                    progress.advance(batch_task)

            avg_loss = torch.tensor(total_loss / len(train_loader), device=device)
            if world_size > 1:
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

            reduced_components: dict[str, float] = {}
            for key, value in component_sums.items():
                avg_component = torch.tensor(value / len(train_loader), device=device)
                if world_size > 1:
                    dist.all_reduce(avg_component, op=dist.ReduceOp.AVG)
                reduced_components[key] = avg_component.item()

            diag_metrics: dict[str, float] = {}
            if diag_tracker is not None:
                diag_metrics = collect_training_diagnostics(model, diag_tracker)
                diag_metrics = _average_metrics_across_ranks(
                    diag_metrics,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                )

            if is_rank0:
                metrics = {"train/loss": avg_loss.item(), **reduced_components, **diag_metrics}
                if scheduler is not None:
                    metrics["train/lr"] = scheduler.get_last_lr()[0]
                log_metrics(metrics, step=epoch)
                detail = ""
                if "train/loss_charbonnier" in metrics:
                    detail = (
                        f" | charb={metrics['train/loss_charbonnier']:.4f}"
                        f" | dct={metrics.get('train/loss_dct_weighted', 0.0):.4f}"
                        f" | kd={metrics.get('train/loss_kd_weighted', 0.0):.4f}"
                    )
                progress.console.print(
                    f"Epoch {epoch}/{epochs} | loss={avg_loss.item():.4f}{detail}",
                    highlight=False,
                )

            if scheduler is not None:
                scheduler.step()

            if val_loader is not None and epoch % val_every == 0:
                if extended_val:
                    psnr, val_metrics = validate_ddp_extended(
                        model,
                        val_loader,
                        rank,
                        world_size,
                        scale=val_scale,
                        compute_lpips=val_lpips,
                    )
                else:
                    psnr = validate_ddp(
                        model,
                        val_loader,
                        rank,
                        world_size,
                        scale=val_scale,
                    )
                    val_metrics = None
                if is_rank0:
                    if val_metrics is not None:
                        val_log = val_metrics.to_log_dict()
                        val_log["val/best_psnr"] = max(best_psnr, psnr)
                        log_metrics(val_log, step=epoch)
                    else:
                        log_metrics({"val/psnr": psnr}, step=epoch)
                    if model_diag:
                        deploy_metrics = check_deploy_consistency(model, val_loader, device)
                        if deploy_metrics:
                            log_metrics(deploy_metrics, step=epoch)
                    if log_images:
                        log_validation_sr_images(
                            model,
                            val_loader,
                            device,
                            step=epoch,
                            num_samples=vis_samples,
                            max_size=vis_max_size,
                        )
                    detail = ""
                    if val_metrics is not None:
                        detail = (
                            f" | Y-PSNR={val_metrics.y_psnr:.2f}"
                            f" | SSIM={val_metrics.ssim:.4f}"
                        )
                        if val_metrics.lpips is not None:
                            detail += f" | LPIPS={val_metrics.lpips:.4f}"
                    progress.console.print(
                        f"  val PSNR={psnr:.2f}{detail}",
                        highlight=False,
                    )
                    if psnr > best_psnr:
                        best_psnr = psnr
                        if save_dir is not None:
                            best_path = save_dir / "best.pth"
                            saved = save_checkpoint_dict(
                                unwrap.state_dict(),
                                best_path,
                                is_best=True,
                                best_metric_tracker=best_metric_tracker,
                            )
                            if saved and save_best_extra is not None:
                                save_best_extra(best_path)

            if is_rank0 and save_dir is not None and epoch % save_every == 0:
                torch.save(unwrap.state_dict(), save_dir / f"epoch_{epoch}.pth")

            if is_rank0:
                progress.advance(epoch_task)
    finally:
        if diag_tracker is not None:
            diag_tracker.close()
        if progress is not None:
            progress.stop()

    if is_rank0 and save_dir is not None:
        torch.save(unwrap.state_dict(), save_dir / "last.pth")

    return best_psnr


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate PSNR on the validation set, averaged per sample."""
    model.eval()
    total_psnr = 0.0
    total_samples = 0
    for lr, hr in dataloader:
        lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
        out = torch.clamp(model(lr), 0.0, 255.0)
        mse = torch.mean((out - hr) ** 2, dim=[1, 2, 3])
        psnr = 10 * torch.log10(255.0 * 255.0 / mse)
        total_psnr += psnr.sum().item()
        total_samples += psnr.numel()
    model.train()
    return total_psnr / total_samples


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


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    device: torch.device,
    *,
    epochs: int,
    val_loader: DataLoader | None = None,
    save_dir: Path | None = None,
    val_every: int = 10,
    save_every: int = 50,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
    epoch_start: Callable[[int], None] | None = None,
    post_step: Callable[[nn.Module], None] | None = None,
    save_best_extra: Callable[[Path], None] | None = None,
    best_metric_tracker: dict | None = None,
) -> float:
    """Generic single-GPU epoch-based training loop.

    Args:
        model: Network to train.
        train_loader: Training dataloader.
        loss_fn: ``loss_fn(model, lr, hr) -> loss_tensor``.
        optimizer: Optimizer.
        device: Training device.
        epochs: Total epochs.
        val_loader: Optional validation loader.
        save_dir: Directory for checkpoints.
        val_every: Validate every N epochs.
        save_every: Save checkpoint every N epochs.
        scheduler: Optional LR scheduler, called once per epoch.
        epoch_start: Optional callback invoked at the start of each epoch.
        post_step: Optional callback invoked after ``optimizer.step()``.
        save_best_extra: Optional callback invoked with the best checkpoint path
            when a new best validation PSNR is achieved.
        best_metric_tracker: Optional ``{"value": float}`` dict used to track
            the best validation metric across the run.

    Returns:
        Best validation PSNR observed, or -1.0 if validation was not run.
    """
    if best_metric_tracker is None:
        best_metric_tracker = {"value": -1.0}
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    )

    best_psnr = -1.0
    with progress:
        epoch_task = progress.add_task("Epochs", total=epochs)
        batch_task = progress.add_task("Batches", visible=False)

        for epoch in range(1, epochs + 1):
            if epoch_start is not None:
                epoch_start(epoch)

            model.train()
            total_loss = 0.0
            progress.reset(batch_task, total=len(train_loader), visible=True)
            progress.update(batch_task, description=f"Epoch {epoch}")

            for lr, hr in train_loader:
                with trace_training_step(model):
                    lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model, lr, hr)
                    loss.backward()
                    optimizer.step()
                    if post_step is not None:
                        post_step(model)
                total_loss += loss.item()
                progress.advance(batch_task)

            avg_loss = total_loss / len(train_loader)
            metrics = {"train/loss": avg_loss}
            if scheduler is not None:
                metrics["train/lr"] = scheduler.get_last_lr()[0]
            log_metrics(metrics, step=epoch)
            progress.console.print(f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f}", highlight=False)

            if scheduler is not None:
                scheduler.step()

            if val_loader is not None and epoch % val_every == 0:
                psnr = validate(model, val_loader, device)
                log_metrics({"val/psnr": psnr}, step=epoch)
                progress.console.print(f"  val PSNR={psnr:.2f}", highlight=False)

                if psnr > best_psnr:
                    best_psnr = psnr
                    if save_dir is not None:
                        best_path = save_dir / "best.pth"
                        saved = save_checkpoint_dict(
                            model.state_dict(),
                            best_path,
                            is_best=True,
                            best_metric_tracker=best_metric_tracker,
                        )
                        if saved and save_best_extra is not None:
                            save_best_extra(best_path)

            if save_dir is not None and epoch % save_every == 0:
                torch.save(model.state_dict(), save_dir / f"epoch_{epoch}.pth")

            progress.advance(epoch_task)

    if save_dir is not None:
        torch.save(model.state_dict(), save_dir / "last.pth")

    return best_psnr
