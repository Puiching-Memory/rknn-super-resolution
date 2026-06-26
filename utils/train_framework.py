"""Shared training framework for MobileOneSR stages."""

from __future__ import annotations

import argparse
import os
import socket
from collections.abc import Callable
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
from torch.utils.tensorboard import SummaryWriter

from data.div2k_loader import build_dataloader
from models.mobileone_sr import MobileOneSR


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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def setup_device(args: argparse.Namespace) -> torch.device:
    """Resolve the torch device from CLI args."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    return device


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
) -> tuple[DataLoader, DistributedSampler | None, DataLoader | None]:
    """Build train/validation dataloaders."""
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


def setup_ddp(rank: int, world_size: int) -> None:
    """Initialize NCCL process group for DDP."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    if "MASTER_PORT" not in os.environ:
        # MASTER_PORT must be set once by the launcher before spawning workers;
        # picking a port independently per rank would break NCCL rendezvous.
        raise RuntimeError(
            "MASTER_PORT is not set. Set it in the launcher process before spawning workers."
        )
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )


def cleanup_ddp() -> None:
    dist.destroy_process_group()


@torch.no_grad()
def validate_ddp(
    model: nn.Module,
    val_loader: DataLoader,
    rank: int,
    world_size: int,
) -> float:
    """Evaluate PSNR on the validation set, aggregated across DDP ranks.

    The returned value is the per-sample average PSNR over the full validation
    set, correctly weighted in case ranks see different sample counts.
    """
    model.eval()
    device = torch.device(f"cuda:{rank}")
    local_psnr = 0.0
    local_samples = 0
    for lr, hr in val_loader:
        lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
        out = torch.clamp(model(lr), 0.0, 255.0)
        mse = torch.mean((out - hr) ** 2, dim=[1, 2, 3])
        psnr = 10 * torch.log10(255.0 * 255.0 / mse)
        local_psnr += psnr.sum().item()
        local_samples += psnr.numel()
    model.train()

    stats = torch.tensor([local_psnr, local_samples], device=rank, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_psnr, total_samples = stats.tolist()
    return total_psnr / total_samples if total_samples > 0 else 0.0


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
    writer: SummaryWriter | None = None,
    save_dir: Path | None = None,
    val_every: int = 10,
    save_every: int = 50,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
    epoch_start: Callable[[int], None] | None = None,
    post_step: Callable[[nn.Module], None] | None = None,
    save_best_extra: Callable[[Path], None] | None = None,
    best_metric_tracker: dict | None = None,
) -> float:
    """Generic DDP epoch-based training loop.

    ``model`` is expected to be a ``DistributedDataParallel`` wrapper.  The
    underlying ``model.module`` is passed to ``loss_fn`` and ``post_step`` so
    that callbacks do not need to handle DDP themselves.
    """
    if best_metric_tracker is None:
        best_metric_tracker = {"value": -1.0}
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
    try:
        for epoch in range(1, epochs + 1):
            if epoch_start is not None:
                epoch_start(epoch)

            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            model.train()
            total_loss = 0.0
            if is_rank0:
                progress.reset(batch_task, total=len(train_loader), visible=True)
                progress.update(batch_task, description=f"Epoch {epoch}")

            for lr, hr in train_loader:
                lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = loss_fn(unwrap, lr, hr)
                loss.backward()
                optimizer.step()
                if post_step is not None:
                    post_step(unwrap)
                total_loss += loss.item()
                if is_rank0:
                    progress.advance(batch_task)

            avg_loss = torch.tensor(total_loss / len(train_loader), device=device)
            if world_size > 1:
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            if writer is not None and is_rank0:
                writer.add_scalar("train/loss", avg_loss.item(), epoch)
                if scheduler is not None:
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
            if is_rank0:
                progress.console.print(
                    f"Epoch {epoch}/{epochs} | loss={avg_loss.item():.4f}",
                    highlight=False,
                )

            if scheduler is not None:
                scheduler.step()

            if val_loader is not None and epoch % val_every == 0:
                psnr = validate_ddp(model, val_loader, rank, world_size)
                if writer is not None and is_rank0:
                    writer.add_scalar("val/psnr", psnr, epoch)
                if is_rank0:
                    progress.console.print(f"  val PSNR={psnr:.2f}", highlight=False)
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
    writer: SummaryWriter | None = None,
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
        writer: Optional TensorBoard writer.
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
                lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
                optimizer.zero_grad()
                loss = loss_fn(model, lr, hr)
                loss.backward()
                optimizer.step()
                if post_step is not None:
                    post_step(model)
                total_loss += loss.item()
                progress.advance(batch_task)

            avg_loss = total_loss / len(train_loader)
            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, epoch)
                if scheduler is not None:
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
            progress.console.print(f"Epoch {epoch}/{epochs} | loss={avg_loss:.4f}", highlight=False)

            if scheduler is not None:
                scheduler.step()

            if val_loader is not None and epoch % val_every == 0:
                psnr = validate(model, val_loader, device)
                if writer is not None:
                    writer.add_scalar("val/psnr", psnr, epoch)
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
