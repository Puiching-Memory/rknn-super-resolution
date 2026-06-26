"""Stage 1: FP32 baseline training with L1 loss (DDP)."""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from utils.train_framework import (
    add_common_args,
    build_loaders,
    build_model,
    cleanup_ddp,
    find_free_port,
    save_checkpoint_dict,
    setup_ddp,
)


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(save_dir="./checkpoints/stage1")
    parser.add_argument(
        "--steps", type=int, default=None, help="total training steps; overrides --epochs if set"
    )
    parser.add_argument("--val_every", type=int, default=1000, help="validate every N steps")
    parser.add_argument(
        "--save_every", type=int, default=5000, help="save checkpoint every N steps"
    )
    parser.add_argument(
        "--log_every", type=int, default=100, help="log training loss every N steps"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="number of GPUs; default uses all visible CUDA devices",
    )
    return parser.parse_args()


best_metric_tracker = {"value": -1.0}


def save_full_checkpoint(model, optimizer, scheduler, step, save_dir, filename, best_metric=None):
    if best_metric is not None:
        if best_metric <= best_metric_tracker["value"]:
            return
        best_metric_tracker["value"] = best_metric
    ckpt = {
        "step": step,
        "state_dict": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    save_checkpoint_dict(ckpt, save_dir / filename)


def validate_ddp(model, val_loader, rank, world_size):
    device = torch.device(f"cuda:{rank}")
    model.eval()
    local_psnr = 0.0
    local_samples = 0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            out = torch.clamp(model(lr), 0.0, 255.0)
            mse = torch.mean((out - hr) ** 2, dim=[1, 2, 3])
            psnr = 10 * torch.log10(255.0 * 255.0 / mse)
            local_psnr += psnr.sum().item()
            local_samples += psnr.numel()
    model.train()

    stats = torch.tensor([local_psnr, local_samples], device=rank, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_psnr, total_samples = stats.tolist()
    return total_psnr / total_samples


def train_steps(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    rank,
    world_size,
    global_step,
    max_steps,
    val_loader,
    writer,
    save_dir,
    args,
    sampler=None,
):
    model.train()

    progress = None
    task_id = None
    if rank == 0:
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
        task_id = progress.add_task("Training", total=max_steps, completed=global_step)

    local_step = 0
    epoch_loss = 0.0
    epoch = global_step // len(dataloader) + 1
    try:
        while global_step < max_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch += 1
            for lr, hr in dataloader:
                lr, hr = lr.cuda(rank, non_blocking=True), hr.cuda(rank, non_blocking=True)
                optimizer.zero_grad()
                out = model(lr)
                loss = criterion(out, hr)
                loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1
                local_step += 1
                epoch_loss += loss.item()

                if rank == 0 and progress is not None:
                    progress.update(task_id, advance=1)

                if global_step % args.log_every == 0:
                    avg_loss = torch.tensor(epoch_loss / local_step, device=rank)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                    if rank == 0:
                        writer.add_scalar("train/loss", avg_loss.item(), global_step)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                        progress.console.print(
                            f"Step {global_step}/{max_steps} | loss={avg_loss.item():.4f} | "
                            f"lr={optimizer.param_groups[0]['lr']:.6f}",
                            highlight=False,
                        )
                    epoch_loss = 0.0
                    local_step = 0

                if global_step % args.val_every == 0 and val_loader is not None:
                    psnr = validate_ddp(model, val_loader, rank, world_size)
                    if rank == 0:
                        writer.add_scalar("val/psnr", psnr, global_step)
                        progress.console.print(
                            f"Step {global_step}/{max_steps} | val PSNR={psnr:.2f}",
                            highlight=False,
                        )
                        save_full_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            save_dir,
                            "best.pth",
                            best_metric=psnr,
                        )

                if global_step % args.save_every == 0:
                    if rank == 0:
                        save_full_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            save_dir,
                            f"step_{global_step}.pth",
                        )

                if global_step >= max_steps:
                    break
    finally:
        if progress is not None:
            progress.stop()

    return global_step


def run_worker(rank, world_size, args):
    setup_ddp(rank, world_size)

    save_dir = Path(args.save_dir)
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    device = torch.device(f"cuda:{rank}")
    model = build_model(args, device)
    # Synchronize BN running stats across GPUs; otherwise each rank keeps its
    # own batch statistics and the saved checkpoint depends on rank 0's subset.
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    train_loader, train_sampler, val_loader = build_loaders(
        args,
        device,
        train_aug=True,
        val_bs=1,
        distributed=True,
    )

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    if args.steps is not None:
        max_steps = args.steps
    else:
        steps_per_epoch = len(train_loader)
        max_steps = args.epochs * steps_per_epoch

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    writer = SummaryWriter(log_dir=str(save_dir / "logs")) if rank == 0 else None

    global_step = 0
    global_step = train_steps(
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        rank,
        world_size,
        global_step,
        max_steps,
        val_loader,
        writer,
        save_dir,
        args,
        sampler=train_sampler,
    )

    if rank == 0:
        save_full_checkpoint(model, optimizer, scheduler, global_step, save_dir, "last.pth")
        writer.close()

    cleanup_ddp()


def main():
    args = parse_args()
    world_size = args.world_size or torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available")
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(find_free_port())
    mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
