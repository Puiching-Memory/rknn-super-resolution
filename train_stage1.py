"""Stage 1: L1 baseline training with optional AMP + torch.compile (DDP via torchrun)."""

import argparse
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
)
from torch.nn.parallel import DistributedDataParallel as DDP

from data.prefetch import BatchPrefetcher
from utils.model_diagnostics import (
    ForwardDiagnosticsTracker,
    check_deploy_consistency,
    collect_training_diagnostics,
)
from utils.sr_metrics import validate_ddp_extended
from utils.swanlab_logging import (
    finish_swanlab,
    log_metrics,
    log_validation_sr_images,
    setup_swanlab,
)
from utils.traceml_profiling import finish_traceml, setup_traceml, trace_training_step
from utils.train_framework import (
    TrainAccel,
    _average_metrics_across_ranks,
    add_common_args,
    amp_autocast,
    build_loaders,
    build_model,
    build_train_accel,
    cleanup_ddp,
    maybe_sync_batchnorm,
    maybe_compile,
    run_backward,
    save_checkpoint_dict,
    setup_ddp,
)


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(save_dir="./checkpoints/stage1")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100_000,
        help="hard upper limit on training steps (safety cap)",
    )
    parser.add_argument("--val_every", type=int, default=1000, help="validate every N steps")
    parser.add_argument(
        "--save_every", type=int, default=5000, help="save checkpoint every N steps"
    )
    parser.add_argument(
        "--log_every", type=int, default=500, help="log training loss every N steps"
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="stop after this many validations without PSNR improvement",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.01,
        help="minimum val PSNR gain to reset patience (dB)",
    )
    parser.add_argument(
        "--no_early_stop",
        action="store_true",
        help="train until --max_steps instead of using validation early stopping",
    )
    return parser.parse_args()


best_metric_tracker = {"value": -1.0}


def save_full_checkpoint(
    model, optimizer, scheduler, step, save_dir, filename, best_metric=None, scaler=None
):
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
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    save_checkpoint_dict(ckpt, save_dir / filename)


def _broadcast_stop(rank: int, should_stop: bool) -> bool:
    flag = torch.tensor([int(should_stop)], device=f"cuda:{rank}")
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def _to_device(lr, hr, device):
    if lr.device == device and hr.device == device:
        return lr, hr
    return lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)


def _log_training_diagnostics(
    model,
    diag_tracker: ForwardDiagnosticsTracker,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    step: int,
) -> None:
    diag = collect_training_diagnostics(model, diag_tracker)
    diag = _average_metrics_across_ranks(
        diag,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    if rank == 0:
        log_metrics(diag, step=step)


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
    save_dir,
    args,
    train_accel: TrainAccel,
):
    model.train()
    use_early_stop = not args.no_early_stop and val_loader is not None
    best_psnr = best_metric_tracker["value"]
    patience_counter = 0
    should_stop = False

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
        )
        progress.start()
        task_id = progress.add_task("Training", total=max_steps, completed=global_step)

    local_step = 0
    window_loss = 0.0
    device = torch.device(f"cuda:{rank}")
    prefetcher = BatchPrefetcher(dataloader, buffer_size=args.prefetch_batches)
    prefetch_stream = torch.cuda.Stream(device=device)
    pending_batch = None
    diag_tracker = None if args.no_model_diag else ForwardDiagnosticsTracker(model)

    def _fetch_batch():
        lr, hr = next(prefetcher)
        with torch.cuda.stream(prefetch_stream):
            lr, hr = _to_device(lr, hr, device)
        return lr, hr

    try:
        pending_batch = _fetch_batch()
        while global_step < max_steps:
            torch.cuda.current_stream(device).wait_stream(prefetch_stream)
            lr, hr = pending_batch
            if global_step + 1 < max_steps:
                pending_batch = _fetch_batch()
            with trace_training_step(model):
                optimizer.zero_grad(set_to_none=True)
                with amp_autocast(train_accel):
                    out = model(lr)
                    loss = criterion(out, hr)
                run_backward(loss, optimizer, train_accel)
                scheduler.step()
            global_step += 1
            local_step += 1
            window_loss += loss.item()

            if rank == 0 and progress is not None:
                progress.update(task_id, advance=1)

            if global_step % args.log_every == 0:
                avg_loss = torch.tensor(window_loss / local_step, device=rank)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                if rank == 0:
                    metrics = {
                        "train/loss": avg_loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                    log_metrics(metrics, step=global_step)
                    progress.console.print(
                        f"Step {global_step} | loss={avg_loss.item():.4f} | "
                        f"lr={optimizer.param_groups[0]['lr']:.6f}",
                        highlight=False,
                    )
                if diag_tracker is not None:
                    _log_training_diagnostics(
                        model,
                        diag_tracker,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        step=global_step,
                    )
                window_loss = 0.0
                local_step = 0

            if global_step % args.val_every == 0 and val_loader is not None:
                psnr, val_metrics = validate_ddp_extended(
                    model,
                    val_loader,
                    rank,
                    world_size,
                    scale=args.scale,
                )
                if rank == 0:
                    improved = psnr > best_psnr + args.early_stop_min_delta
                    if improved:
                        best_psnr = psnr
                        patience_counter = 0
                    elif use_early_stop:
                        patience_counter += 1

                    metrics = {
                        "val/best_psnr": best_psnr,
                    }
                    if val_metrics is not None:
                        metrics.update(val_metrics.to_log_dict())
                    else:
                        metrics["val/psnr"] = psnr
                    if use_early_stop:
                        metrics["early_stop/patience"] = patience_counter
                    if not args.no_model_diag:
                        deploy_metrics = check_deploy_consistency(
                            model,
                            val_loader,
                            device,
                        )
                        if deploy_metrics:
                            metrics.update(deploy_metrics)
                    log_metrics(metrics, step=global_step)

                    if not args.no_vis:
                        log_validation_sr_images(
                            model,
                            val_loader,
                            device,
                            step=global_step,
                            num_samples=args.vis_samples,
                            max_size=args.vis_max_size,
                        )

                    stop_msg = ""
                    if use_early_stop and patience_counter >= args.early_stop_patience:
                        should_stop = True
                        stop_msg = " | early stop"
                    detail = ""
                    if val_metrics is not None:
                        detail = (
                            f" | Y-PSNR={val_metrics.y_psnr:.2f}"
                            f" | SSIM={val_metrics.ssim:.4f}"
                            f" | L1={val_metrics.l1:.2f}"
                        )
                    deploy_note = ""
                    if not args.no_model_diag and "deploy/max_abs_diff" in metrics:
                        deploy_note = (
                            f" | deploy_diff={metrics['deploy/max_abs_diff']:.4f}"
                        )
                    progress.console.print(
                        f"Step {global_step} | val PSNR={psnr:.2f}{detail}{deploy_note} | "
                        f"best={best_psnr:.2f} | patience={patience_counter}/{args.early_stop_patience}"
                        f"{stop_msg}",
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
                        scaler=train_accel.scaler,
                    )

                if world_size > 1:
                    should_stop = _broadcast_stop(rank, should_stop)
                if should_stop:
                    break

            if global_step % args.save_every == 0:
                if rank == 0:
                    save_full_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        save_dir,
                        f"step_{global_step}.pth",
                        scaler=train_accel.scaler,
                    )
    finally:
        prefetcher.close()
        if diag_tracker is not None:
            diag_tracker.close()
        if progress is not None:
            progress.stop()

    return global_step


def main():
    args = parse_args()
    rank = setup_ddp()
    world_size = dist.get_world_size()

    save_dir = Path(args.save_dir)
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    device = torch.device(f"cuda:{rank}")
    model = build_model(args, device)
    model = maybe_sync_batchnorm(model, args)
    model = maybe_compile(model, args)
    model = DDP(model, device_ids=[rank])

    train_accel = build_train_accel(args)
    if rank == 0 and args.compile:
        print("torch.compile enabled (first steps may be slower while graphs compile).")
    if rank == 0 and train_accel.enabled:
        print(f"AMP enabled (dtype={train_accel.dtype}).")

    train_loader, _, val_loader = build_loaders(
        args,
        device,
        train_aug=True,
        val_bs=1,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    setup_swanlab(
        rank=rank,
        save_dir=save_dir,
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment or "stage1",
        config=vars(args),
        disabled=args.no_swanlab,
    )
    setup_traceml(args, rank=rank)

    global_step = 0
    try:
        global_step = train_steps(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            rank,
            world_size,
            global_step,
            args.max_steps,
            val_loader,
            save_dir,
            args,
            train_accel,
        )

        if rank == 0:
            save_full_checkpoint(
                model,
                optimizer,
                scheduler,
                global_step,
                save_dir,
                "last.pth",
                scaler=train_accel.scaler,
            )
    finally:
        finish_traceml(rank=rank)
        finish_swanlab()

    cleanup_ddp()


if __name__ == "__main__":
    main()
