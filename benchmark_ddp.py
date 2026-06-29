"""Quick multi-GPU training step benchmark via torchrun."""

from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.train_framework import build_loaders, build_model, cleanup_ddp, setup_ddp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", default="data/DIV2K_train_HR")
    parser.add_argument("--lr_dir", default="data/DIV2K_train_LR_bicubic/X3")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--use_dali", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rank = setup_ddp()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    ns = argparse.Namespace(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        val_hr_dir=None,
        val_lr_dir=None,
        scale=3,
        num_channels=32,
        num_blocks=8,
        num_conv_branches=4,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=4,
        use_dali=args.use_dali,
        dali_num_threads=4,
        samples_per_image=1,
    )

    model = build_model(ns, device)
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
    train_loader, _, _ = build_loaders(
        ns, device, train_aug=True, distributed=True, rank=rank, world_size=world_size
    )

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def run_steps(n: int) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        count = 0
        for lr, hr in train_loader:
            lr = lr.to(device, non_blocking=True) if lr.device != device else lr
            hr = hr.to(device, non_blocking=True) if hr.device != device else hr
            optimizer.zero_grad(set_to_none=True)
            out = model(lr)
            loss = criterion(out, hr)
            loss.backward()
            optimizer.step()
            count += 1
            if count >= n:
                break
        torch.cuda.synchronize()
        dist.barrier()
        return time.perf_counter() - t0

    for _ in range(args.warmup):
        run_steps(1)

    elapsed = run_steps(args.steps)
    global_batch = args.batch_size * world_size
    img_per_s = args.steps * global_batch / elapsed

    if rank == 0:
        backend = "DALI" if args.use_dali else "PyTorch"
        print(
            f"[{backend}] {world_size} GPUs x batch {args.batch_size} | "
            f"{args.steps} steps in {elapsed:.2f}s | "
            f"{args.steps / elapsed:.2f} step/s | {img_per_s:.0f} img/s"
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()
