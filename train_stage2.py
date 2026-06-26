"""Stage 2: Fidelity finetuning + teacher distillation (DDP)."""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from losses import CharbonnierLoss, ConfidenceWeightedKDLoss, DCTLoss
from models.teacher_wrapper import load_teacher
from utils.train_framework import (
    add_common_args,
    build_loaders,
    build_model,
    cleanup_ddp,
    find_free_port,
    make_optimizer,
    setup_ddp,
    train_epochs_ddp,
)


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(patch_size=160, epochs=200, lr=3e-5, save_dir="./checkpoints/stage2")
    parser.add_argument("--teacher_arch", type=str, required=True, choices=["real_esrgan", "edsr"])
    parser.add_argument("--teacher_weight", type=str, required=True)
    parser.add_argument("--stage1_weight", type=str, required=True)
    parser.add_argument("--lambda_dct", type=float, default=0.02)
    parser.add_argument("--lambda_kd", type=float, default=0.03)
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="number of GPUs; default uses all visible CUDA devices",
    )
    return parser.parse_args()


def run_worker(rank, world_size, args):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    save_dir = Path(args.save_dir)
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    model = build_model(args, device, weight_path=args.stage1_weight)
    # MobileOne blocks contain BN on every branch; keep stats synchronized.
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    teacher = load_teacher(
        args.teacher_arch, args.teacher_weight, scale=args.scale, device=str(device)
    )

    train_loader, _, val_loader = build_loaders(
        args, device, train_aug=True, val_bs=1, distributed=True
    )

    charbonnier = CharbonnierLoss()
    dct = DCTLoss()
    kd = ConfidenceWeightedKDLoss()

    def loss_fn(m, lr, hr):
        pred = m(lr)
        with torch.no_grad():
            tea = teacher(lr)
        loss = charbonnier(pred, hr)
        loss += args.lambda_dct * dct(pred, hr)
        loss += args.lambda_kd * kd(pred, tea, hr)
        return loss

    optimizer = make_optimizer(model.module, args.lr)
    writer = SummaryWriter(log_dir=str(save_dir / "logs")) if rank == 0 else None

    train_epochs_ddp(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device,
        rank,
        world_size,
        epochs=args.epochs,
        val_loader=val_loader,
        writer=writer,
        save_dir=save_dir,
        val_every=10,
        save_every=50,
    )

    if rank == 0:
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
