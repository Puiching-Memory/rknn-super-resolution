"""Stage 2: Fidelity finetuning + teacher distillation (DDP via torchrun)."""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from losses import Stage2Loss
from models.teacher_wrapper import load_teacher
from utils.swanlab_logging import finish_swanlab, setup_swanlab
from utils.traceml_profiling import finish_traceml, setup_traceml
from utils.train_framework import (
    add_common_args,
    build_loaders,
    build_model,
    build_train_accel,
    cleanup_ddp,
    make_optimizer,
    maybe_compile,
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
        "--no_val_lpips",
        action="store_true",
        help="skip LPIPS during validation (faster)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rank = setup_ddp()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    save_dir = Path(args.save_dir)
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    model = build_model(args, device, weight_path=args.stage1_weight)
    # MobileOne blocks contain BN on every branch; keep stats synchronized.
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = maybe_compile(model, args)
    model = DDP(model, device_ids=[rank])

    train_accel = build_train_accel(args)

    teacher = load_teacher(
        args.teacher_arch, args.teacher_weight, scale=args.scale, device=str(device)
    )
    stage2_loss = Stage2Loss(lambda_dct=args.lambda_dct, lambda_kd=args.lambda_kd)

    train_loader, _, val_loader = build_loaders(
        args, device, train_aug=True, val_bs=1, distributed=True
    )

    def loss_fn(m, lr, hr):
        pred = m(lr)
        with torch.no_grad():
            tea = teacher(lr)
        out = stage2_loss(pred, hr, tea)
        return out.total, out.log_dict()

    optimizer = make_optimizer(model.module, args.lr)

    setup_swanlab(
        rank=rank,
        save_dir=save_dir,
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment or "stage2",
        config=vars(args),
        disabled=args.no_swanlab,
    )
    setup_traceml(args, rank=rank)
    try:
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
            save_dir=save_dir,
            val_every=10,
            save_every=50,
            vis_samples=args.vis_samples,
            log_images=not args.no_vis,
            vis_max_size=args.vis_max_size,
            train_accel=train_accel,
            val_scale=args.scale,
            extended_val=True,
            val_lpips=not args.no_val_lpips,
            model_diag=not args.no_model_diag,
        )
    finally:
        finish_traceml(rank=rank)
        finish_swanlab()

    cleanup_ddp()


if __name__ == "__main__":
    main()
