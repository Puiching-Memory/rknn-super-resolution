"""Stage 3: Deploy-before-QAT with fused MobileOne blocks (DDP)."""

import argparse
import copy
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from models.qat_utils import bn_recalibrate, convert_qat_model, prepare_model_for_qat
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
    parser.set_defaults(
        patch_size=144,
        batch_size=1,
        epochs=150,
        lr=1e-6,
        save_dir="./checkpoints/stage3",
    )
    parser.add_argument("--stage2_weight", type=str, required=True)
    parser.add_argument("--phase1", type=int, default=30)
    parser.add_argument("--phase2", type=int, default=90)
    parser.add_argument("--clip_min", type=float, default=-1.0)
    parser.add_argument("--clip_max", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--bn_batches", type=int, default=64)
    parser.add_argument("--backend", type=str, default="qnnpack")
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="number of GPUs; default uses all visible CUDA devices",
    )
    return parser.parse_args()


def weight_clip(model: nn.Module, clip_min: float, clip_max: float) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data.clamp_(clip_min, clip_max)


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.copy_(decay * ema_p + (1.0 - decay) * p)


def run_worker(rank, world_size, args):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    save_dir = Path(args.save_dir)
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    base_model = build_model(args, device, weight_path=args.stage2_weight)

    # BN recalibration before fusion. Use DistributedSampler so every rank
    # sees a disjoint subset; there are no BNs left after fusion, but this
    # keeps the recalibration consistent with DDP training.
    recal_loader, _, _ = build_loaders(
        args, device, train_aug=False, val_bs=1, distributed=True
    )
    bn_recalibrate(base_model, recal_loader, device, batches=args.bn_batches)

    # Prepare QAT on fused deploy graph
    example_inputs = (torch.randn(1, 3, args.patch_size, args.patch_size).to(device),)
    model = prepare_model_for_qat(base_model, backend=args.backend, example_inputs=example_inputs)
    model = model.to(device)

    train_loader, _, val_loader = build_loaders(
        args, device, train_aug=True, val_bs=1, distributed=True
    )

    optimizer = make_optimizer(model, args.lr)
    criterion = nn.L1Loss()
    writer = SummaryWriter(log_dir=str(save_dir / "logs")) if rank == 0 else None

    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)

    # DDP must be set up after EMA snapshot so the EMA does not carry DDP state.
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    train_model = model.module if world_size > 1 else model

    def loss_fn(m, lr, hr):
        return criterion(m(lr), hr)

    def epoch_start(epoch):
        if epoch == args.phase1 + 1:
            train_model.apply(torch.quantization.disable_observer)
            if rank == 0:
                print("Observer frozen.")
        if epoch == args.phase2 + 1:
            train_model.apply(torch.quantization.disable_fake_quant)
            if rank == 0:
                print("Fake-quant frozen.")

    def post_step(m):
        weight_clip(m, args.clip_min, args.clip_max)
        update_ema(ema_model, m, args.ema_decay)

    def save_best_extra(best_path: Path):
        torch.save(ema_model.state_dict(), best_path.with_stem(best_path.stem + "_ema"))

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
        epoch_start=epoch_start,
        post_step=post_step,
        save_best_extra=save_best_extra,
    )

    if rank == 0:
        torch.save(ema_model.state_dict(), save_dir / "last_ema.pth")

        # Save quantized model
        quantized = convert_qat_model(train_model)
        torch.jit.save(torch.jit.script(quantized), str(save_dir / "quantized.pt"))
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
