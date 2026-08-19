"""Stage 1: L1 baseline training with optional AMP + torch.compile (DDP via torchrun)."""

import argparse
from pathlib import Path

import torch.nn as nn
import torch.optim as optim

from rk3588_mobile_sr.distributed import (
    SyncBnPolicy,
    distributed_session,
    wrap_training_model,
)
from rk3588_mobile_sr.distributed.validation import EarlyStopState, ValidationConfig
from rk3588_mobile_sr.train.session import TrainSession
from rk3588_mobile_sr.train.types import TrainConfig, TrainHooks
from rk3588_mobile_sr.utils.run_logger import logger
from rk3588_mobile_sr.utils.swanlab_logging import get_swanlab_run_id
from rk3588_mobile_sr.utils.train_framework import (
    add_common_args,
    build_model,
    build_train_accel,
    load_training_checkpoint,
    resolve_colorspace,
    resolve_prefetch_batches,
    save_checkpoint_dict,
    training_module_state_dict,
)


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(save_dir="./checkpoints/stage1", log_every=500)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.1,
        help="min VMAF improvement to reset early-stop patience (VMAF scale ~0-100)",
    )
    parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument(
        "--vmaf_model",
        type=str,
        default="1080p",
        help="VMAF v1 model alias: 1080p|phone|phone_hfr|1080p_hfr|4k|4k_3h "
        "(see Netflix models_v1.md); or version=/path= override",
    )
    parser.add_argument(
        "--no_vmaf",
        action="store_true",
        help="disable VMAF; fall back to PSNR as the primary val / early-stop metric",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="resume from a full stage1 checkpoint (best.pth / step_*.pth)",
    )
    return parser.parse_args()


def _full_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    *,
    scaler=None,
) -> dict:
    ckpt = {
        "step": step,
        "state_dict": training_module_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    run_id = get_swanlab_run_id()
    if run_id:
        ckpt["swanlab_run_id"] = run_id
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    return ckpt


def main():
    args = parse_args()
    with distributed_session() as ctx:
        session = TrainSession(ctx, args, save_dir=args.save_dir, experiment_name="stage1")
        session.prepare()

        train_accel = build_train_accel(args)
        if ctx.is_main and args.compile:
            logger.info("torch.compile enabled (first steps may be slower while graphs compile).")
        if ctx.is_main and train_accel.enabled:
            logger.info(f"AMP enabled (dtype={train_accel.dtype}).")

        model = build_model(args, ctx.device)
        model = wrap_training_model(
            model,
            ctx,
            compile_model=args.compile,
            sync_bn=SyncBnPolicy.IF_FLAG,
            sync_bn_flag=args.sync_bn,
        )

        loaders = session.build_loaders()
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

        global_step = 0
        if args.resume:
            global_step = load_training_checkpoint(
                args.resume,
                model,
                optimizer,
                scheduler,
                train_accel=train_accel,
                device=ctx.device,
            )
            if ctx.is_main:
                logger.info("resumed from {} at step {}", args.resume, global_step)

        def loss_fn(m, lr, hr):
            return criterion(m(lr), hr)

        def on_save_best(path: Path, step: int) -> None:
            save_checkpoint_dict(
                _full_checkpoint(
                    model, optimizer, scheduler, step, scaler=train_accel.scaler
                ),
                path,
            )

        def on_save_step(step: int, path: Path) -> None:
            save_checkpoint_dict(
                _full_checkpoint(model, optimizer, scheduler, step, scaler=train_accel.scaler),
                path,
            )

        def on_save_last(step: int, path: Path) -> None:
            save_checkpoint_dict(
                _full_checkpoint(model, optimizer, scheduler, step, scaler=train_accel.scaler),
                path,
            )

        hooks = TrainHooks(
            loss_fn=loss_fn,
            scheduler=scheduler,
            on_save_best=on_save_best,
            on_save_step=on_save_step,
            on_save_last=on_save_last,
        )

        early_stop = EarlyStopState(
            enabled=not args.no_early_stop and loaders.val is not None,
            patience=args.early_stop_patience,
            min_delta=args.early_stop_min_delta,
        )

        config = TrainConfig(
            max_steps=args.max_steps,
            log_every=args.log_every,
            val_every=args.val_every,
            save_every=args.save_every,
            prefetch_batches=resolve_prefetch_batches(args),
            val_scale=args.scale,
        )
        val_config = ValidationConfig(
            scale=args.scale,
            extended=True,
            log_images=not args.no_vis,
            deploy_check=not args.no_model_diag,
            vis_samples=args.vis_samples,
            vis_max_size=args.vis_max_size,
            colorspace=resolve_colorspace(args),
            data_preview=not args.no_data_preview,
            compute_vmaf=not args.no_vmaf,
            vmaf_model=args.vmaf_model,
        )

        try:
            session.run_trainer(
                model,
                optimizer,
                loaders,
                config,
                hooks,
                train_accel=train_accel,
                validation_config=val_config,
                early_stop=early_stop,
                model_diag=not args.no_model_diag,
                global_step=global_step,
            )
        finally:
            session.finalize()


if __name__ == "__main__":
    main()
