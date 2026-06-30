"""Stage 2: Fidelity finetuning + teacher distillation (DDP via torchrun)."""

import argparse

from rk3588_mobile_sr.distributed import (
    SyncBnPolicy,
    distributed_session,
    unwrap_model,
    wrap_training_model,
)
from rk3588_mobile_sr.distributed.validation import EarlyStopState, ValidationConfig
from rk3588_mobile_sr.losses import Stage2Loss
from rk3588_mobile_sr.models.teacher_wrapper import load_teacher
from rk3588_mobile_sr.train.session import TrainSession
from rk3588_mobile_sr.train.types import TrainConfig, TrainHooks
from rk3588_mobile_sr.utils.train_framework import (
    add_common_args,
    build_model,
    build_train_accel,
    make_optimizer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(
        patch_size=160,
        batch_size=64,
        lr=6e-5,
        log_every=500,
        save_dir="./checkpoints/stage2",
    )
    parser.add_argument("--max_steps", type=int, default=80_000)
    parser.add_argument("--val_every", type=int, default=4000)
    parser.add_argument("--save_every", type=int, default=20_000)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.005)
    parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument("--teacher_arch", type=str, required=True, choices=["real_esrgan", "edsr"])
    parser.add_argument("--teacher_weight", type=str, required=True)
    parser.add_argument("--stage1_weight", type=str, required=True)
    parser.add_argument("--lambda_dct", type=float, default=0.02)
    parser.add_argument("--lambda_kd", type=float, default=0.03)
    parser.add_argument("--no_val_lpips", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with distributed_session() as ctx:
        session = TrainSession(ctx, args, save_dir=args.save_dir, experiment_name="stage2")
        session.prepare()
        session.log_config(
            batch_size=args.batch_size,
            lr=args.lr,
            patch_size=args.patch_size,
            max_steps=args.max_steps,
            world_size=ctx.world_size,
        )

        model = build_model(args, ctx.device, weight_path=args.stage1_weight)
        model = wrap_training_model(
            model,
            ctx,
            compile_model=args.compile,
            sync_bn=SyncBnPolicy.ALWAYS,
        )

        train_accel = build_train_accel(args)
        teacher = load_teacher(
            args.teacher_arch,
            args.teacher_weight,
            scale=args.scale,
            device=str(ctx.device),
            compile_model=args.compile,
        )
        stage2_loss = Stage2Loss(lambda_dct=args.lambda_dct, lambda_kd=args.lambda_kd)
        loaders = session.build_loaders()

        def loss_fn(m, lr, hr):
            pred = m(lr)
            tea = teacher(lr)
            out = stage2_loss(pred, hr, tea)
            return out.total, out.log_dict()

        optimizer = make_optimizer(unwrap_model(model), args.lr)
        hooks = TrainHooks(loss_fn=loss_fn)

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
            prefetch_batches=args.prefetch_batches,
            val_scale=args.scale,
        )
        val_config = ValidationConfig(
            scale=args.scale,
            extended=True,
            compute_lpips=not args.no_val_lpips,
            log_images=not args.no_vis,
            deploy_check=not args.no_model_diag,
            vis_samples=args.vis_samples,
            vis_max_size=args.vis_max_size,
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
            )
        finally:
            session.finalize()


if __name__ == "__main__":
    main()
