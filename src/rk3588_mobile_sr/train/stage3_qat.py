"""Stage 3: Deploy-before-QAT with fused MobileOne blocks (DDP via torchrun)."""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn

from rk3588_mobile_sr.distributed import distributed_session, unwrap_model, wrap_training_model
from rk3588_mobile_sr.distributed.validation import EarlyStopState, ValidationConfig
from rk3588_mobile_sr.models.qat_utils import (
    bn_recalibrate,
    convert_qat_model,
    prepare_model_for_qat,
)
from rk3588_mobile_sr.train.session import TrainSession
from rk3588_mobile_sr.train.types import TrainConfig, TrainHooks
from rk3588_mobile_sr.utils.run_logger import logger
from rk3588_mobile_sr.utils.train_framework import (
    add_common_args,
    build_model,
    build_train_accel,
    make_optimizer,
    resolve_colorspace,
    resolve_prefetch_batches,
)


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(
        patch_size=144,
        batch_size=1,
        lr=1e-6,
        log_every=500,
        save_dir="./checkpoints/stage3",
        amp=False,
        compile=False,
    )
    parser.add_argument("--stage2_weight", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=15_000)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--phase1_steps", type=int, default=3000)
    parser.add_argument("--phase2_steps", type=int, default=9000)
    parser.add_argument("--clip_min", type=float, default=-1.0)
    parser.add_argument("--clip_max", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--bn_batches", type=int, default=64)
    parser.add_argument("--backend", type=str, default="qnnpack")
    return parser.parse_args()


def weight_clip(model: nn.Module, clip_min: float, clip_max: float) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data.clamp_(clip_min, clip_max)


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters(), strict=True):
            ema_p.copy_(decay * ema_p + (1.0 - decay) * p)


def main():
    args = parse_args()
    with distributed_session() as ctx:
        session = TrainSession(ctx, args, save_dir=args.save_dir, experiment_name="stage3_qat")
        session.prepare()
        session.log_config(
            batch_size=args.batch_size,
            lr=args.lr,
            max_steps=args.max_steps,
            phase1_steps=args.phase1_steps,
            phase2_steps=args.phase2_steps,
        )

        base_model = build_model(args, ctx.device, weight_path=args.stage2_weight)
        recal_bundle = session.build_loaders(train_aug=False)
        bn_recalibrate(base_model, recal_bundle.train, ctx.device, batches=args.bn_batches)
        ctx.barrier()

        example_inputs = (torch.randn(1, 3, args.patch_size, args.patch_size).to(ctx.device),)
        qat_model = prepare_model_for_qat(
            base_model, backend=args.backend, example_inputs=example_inputs
        ).to(ctx.device)

        ema_model = copy.deepcopy(qat_model)
        ema_model.requires_grad_(False)

        model = wrap_training_model(qat_model, ctx, compile_model=False)
        train_model = unwrap_model(model)
        loaders = session.build_loaders()

        optimizer = make_optimizer(unwrap_model(model), args.lr)
        train_accel = build_train_accel(args)
        criterion = nn.L1Loss()

        observer_frozen = False
        fake_quant_frozen = False

        def loss_fn(m, lr, hr):
            return criterion(m(lr), hr)

        def on_step(step: int) -> None:
            nonlocal observer_frozen, fake_quant_frozen
            if not observer_frozen and step > args.phase1_steps:
                train_model.apply(torch.quantization.disable_observer)
                observer_frozen = True
                if ctx.is_main:
                    logger.info("Observer frozen at step {}.", step)
            if not fake_quant_frozen and step > args.phase2_steps:
                train_model.apply(torch.quantization.disable_fake_quant)
                fake_quant_frozen = True
                if ctx.is_main:
                    logger.info("Fake-quant frozen at step {}.", step)

        def post_step(m: nn.Module) -> None:
            weight_clip(m, args.clip_min, args.clip_max)
            update_ema(ema_model, m, args.ema_decay)

        def save_best_extra(best_path: Path) -> None:
            torch.save(ema_model.state_dict(), best_path.with_stem(best_path.stem + "_ema"))

        hooks = TrainHooks(
            loss_fn=loss_fn,
            on_step=on_step,
            post_step=post_step,
            save_best_extra=save_best_extra,
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
            extended=False,
            log_images=not args.no_vis,
            deploy_check=False,
            vis_samples=args.vis_samples,
            vis_max_size=args.vis_max_size,
            colorspace=resolve_colorspace(args),
            data_preview=not args.no_data_preview,
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
                early_stop=EarlyStopState(enabled=False),
                model_diag=False,
            )
            if ctx.is_main:
                torch.save(ema_model.state_dict(), session.save_dir / "last_ema.pth")
                try:
                    quantized = convert_qat_model(train_model)
                    torch.save(
                        quantized.state_dict(),
                        session.save_dir / "quantized_state_dict.pth",
                    )
                    logger.info(
                        "Saved PyTorch INT8 state_dict to quantized_state_dict.pth. "
                        "For RKNN deploy, export FP32 ONNX with: "
                        "rk3588-mobile-sr export-onnx --from-qat --weight <best_ema.pth>."
                    )
                except Exception as exc:
                    logger.warning(
                        "Post-training convert_qat_model failed ({}). "
                        "QAT checkpoints are still valid for --from-qat ONNX export.",
                        exc,
                    )
        finally:
            session.finalize()


if __name__ == "__main__":
    main()
