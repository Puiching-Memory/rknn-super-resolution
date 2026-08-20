"""Single-run FP32-to-QAT training driven by validation plateaus."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from rk3588_mobile_sr.config import load_config
from rk3588_mobile_sr.distributed import (
    SyncBnPolicy,
    distributed_session,
    unwrap_model,
    wrap_training_model,
)
from rk3588_mobile_sr.distributed.validation import EarlyStopState, ValidationConfig
from rk3588_mobile_sr.models.qat_utils import (
    bn_recalibrate,
    convert_qat_model,
    prepare_model_for_qat,
)
from rk3588_mobile_sr.train.session import TrainSession
from rk3588_mobile_sr.train.types import TrainConfig, TrainHooks
from rk3588_mobile_sr.utils.run_logger import logger
from rk3588_mobile_sr.utils.swanlab_logging import get_swanlab_run_id
from rk3588_mobile_sr.utils.train_framework import (
    TrainAccel,
    _normalize_state_dict,
    _training_module_for_state_dict,
    add_common_args,
    build_model,
    build_train_accel,
    resolve_colorspace,
    resolve_model_args,
    resolve_prefetch_batches,
    save_checkpoint_dict,
    training_module_state_dict,
)

FLOAT = "float"
QAT_OBSERVE = "qat_observe"
QAT_STABLE = "qat_stable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(
        patch_size=None,
        batch_size=None,
        log_every=None,
        save_dir="./checkpoints/train",
    )
    parser.add_argument("--qat_patch_size", type=int, default=None)
    parser.add_argument("--qat_batch_size", type=int, default=None)
    parser.add_argument("--float_lr", type=float, default=None)
    parser.add_argument("--qat_lr", type=float, default=None)
    parser.add_argument("--val_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--float_patience", type=int, default=None)
    parser.add_argument("--float_min_delta", type=float, default=None)
    parser.add_argument("--float_min_evaluations", type=int, default=None)
    parser.add_argument("--float_safety_max_steps", type=int, default=None)
    parser.add_argument("--observer_patience", type=int, default=None)
    parser.add_argument("--observer_min_delta", type=float, default=None)
    parser.add_argument("--observer_min_evaluations", type=int, default=None)
    parser.add_argument("--observer_safety_max_steps", type=int, default=None)
    parser.add_argument("--qat_patience", type=int, default=None)
    parser.add_argument("--qat_min_delta", type=float, default=None)
    parser.add_argument("--qat_min_evaluations", type=int, default=None)
    parser.add_argument("--qat_safety_max_steps", type=int, default=None)
    parser.add_argument("--clip_min", type=float, default=None)
    parser.add_argument("--clip_max", type=float, default=None)
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--bn_batches", type=int, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="resume a unified checkpoint; its phase controls graph reconstruction",
    )
    parser.add_argument(
        "--vmaf_model",
        type=str,
        default="1080p",
        help="VMAF v1 model alias or version=/path= override",
    )
    parser.add_argument("--no_vmaf", action="store_true")
    return parser.parse_args()


def resolve_training_args(args: argparse.Namespace) -> argparse.Namespace:
    training = load_config(getattr(args, "config", None)).training
    for name in (
        "patch_size",
        "batch_size",
        "qat_patch_size",
        "qat_batch_size",
        "log_every",
        "float_lr",
        "qat_lr",
        "val_every",
        "save_every",
        "float_patience",
        "float_min_delta",
        "float_min_evaluations",
        "float_safety_max_steps",
        "observer_patience",
        "observer_min_delta",
        "observer_min_evaluations",
        "observer_safety_max_steps",
        "qat_patience",
        "qat_min_delta",
        "qat_min_evaluations",
        "qat_safety_max_steps",
        "clip_min",
        "clip_max",
        "ema_decay",
        "bn_batches",
        "backend",
    ):
        if getattr(args, name, None) is None:
            setattr(args, name, getattr(training, name))
    return args


def validate_training_args(args: argparse.Namespace) -> None:
    for name in (
        "patch_size",
        "batch_size",
        "qat_patch_size",
        "qat_batch_size",
        "log_every",
        "val_every",
        "save_every",
        "float_patience",
        "float_min_evaluations",
        "float_safety_max_steps",
        "observer_patience",
        "observer_min_evaluations",
        "observer_safety_max_steps",
        "qat_patience",
        "qat_min_evaluations",
        "qat_safety_max_steps",
        "bn_batches",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("float_min_delta", "observer_min_delta", "qat_min_delta"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in (0, 1)")
    if args.clip_min >= args.clip_max:
        raise ValueError("clip_min must be less than clip_max")
    for name in ("patch_size", "qat_patch_size"):
        if getattr(args, name) % args.phase_factor:
            raise ValueError(f"{name} must be divisible by phase_factor")
    phase_guards = (
        ("float", args.float_safety_max_steps, args.float_min_evaluations),
        ("observer", args.observer_safety_max_steps, args.observer_min_evaluations),
        ("qat", args.qat_safety_max_steps, args.qat_min_evaluations),
    )
    for phase, safety_steps, min_evaluations in phase_guards:
        if safety_steps < args.val_every * min_evaluations:
            raise ValueError(
                f"{phase}_safety_max_steps must allow at least "
                f"{min_evaluations} validations"
            )


def _early_stop(patience: int, min_delta: float, min_evaluations: int) -> EarlyStopState:
    return EarlyStopState(
        enabled=True,
        patience=patience,
        min_delta=min_delta,
        min_evaluations=min_evaluations,
    )


def _stop_reason(state: EarlyStopState) -> str:
    plateau = (
        state.evaluations >= state.min_evaluations
        and state.patience_counter >= state.patience
    )
    return "validation plateau" if plateau else "safety step cap"


def _checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    phase: str,
    *,
    ema_model: nn.Module | None = None,
    train_accel: TrainAccel | None = None,
    early_stop: EarlyStopState | None = None,
    lr_scheduler: optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> dict:
    checkpoint = {
        "phase": phase,
        "step": step,
        "state_dict": training_module_state_dict(model),
        "optimizer": optimizer.state_dict(),
    }
    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()
    if train_accel is not None and train_accel.scaler is not None:
        checkpoint["scaler"] = train_accel.scaler.state_dict()
    if early_stop is not None:
        checkpoint["early_stop"] = {
            "best_score": early_stop.best_score,
            "patience_counter": early_stop.patience_counter,
            "evaluations": early_stop.evaluations,
        }
    if lr_scheduler is not None:
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
    run_id = get_swanlab_run_id()
    if run_id:
        checkpoint["swanlab_run_id"] = run_id
    return checkpoint


def _load_raw(path: str | Path, device: torch.device) -> dict:
    raw = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(raw, dict) or "state_dict" not in raw or "phase" not in raw:
        raise TypeError(f"Expected a unified training checkpoint in {path}")
    return raw


def _restore(
    raw: dict,
    model: nn.Module,
    optimizer: optim.Optimizer,
    *,
    ema_model: nn.Module | None = None,
    train_accel: TrainAccel | None = None,
    early_stop: EarlyStopState | None = None,
    lr_scheduler: optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> int:
    _training_module_for_state_dict(model).load_state_dict(
        _normalize_state_dict(raw["state_dict"]), strict=True
    )
    optimizer.load_state_dict(raw["optimizer"])
    if ema_model is not None and "ema_state_dict" in raw:
        ema_model.load_state_dict(raw["ema_state_dict"], strict=True)
    if (
        train_accel is not None
        and train_accel.scaler is not None
        and "scaler" in raw
    ):
        train_accel.scaler.load_state_dict(raw["scaler"])
    if early_stop is not None:
        saved_early_stop = raw.get("early_stop", {})
        early_stop.best_score = float(saved_early_stop.get("best_score", -1.0))
        early_stop.patience_counter = int(
            saved_early_stop.get("patience_counter", 0)
        )
        early_stop.evaluations = int(saved_early_stop.get("evaluations", 0))
    if lr_scheduler is not None and "lr_scheduler" in raw:
        lr_scheduler.load_state_dict(raw["lr_scheduler"])
    return int(raw.get("step", 0))


def _weight_clip(model: nn.Module, clip_min: float, clip_max: float) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data.clamp_(clip_min, clip_max)


def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(
            ema_model.parameters(), model.parameters(), strict=True
        ):
            ema_param.lerp_(param, 1.0 - decay)
        for ema_buffer, buffer in zip(
            ema_model.buffers(), model.buffers(), strict=True
        ):
            ema_buffer.copy_(buffer)


def _average_bn_stats(model: nn.Module, world_size: int) -> None:
    """Make recalibrated BN buffers identical before deploy fusion and DDP."""
    if world_size <= 1:
        return
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        for value in (module.running_mean, module.running_var):
            if value is not None:
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                value.div_(world_size)


def _train_config(args: argparse.Namespace, max_steps: int) -> TrainConfig:
    return TrainConfig(
        max_steps=max_steps,
        log_every=args.log_every,
        val_every=args.val_every,
        save_every=args.save_every,
        prefetch_batches=resolve_prefetch_batches(args),
        val_scale=args.scale,
    )


def _plateau_scheduler(
    optimizer: optim.Optimizer,
    *,
    patience: int,
    min_delta: float,
    min_lr: float,
) -> optim.lr_scheduler.ReduceLROnPlateau:
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, patience // 3),
        threshold=min_delta,
        threshold_mode="abs",
        min_lr=min_lr,
    )


def _validation_config(
    args: argparse.Namespace,
    *,
    extended: bool,
    data_preview: bool,
    final_preview: bool,
) -> ValidationConfig:
    return ValidationConfig(
        scale=args.scale,
        extended=extended,
        log_images=not args.no_vis,
        deploy_check=extended and not args.no_model_diag,
        vis_samples=args.vis_samples,
        vis_max_size=args.vis_max_size,
        colorspace=resolve_colorspace(args),
        data_preview=data_preview and not args.no_data_preview,
        final_preview=final_preview,
        compute_vmaf=extended and not args.no_vmaf,
        vmaf_model=args.vmaf_model,
    )


def _phase_hooks(
    model: nn.Module,
    optimizer: optim.Optimizer,
    phase: str,
    criterion: nn.Module,
    *,
    ema_model: nn.Module | None = None,
    train_accel: TrainAccel | None = None,
    post_step=None,
    early_stop: EarlyStopState | None = None,
    lr_scheduler: optim.lr_scheduler.ReduceLROnPlateau | None = None,
) -> TrainHooks:
    def loss_fn(module, lr, hr):
        return criterion(module(lr), hr)

    def save(path: Path, step: int) -> None:
        save_checkpoint_dict(
            _checkpoint(
                model,
                optimizer,
                step,
                phase,
                ema_model=ema_model,
                train_accel=train_accel,
                early_stop=early_stop,
                lr_scheduler=lr_scheduler,
            ),
            path,
        )

    def on_validation(result) -> None:
        if lr_scheduler is not None:
            lr_scheduler.step(result.score)

    return TrainHooks(
        loss_fn=loss_fn,
        post_step=post_step,
        on_save_best=save,
        on_save_step=lambda step, path: save(path, step),
        on_save_last=lambda step, path: save(path, step),
        on_validation=on_validation,
    )


def _close_loader(loader) -> None:
    close = getattr(loader, "close", None)
    if close is not None:
        close()


def main() -> None:
    args = resolve_training_args(resolve_model_args(parse_args()))
    validate_training_args(args)
    resume_raw: dict | None = None

    with distributed_session() as ctx:
        if args.resume:
            resume_raw = _load_raw(args.resume, ctx.device)
        resume_phase = str(resume_raw.get("phase", FLOAT)) if resume_raw else FLOAT
        if resume_phase not in {FLOAT, QAT_OBSERVE, QAT_STABLE}:
            raise ValueError(f"Unknown checkpoint phase: {resume_phase}")

        session = TrainSession(ctx, args, save_dir=args.save_dir, experiment_name="train")
        session.prepare()
        criterion = nn.L1Loss()
        global_step = 0

        try:
            if resume_phase == FLOAT:
                model = build_model(args, ctx.device)
                model = wrap_training_model(
                    model,
                    ctx,
                    compile_model=args.compile,
                    sync_bn=SyncBnPolicy.IF_FLAG,
                    sync_bn_flag=args.sync_bn,
                )
                optimizer = optim.Adam(model.parameters(), lr=args.float_lr, betas=(0.9, 0.999))
                train_accel = build_train_accel(args)
                float_early_stop = _early_stop(
                    args.float_patience,
                    args.float_min_delta,
                    args.float_min_evaluations,
                )
                float_lr_scheduler = _plateau_scheduler(
                    optimizer,
                    patience=args.float_patience,
                    min_delta=args.float_min_delta,
                    min_lr=args.float_lr * 0.01,
                )
                if resume_raw is not None:
                    global_step = _restore(
                        resume_raw,
                        model,
                        optimizer,
                        train_accel=train_accel,
                        early_stop=float_early_stop,
                        lr_scheduler=float_lr_scheduler,
                    )

                loaders = session.build_loaders()
                float_dir = session.save_dir / "float"
                global_step = session.run_trainer(
                    model,
                    optimizer,
                    loaders,
                    _train_config(args, args.float_safety_max_steps),
                    _phase_hooks(
                        model,
                        optimizer,
                        FLOAT,
                        criterion,
                        train_accel=train_accel,
                        early_stop=float_early_stop,
                        lr_scheduler=float_lr_scheduler,
                    ),
                    train_accel=train_accel,
                    validation_config=_validation_config(
                        args, extended=True, data_preview=True, final_preview=False
                    ),
                    early_stop=float_early_stop,
                    model_diag=not args.no_model_diag,
                    global_step=global_step,
                    save_dir=float_dir,
                )
                if ctx.is_main:
                    logger.info(
                        "Float phase ended at step {} ({}) -> QAT observer phase.",
                        global_step,
                        _stop_reason(float_early_stop),
                    )
                _close_loader(loaders.train)
                del loaders, optimizer, model
                ctx.barrier()
                float_best = float_dir / "best.pth"
                if not float_best.is_file():
                    raise RuntimeError("Float phase produced no best checkpoint")
                base_model = build_model(args, ctx.device, weight_path=str(float_best))
            else:
                base_model = build_model(args, ctx.device)

            args.patch_size = args.qat_patch_size
            args.batch_size = args.qat_batch_size

            if resume_phase == FLOAT:
                recal_bundle = session.build_loaders(train_aug=False)
                bn_recalibrate(
                    base_model,
                    recal_bundle.train,
                    ctx.device,
                    batches=args.bn_batches,
                )
                _average_bn_stats(base_model, ctx.world_size)
                _close_loader(recal_bundle.train)
                del recal_bundle
                ctx.barrier()

            example_inputs = (
                torch.randn(1, 3, args.qat_patch_size, args.qat_patch_size).to(ctx.device),
            )
            qat_model = prepare_model_for_qat(
                base_model,
                backend=args.backend,
                example_inputs=example_inputs,
            ).to(ctx.device)
            ema_model = copy.deepcopy(qat_model)
            ema_model.requires_grad_(False)
            model = wrap_training_model(qat_model, ctx, compile_model=False)
            train_model = unwrap_model(model)
            optimizer = optim.Adam(train_model.parameters(), lr=args.qat_lr, betas=(0.9, 0.999))
            qat_accel = TrainAccel(enabled=False, dtype=torch.float32, scaler=None)

            if resume_raw is not None and resume_phase != FLOAT:
                resume_early_stop = _early_stop(
                    args.observer_patience
                    if resume_phase == QAT_OBSERVE
                    else args.qat_patience,
                    args.observer_min_delta
                    if resume_phase == QAT_OBSERVE
                    else args.qat_min_delta,
                    args.observer_min_evaluations
                    if resume_phase == QAT_OBSERVE
                    else args.qat_min_evaluations,
                )
                resume_lr_scheduler = _plateau_scheduler(
                    optimizer,
                    patience=(
                        args.observer_patience
                        if resume_phase == QAT_OBSERVE
                        else args.qat_patience
                    ),
                    min_delta=(
                        args.observer_min_delta
                        if resume_phase == QAT_OBSERVE
                        else args.qat_min_delta
                    ),
                    min_lr=args.qat_lr * 0.01,
                )
                global_step = _restore(
                    resume_raw,
                    model,
                    optimizer,
                    ema_model=ema_model,
                    early_stop=resume_early_stop,
                    lr_scheduler=resume_lr_scheduler,
                )
            else:
                resume_early_stop = None
                resume_lr_scheduler = None

            def post_step(module: nn.Module) -> None:
                _weight_clip(module, args.clip_min, args.clip_max)
                _update_ema(ema_model, module, args.ema_decay)

            loaders = session.build_loaders()
            if resume_phase != QAT_STABLE:
                observe_early_stop = resume_early_stop or _early_stop(
                    args.observer_patience,
                    args.observer_min_delta,
                    args.observer_min_evaluations,
                )
                observe_stop = global_step + args.observer_safety_max_steps
                observe_lr_scheduler = resume_lr_scheduler or _plateau_scheduler(
                    optimizer,
                    patience=args.observer_patience,
                    min_delta=args.observer_min_delta,
                    min_lr=args.qat_lr * 0.01,
                )
                global_step = session.run_trainer(
                    model,
                    optimizer,
                    loaders,
                    _train_config(args, observe_stop),
                    _phase_hooks(
                        model,
                        optimizer,
                        QAT_OBSERVE,
                        criterion,
                        ema_model=ema_model,
                        post_step=post_step,
                        early_stop=observe_early_stop,
                        lr_scheduler=observe_lr_scheduler,
                    ),
                    train_accel=qat_accel,
                    validation_config=_validation_config(
                        args, extended=False, data_preview=False, final_preview=False
                    ),
                    early_stop=observe_early_stop,
                    model_diag=False,
                    global_step=global_step,
                    save_dir=session.save_dir / "qat_observe",
                )
                if ctx.is_main:
                    logger.info(
                        "QAT observer phase ended at step {} ({}) -> freeze observers.",
                        global_step,
                        _stop_reason(observe_early_stop),
                    )

            train_model.apply(torch.quantization.disable_observer)
            ema_model.apply(torch.quantization.disable_observer)
            stable_stop = global_step + args.qat_safety_max_steps
            stable_early_stop = (
                resume_early_stop
                if resume_phase == QAT_STABLE
                else _early_stop(
                    args.qat_patience,
                    args.qat_min_delta,
                    args.qat_min_evaluations,
                )
            )
            stable_lr_scheduler = (
                resume_lr_scheduler
                if resume_phase == QAT_STABLE
                else _plateau_scheduler(
                    optimizer,
                    patience=args.qat_patience,
                    min_delta=args.qat_min_delta,
                    min_lr=args.qat_lr * 0.01,
                )
            )

            def save_best_ema(best_path: Path) -> None:
                torch.save(
                    ema_model.state_dict(),
                    best_path.with_name("best_ema.pth"),
                )

            stable_hooks = _phase_hooks(
                model,
                optimizer,
                QAT_STABLE,
                criterion,
                ema_model=ema_model,
                post_step=post_step,
                early_stop=stable_early_stop,
                lr_scheduler=stable_lr_scheduler,
            )
            stable_hooks.save_best_extra = save_best_ema
            global_step = session.run_trainer(
                model,
                optimizer,
                loaders,
                _train_config(args, stable_stop),
                stable_hooks,
                train_accel=qat_accel,
                validation_config=_validation_config(
                    args, extended=False, data_preview=False, final_preview=True
                ),
                early_stop=stable_early_stop,
                model_diag=False,
                global_step=global_step,
                save_dir=session.save_dir,
            )
            _close_loader(loaders.train)

            if ctx.is_main:
                torch.save(ema_model.state_dict(), session.save_dir / "last_ema.pth")
                try:
                    quantized = convert_qat_model(train_model)
                    torch.save(
                        quantized.state_dict(),
                        session.save_dir / "quantized_state_dict.pth",
                    )
                except Exception as exc:
                    logger.warning("Post-training QAT conversion failed: {}", exc)
                logger.info(
                    "Training completed at step {} ({}).",
                    global_step,
                    _stop_reason(stable_early_stop),
                )
        finally:
            session.finalize()


if __name__ == "__main__":
    main()
