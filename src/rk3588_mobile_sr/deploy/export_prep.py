"""Prepare float checkpoints for ONNX export (BN recalibrate, safe fuse, weight clip)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rk3588_mobile_sr.models import MobileOneSR
from rk3588_mobile_sr.models.qat_utils import bn_recalibrate


def clip_deploy_weights(model: nn.Module, clip_min: float, clip_max: float) -> None:
    """Clamp fused Conv/Linear weights after deploy fuse."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data.clamp_(clip_min, clip_max)


def fused_weight_report(model: MobileOneSR) -> dict[str, float]:
    """Max fused |weight| per body block and out_conv (deploy graph only)."""
    report: dict[str, float] = {}
    for i, block in enumerate(model.body):
        if block.inference_mode:
            report[f"body.{i}"] = float(block.reparam_conv.weight.abs().max().item())
    report["out_conv"] = float(model.out_conv.weight.abs().max().item())
    return report


def _load_calib_batch(
    path: Path,
    *,
    input_h: int,
    input_w: int,
    phase_factor: int,
    device: torch.device,
) -> torch.Tensor:
    """Load phase-packed YCbCr calibration data and restore the training input."""
    array = np.load(path)
    phases = torch.from_numpy(array)
    if phases.ndim == 3:
        phases = phases.unsqueeze(0)
    expected_channels = 3 * phase_factor * phase_factor
    if phases.ndim != 4 or phases.shape[1] != expected_channels:
        raise ValueError(
            f"Expected NCHW calibration with {expected_channels} channels, got {phases.shape}"
        )
    lr = torch.nn.functional.pixel_shuffle(phases.to(device).float(), phase_factor)
    if lr.shape[-2:] != (input_h, input_w):
        raise ValueError(
            f"Expected calibration size {(input_h, input_w)}, got {tuple(lr.shape[-2:])}"
        )
    return lr


def recalibrate_bn_from_calib_list(
    model: nn.Module,
    calib_list: str | Path,
    *,
    input_h: int,
    input_w: int,
    device: torch.device,
    batches: int,
) -> int:
    """Refresh BN running stats using paths listed in a calib text file."""
    paths = [line.strip() for line in Path(calib_list).read_text().splitlines() if line.strip()]
    if not paths:
        raise ValueError(f"No calibration paths in {calib_list}")

    model.train()

    class _CalibLoader:
        def __init__(self, image_paths: list[str]):
            self.image_paths = image_paths

        def __iter__(self):
            for path in self.image_paths:
                lr = _load_calib_batch(
                    Path(path),
                    input_h=input_h,
                    input_w=input_w,
                    phase_factor=model.phase_factor,
                    device=device,
                )
                yield lr, lr

    n = min(batches, len(paths))
    bn_recalibrate(model, _CalibLoader(paths[:n]), device, batches=n)
    return n


def prepare_float_for_export(
    model: MobileOneSR,
    *,
    device: torch.device,
    calib_list: str | Path | None,
    input_h: int,
    input_w: int,
    bn_batches: int,
    identity_var_floor: float,
    clip_min: float | None,
    clip_max: float | None,
    do_bn_recalibrate: bool,
) -> dict[str, float]:
    """BN recalibrate (optional) ? deploy fuse ? weight clip (optional)."""
    if do_bn_recalibrate:
        if calib_list is None:
            raise ValueError("calib_list is required when BN recalibration is enabled.")
        n = recalibrate_bn_from_calib_list(
            model,
            calib_list,
            input_h=input_h,
            input_w=input_w,
            device=device,
            batches=bn_batches,
        )
        print(f"--> BN recalibrated on {n} calibration images")

    model.switch_to_deploy(identity_var_floor=identity_var_floor)
    if identity_var_floor > 0.0:
        print(f"--> Deploy fuse with identity_var_floor={identity_var_floor}")

    if clip_min is not None and clip_max is not None:
        clip_deploy_weights(model, clip_min, clip_max)
        print(f"--> Fused weights clipped to [{clip_min}, {clip_max}]")

    model.eval()
    report = fused_weight_report(model)
    peak = max(report.values()) if report else 0.0
    print(f"--> Fused weight peaks: {report} (max={peak:.4f})")
    return report
