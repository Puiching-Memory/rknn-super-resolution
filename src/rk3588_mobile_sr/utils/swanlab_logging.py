"""SwanLab experiment logging helpers for DDP training."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import swanlab
import torch
import torch.nn as nn
import torch.nn.functional as F

from rk3588_mobile_sr.utils.run_logger import logger

_active = False


def setup_swanlab(
    *,
    rank: int,
    save_dir: Path,
    project: str,
    experiment_name: str | None = None,
    config: dict[str, Any] | None = None,
    disabled: bool = False,
) -> None:
    """Initialize SwanLab on rank 0."""
    global _active
    _active = False
    if rank != 0 or disabled:
        return

    try:
        swanlab.init(
            project=project,
            experiment_name=experiment_name,
            config=config or {},
            logdir=str(save_dir / "swanlog"),
        )
        _active = True
        logger.info("swanlab initialized: project={} experiment={}", project, experiment_name)
    except Exception as exc:
        logger.warning("swanlab init failed ({}); continuing without SwanLab logging.", exc)
        _active = False


def log_metrics(metrics: dict[str, Any], *, step: int | None = None) -> None:
    """Log metrics when SwanLab is active on this process."""
    if not _active:
        return
    try:
        if step is not None:
            swanlab.log(metrics, step=step)
        else:
            swanlab.log(metrics)
    except Exception as exc:
        logger.warning("swanlab log failed (step={}): {}", step, exc)


def finish_swanlab() -> None:
    """Finish the SwanLab run on rank 0."""
    global _active
    if not _active:
        return
    swanlab.finish()
    _active = False


def chw_tensor_to_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW float tensor in [0, 255] to HWC uint8 numpy."""
    x = tensor.detach().float().clamp(0.0, 255.0).cpu()
    if x.dim() == 4:
        x = x[0]
    return x.permute(1, 2, 0).numpy().astype(np.uint8)


def _grayscale_to_heat_hwc(gray: np.ndarray) -> np.ndarray:
    """Map grayscale [0, 255] to a black → red → yellow → white heat RGB."""
    t = gray.astype(np.float32) / 255.0
    r = np.clip(3.0 * t, 0.0, 1.0)
    g = np.clip(3.0 * t - 1.0, 0.0, 1.0)
    b = np.clip(3.0 * t - 2.0, 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _abs_diff_heatmap(
    a_hwc: np.ndarray,
    b_hwc: np.ndarray,
    *,
    gain: float = 8.0,
) -> np.ndarray:
    """RGB heatmap of amplified mean-channel |a - b|."""
    diff = np.abs(a_hwc.astype(np.float32) - b_hwc.astype(np.float32)).mean(axis=-1)
    return _grayscale_to_heat_hwc(np.clip(diff * gain, 0.0, 255.0).astype(np.uint8))


def _mean_abs_diff_map(a_hwc: np.ndarray, b_hwc: np.ndarray) -> np.ndarray:
    return np.abs(a_hwc.astype(np.float32) - b_hwc.astype(np.float32)).mean(axis=-1)


def _best_patch_crop_box(
    score_map: np.ndarray,
    *,
    fraction: float = 0.25,
) -> tuple[int, int, int, int]:
    """Return y0, x0, crop_h, crop_w for the patch with the largest score sum."""
    h, w = score_map.shape[:2]
    crop_h = max(8, int(h * fraction))
    crop_w = max(8, int(w * fraction))
    best_sum = -1.0
    best_y0, best_x0 = (h - crop_h) // 2, (w - crop_w) // 2
    for y0 in range(0, h - crop_h + 1, max(1, crop_h // 4)):
        for x0 in range(0, w - crop_w + 1, max(1, crop_w // 4)):
            patch_sum = float(score_map[y0 : y0 + crop_h, x0 : x0 + crop_w].sum())
            if patch_sum > best_sum:
                best_sum = patch_sum
                best_y0, best_x0 = y0, x0
    return best_y0, best_x0, crop_h, crop_w


def _max_error_crop_box(
    a_hwc: np.ndarray,
    b_hwc: np.ndarray,
    *,
    fraction: float = 0.25,
) -> tuple[int, int, int, int]:
    """Return y0, x0, crop_h, crop_w for the patch with largest |a - b|."""
    return _best_patch_crop_box(_mean_abs_diff_map(a_hwc, b_hwc), fraction=fraction)


def _max_gain_crop_box(
    baseline_hwc: np.ndarray,
    candidate_hwc: np.ndarray,
    reference_hwc: np.ndarray,
    *,
    fraction: float = 0.25,
) -> tuple[int, int, int, int]:
    """Patch where candidate is most improved over baseline vs reference."""
    gain = _mean_abs_diff_map(baseline_hwc, reference_hwc) - _mean_abs_diff_map(
        candidate_hwc, reference_hwc
    )
    return _best_patch_crop_box(gain, fraction=fraction)


def _crop_zoom_pair(
    a_hwc: np.ndarray,
    b_hwc: np.ndarray,
    *,
    fraction: float = 0.25,
    scale: int = 4,
    target_width: int | None = None,
    crop_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Crop a region, nearest-neighbor upscale, return a | b."""
    if crop_box is None:
        y0, x0, crop_h, crop_w = _max_error_crop_box(a_hwc, b_hwc, fraction=fraction)
    else:
        y0, x0, crop_h, crop_w = crop_box
    if target_width is not None:
        scale = max(1, min(scale, target_width // max(1, 2 * crop_w)))
    crop_a = a_hwc[y0 : y0 + crop_h, x0 : x0 + crop_w]
    crop_b = b_hwc[y0 : y0 + crop_h, x0 : x0 + crop_w]
    crop_a = np.repeat(np.repeat(crop_a, scale, axis=0), scale, axis=1)
    crop_b = np.repeat(np.repeat(crop_b, scale, axis=0), scale, axis=1)
    return np.concatenate([crop_a, crop_b], axis=1)


def _paste_zoom_in_column(
    zoom: np.ndarray,
    *,
    col_w: int,
    row_h: int,
) -> np.ndarray:
    """Center a zoom tile inside a fixed-size column slot."""
    zoom_h, zoom_w = zoom.shape[:2]
    slot = np.full((row_h, col_w, 3), 32, dtype=np.uint8)
    y0 = max(0, (row_h - zoom_h) // 2)
    x0 = max(0, (col_w - zoom_w) // 2)
    paste_h = min(zoom_h, row_h - y0)
    paste_w = min(zoom_w, col_w - x0)
    slot[y0 : y0 + paste_h, x0 : x0 + paste_w] = zoom[:paste_h, :paste_w]
    return slot


def make_sr_panel(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    *,
    error_gain: float = 8.0,
    zoom_fraction: float = 0.25,
    zoom_scale: int = 4,
    include_detail: bool = True,
) -> np.ndarray:
    """Build LR | SR | HR panel, optionally with error maps and zoomed detail."""
    hr_hw = hr.shape[-2:]
    lr_up = F.interpolate(
        lr.unsqueeze(0),
        size=hr_hw,
        mode="bicubic",
        align_corners=False,
    )[0]
    lr_up_np = chw_tensor_to_uint8_hwc(lr_up)
    sr_np = chw_tensor_to_uint8_hwc(sr)
    hr_np = chw_tensor_to_uint8_hwc(hr)

    top = np.concatenate([lr_up_np, sr_np, hr_np], axis=1)
    if not include_detail:
        return top

    err_bicubic = _abs_diff_heatmap(lr_up_np, hr_np, gain=error_gain)
    err_sr = _abs_diff_heatmap(sr_np, hr_np, gain=error_gain)

    col_w = lr_up_np.shape[1]
    row_h = err_bicubic.shape[0]
    box_bicubic_err = _max_error_crop_box(lr_up_np, hr_np, fraction=zoom_fraction)
    box_sr_err = _max_error_crop_box(sr_np, hr_np, fraction=zoom_fraction)
    box_gain = _max_gain_crop_box(lr_up_np, sr_np, hr_np, fraction=zoom_fraction)

    zoom_sr_hr = _paste_zoom_in_column(
        _crop_zoom_pair(
            sr_np,
            hr_np,
            fraction=zoom_fraction,
            scale=zoom_scale,
            target_width=col_w,
            crop_box=box_sr_err,
        ),
        col_w=col_w,
        row_h=row_h,
    )
    mid = np.concatenate([err_bicubic, err_sr, zoom_sr_hr], axis=1)

    compare_row = np.concatenate(
        [
            _paste_zoom_in_column(
                _crop_zoom_pair(
                    lr_up_np,
                    sr_np,
                    fraction=zoom_fraction,
                    scale=zoom_scale,
                    target_width=col_w,
                    crop_box=box_bicubic_err,
                ),
                col_w=col_w,
                row_h=row_h,
            ),
            _paste_zoom_in_column(
                _crop_zoom_pair(
                    lr_up_np,
                    sr_np,
                    fraction=zoom_fraction,
                    scale=zoom_scale,
                    target_width=col_w,
                    crop_box=box_sr_err,
                ),
                col_w=col_w,
                row_h=row_h,
            ),
            _paste_zoom_in_column(
                _crop_zoom_pair(
                    lr_up_np,
                    sr_np,
                    fraction=zoom_fraction,
                    scale=zoom_scale,
                    target_width=col_w,
                    crop_box=box_gain,
                ),
                col_w=col_w,
                row_h=row_h,
            ),
        ],
        axis=1,
    )

    return np.concatenate([top, mid, compare_row], axis=0)


@torch.no_grad()
def collect_sr_validation_panels(
    model: nn.Module,
    val_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    num_samples: int = 4,
    error_gain: float = 8.0,
    include_detail: bool = True,
) -> list[np.ndarray]:
    """Run inference on validation batches and build comparison panels."""
    if num_samples <= 0:
        return []

    unwrap = getattr(model, "module", model)
    was_training = unwrap.training
    unwrap.eval()

    panels: list[np.ndarray] = []
    for lr, hr in val_loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        sr = torch.clamp(unwrap(lr), 0.0, 255.0)
        for i in range(lr.shape[0]):
            panels.append(
                make_sr_panel(
                    lr[i],
                    sr[i],
                    hr[i],
                    error_gain=error_gain,
                    include_detail=include_detail,
                )
            )
            if len(panels) >= num_samples:
                break
        if len(panels) >= num_samples:
            break

    if was_training:
        unwrap.train()
    return panels


def log_sr_panels(
    panels: list[np.ndarray],
    *,
    step: int,
    key_prefix: str = "val/sample",
    max_size: int = 768,
    include_detail: bool = True,
) -> None:
    """Upload SR comparison panels to SwanLab."""
    if not _active or not panels:
        return

    if include_detail:
        caption = (
            "LR(bicubic↑) | SR | HR\n"
            "err(bicubic) | err(SR) | zoom(SR|HR)@max-error\n"
            "zoom(bicubic|SR)@worst-bicubic | zoom(bicubic|SR)@worst-sr | zoom(bicubic|SR)@max-gain"
        )
    else:
        caption = "LR(bicubic↑) | SR | HR"

    payload: dict[str, Any] = {}
    for i, panel in enumerate(panels):
        payload[f"{key_prefix}_{i}"] = swanlab.Image(
            panel,
            caption=caption,
            size=max_size,
        )
    swanlab.log(payload, step=step)


def log_validation_sr_images(
    model: nn.Module,
    val_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    step: int,
    num_samples: int = 4,
    max_size: int = 768,
    error_gain: float = 8.0,
    include_detail: bool = True,
) -> None:
    """Collect and upload validation SR visualizations to SwanLab."""
    panels = collect_sr_validation_panels(
        model,
        val_loader,
        device,
        num_samples=num_samples,
        error_gain=error_gain,
        include_detail=include_detail,
    )
    log_sr_panels(
        panels,
        step=step,
        max_size=max_size,
        include_detail=include_detail,
    )
