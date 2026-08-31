"""SwanLab experiment logging helpers for DDP training."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import swanlab
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rknn_super_resolution.data.yuv_utils import colorspace_roundtrip_rgb
from rknn_super_resolution.models import SRInput, forward_sr, split_sr_input
from rknn_super_resolution.utils.run_logger import logger
from rknn_super_resolution.utils.sr_metrics import iter_val_batches, ssim_rgb

_active = False
_run_id: str | None = None
SWANLAB_RUN_FILE = "swanlab_run.json"
_LOCAL_RUN_ID_RE = re.compile(r"^[a-z0-9]{8}$")


def _looks_like_local_run_suffix(run_id: str) -> bool:
    """Return True for short local logdir suffixes (not cloud API run ids)."""
    return bool(_LOCAL_RUN_ID_RE.fullmatch(run_id))


def _swanlab_workspace() -> str | None:
    try:
        proc = subprocess.run(
            ["swanlab", "verify"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"as (\S+)", proc.stdout)
    return match.group(1) if match else None


def lookup_swanlab_cloud_run_id(
    project: str,
    experiment_name: str | None = None,
) -> str | None:
    """Resolve the cloud API run id for an experiment name via ``swanlab api``."""
    workspace = _swanlab_workspace()
    if not workspace:
        return None
    project_path = project if "/" in project else f"{workspace}/{project}"
    try:
        proc = subprocess.run(
            ["swanlab", "api", "run", "list", project_path],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    runs = payload.get("data", {}).get("list", [])
    if experiment_name:
        runs = [run for run in runs if run.get("name") == experiment_name]
    if not runs:
        return None
    for state in ("RUNNING", "FINISHED", "CRASHED"):
        for run in runs:
            if run.get("state") == state and run.get("run_id"):
                return str(run["run_id"])
    run_id = runs[0].get("run_id")
    return str(run_id) if run_id else None


def _normalize_cloud_run_id(
    run_id: str | None,
    *,
    project: str,
    experiment_name: str | None,
) -> str | None:
    if run_id and not _looks_like_local_run_suffix(run_id):
        return run_id
    return lookup_swanlab_cloud_run_id(project, experiment_name) or run_id


def get_swanlab_run_id() -> str | None:
    """Return the active SwanLab run id after ``setup_swanlab``."""
    return _run_id


def _swanlab_run_record_path(save_dir: Path) -> Path:
    return save_dir / SWANLAB_RUN_FILE


def save_swanlab_run_record(
    save_dir: Path,
    *,
    run_id: str,
    project: str,
    experiment_name: str | None,
) -> None:
    """Persist SwanLab run id for future resume."""
    record = {
        "run_id": run_id,
        "project": project,
        "experiment_name": experiment_name,
    }
    path = _swanlab_run_record_path(save_dir)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def load_swanlab_run_record(save_dir: Path) -> dict[str, Any] | None:
    path = _swanlab_run_record_path(save_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and data.get("run_id") else None


def find_swanlab_run_id(
    save_dir: Path,
    experiment_name: str | None = None,
) -> str | None:
    """Infer the original SwanLab run id from local swanlog metadata."""
    record = load_swanlab_run_record(save_dir)
    if record is not None:
        return str(record["run_id"])

    swanlog = save_dir / "swanlog"
    if not swanlog.is_dir():
        return None

    matches: list[tuple[str, str]] = []
    for run_dir in swanlog.glob("run-*"):
        run_id = run_dir.name.rsplit("-", 1)[-1]
        meta_path = run_dir / "files" / "swanlab-metadata.json"
        if experiment_name and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            command = str(meta.get("runtime", {}).get("command", ""))
            if experiment_name not in command:
                continue
        stamp = run_dir.name.removeprefix("run-").rsplit("-", 1)[0]
        matches.append((stamp, run_id))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def resolve_swanlab_run_id(
    *,
    save_dir: Path,
    project: str,
    experiment_name: str | None,
    resume_checkpoint: str | Path | None = None,
    explicit_run_id: str | None = None,
) -> str | None:
    """Resolve SwanLab cloud run id for resume (CLI > checkpoint > local record > API)."""
    candidate = explicit_run_id or os.environ.get("SWANLAB_RUN_ID")
    if not candidate and resume_checkpoint:
        try:
            raw = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            if isinstance(raw, dict) and raw.get("swanlab_run_id"):
                candidate = str(raw["swanlab_run_id"])
        except Exception:
            pass
    if not candidate:
        record = load_swanlab_run_record(save_dir)
        if record is not None:
            candidate = str(record["run_id"])
    if not candidate:
        candidate = find_swanlab_run_id(save_dir, experiment_name)
    if not candidate and resume_checkpoint:
        return None
    return _normalize_cloud_run_id(
        candidate,
        project=project,
        experiment_name=experiment_name,
    )


def _jsonable_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable_config_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_config_value(item) for key, item in value.items()}
    return value


def build_swanlab_run_config(
    args: Any,
    *,
    world_size: int,
    vmaf_enc_size: tuple[int, int] | None = (360, 640),
) -> dict[str, Any]:
    """Resolved training config for SwanLab (no argparse ``None`` → ``{}`` holes)."""
    from rknn_super_resolution.utils.vmaf_metric import resolved_vmaf_model_id

    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if value is None:
            continue
        config[key] = _jsonable_config_value(value)
    model = str(getattr(args, "vmaf_model", "1080p"))
    config["vmaf_family"] = "v1"
    config["vmaf_model_alias"] = model
    config["vmaf_model_id"] = resolved_vmaf_model_id(model)
    if vmaf_enc_size is not None:
        config["vmaf_enc_size"] = [int(vmaf_enc_size[0]), int(vmaf_enc_size[1])]
        config["vmaf_enc_size_note"] = "CAMBI encode size only; VMAF scores full-resolution frames"
    config["world_size"] = int(world_size)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        config["cuda_visible_devices"] = visible
    return config


def setup_swanlab(
    *,
    rank: int,
    save_dir: Path,
    project: str,
    experiment_name: str | None = None,
    config: dict[str, Any] | None = None,
    disabled: bool = False,
    resume_training: bool = False,
    run_id: str | None = None,
    resume: Literal["must", "allow", "never"] | bool = "must",
    mode: Literal["online", "offline", "local", "disabled"] | None = None,
) -> None:
    """Initialize SwanLab on rank 0."""
    global _active, _run_id
    _active = False
    _run_id = None
    if rank != 0 or disabled:
        return

    init_kwargs: dict[str, Any] = {
        "project": project,
        "experiment_name": experiment_name,
        "config": config or {},
        "log_dir": str(save_dir / "swanlog"),
    }
    if mode is not None:
        init_kwargs["mode"] = mode
    resolved_run_id = _normalize_cloud_run_id(
        run_id,
        project=project,
        experiment_name=experiment_name,
    )
    if resume_training:
        if not resolved_run_id:
            resolved_run_id = find_swanlab_run_id(save_dir, experiment_name)
            resolved_run_id = _normalize_cloud_run_id(
                resolved_run_id,
                project=project,
                experiment_name=experiment_name,
            )
        if resolved_run_id:
            init_kwargs["id"] = resolved_run_id
            init_kwargs["resume"] = resume
        else:
            logger.warning(
                "training resume requested but no SwanLab run id found under {}; "
                "starting a new SwanLab experiment.",
                save_dir,
            )

    try:
        run = swanlab.init(**init_kwargs)
        _run_id = run.id
        _active = True
        save_swanlab_run_record(
            save_dir,
            run_id=_run_id,
            project=project,
            experiment_name=experiment_name,
        )
        if resume_training and resolved_run_id:
            logger.info(
                "swanlab resumed: project={} experiment={} id={}",
                project,
                experiment_name,
                _run_id,
            )
        else:
            logger.info(
                "swanlab initialized: project={} experiment={} id={}",
                project,
                experiment_name,
                _run_id,
            )
    except Exception as exc:
        logger.warning("swanlab init failed ({}); continuing without SwanLab logging.", exc)
        _active = False
        _run_id = None


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


def log_observation_panels(
    panels: dict[str, np.ndarray],
    *,
    step: int,
    key_prefix: str = "model_observatory",
    max_size: int = 1600,
) -> None:
    """Log already-rendered model observation panels on the active rank-0 run."""
    if not _active or not panels:
        return
    payload = {
        f"{key_prefix}/{name}": swanlab.Image(panel, size=max_size)
        for name, panel in panels.items()
    }
    log_metrics(payload, step=step)


def finish_swanlab() -> None:
    """Finish the SwanLab run on rank 0."""
    global _active, _run_id
    if not _active:
        return
    swanlab.finish()
    _active = False
    _run_id = None


def chw_tensor_to_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW float tensor in [0, 255] to HWC uint8 numpy."""
    x = tensor.detach().float().clamp(0.0, 255.0).cpu()
    if x.dim() == 4:
        x = x[0]
    return x.permute(1, 2, 0).numpy().astype(np.uint8)


def colorspace_to_rgb(tensor: torch.Tensor, colorspace: str) -> torch.Tensor:
    """Convert a CHW canvas tensor to RGB for visualization."""
    if colorspace == "rgb":
        return tensor
    from rknn_super_resolution.data.yuv_utils import yuv444_to_rgb

    return yuv444_to_rgb(tensor)


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


def _sr_effect_heatmap(
    baseline_hwc: np.ndarray,
    sr_hwc: np.ndarray,
    hr_hwc: np.ndarray,
    *,
    gain: float = 8.0,
) -> np.ndarray:
    """Show SR's incremental effect over the pre-SR baseline.

    Green means SR reduced mean-channel absolute error; red means it increased
    error. Black means SR had little effect on the baseline error.
    """
    baseline_error = _mean_abs_diff_map(baseline_hwc, hr_hwc)
    sr_error = _mean_abs_diff_map(sr_hwc, hr_hwc)
    improvement = (baseline_error - sr_error) * gain
    effect = np.zeros((*improvement.shape, 3), dtype=np.float32)
    effect[..., 0] = np.clip(-improvement, 0.0, 255.0)
    effect[..., 1] = np.clip(improvement, 0.0, 255.0)
    return effect.astype(np.uint8)


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


def _label_tile(tile: np.ndarray, label: str) -> np.ndarray:
    """Burn a readable label into a visualization tile."""
    if min(tile.shape[:2]) < 256:
        return tile

    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(tile.copy())
    draw = ImageDraw.Draw(image)
    # SwanLab normally scales the full three-column panel to 768 px wide.
    # A relatively large source font remains legible in that thumbnail.
    font_size = max(14, min(tile.shape[:2]) // 12)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    pad = max(4, font_size // 5)
    box_w = right - left + 2 * pad
    box_h = bottom - top + 2 * pad
    draw.rectangle((0, 0, box_w, box_h), fill=(16, 16, 16))
    draw.text((pad - left, pad - top), label, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def upscale_lr_canvas(lr: torch.Tensor, hr_hw: tuple[int, int]) -> torch.Tensor:
    """Upscale MLVC LR to HR size for side-by-side visualization."""
    return F.interpolate(
        lr.unsqueeze(0),
        size=hr_hw,
        mode="nearest",
    )[0]


def bicubic_lr_canvas(lr: torch.Tensor, hr_hw: tuple[int, int]) -> torch.Tensor:
    """Build the same bicubic pre-SR baseline used by Phase-RLFN."""
    if lr.ndim not in (3, 4):
        raise ValueError(f"expected CHW or BCHW LR tensor, got shape {tuple(lr.shape)}")
    squeeze = lr.ndim == 3
    batch = lr.unsqueeze(0) if squeeze else lr
    output = F.interpolate(
        batch,
        size=hr_hw,
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 255.0)
    return output[0] if squeeze else output


def make_sr_panel(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    *,
    baseline: torch.Tensor | None = None,
    error_gain: float = 8.0,
    zoom_fraction: float = 0.25,
    zoom_scale: int = 4,
    include_detail: bool = True,
) -> np.ndarray:
    """Build an attribution panel separating upstream and SR-induced error."""
    hr_hw = hr.shape[-2:]
    if baseline is None:
        baseline = bicubic_lr_canvas(lr, hr_hw)
    baseline_np = chw_tensor_to_uint8_hwc(baseline)
    sr_np = chw_tensor_to_uint8_hwc(sr)
    hr_np = chw_tensor_to_uint8_hwc(hr)

    baseline_psnr = rgb_diff_stats(baseline_np, hr_np)["psnr"]
    sr_psnr = rgb_diff_stats(sr_np, hr_np)["psnr"]
    baseline_ssim = ssim_rgb(baseline_np, hr_np)
    sr_ssim = ssim_rgb(sr_np, hr_np)

    top = np.concatenate(
        [
            _label_tile(
                baseline_np,
                f"BICUBIC {baseline_psnr:.2f}dB SSIM {baseline_ssim:.4f}",
            ),
            _label_tile(
                sr_np,
                f"SR {sr_psnr:.2f}dB (+{sr_psnr - baseline_psnr:.2f}) SSIM {sr_ssim:.4f}",
            ),
            _label_tile(hr_np, "HR TARGET"),
        ],
        axis=1,
    )
    if not include_detail:
        return top

    err_baseline = _abs_diff_heatmap(baseline_np, hr_np, gain=error_gain)
    err_sr = _abs_diff_heatmap(sr_np, hr_np, gain=error_gain)
    sr_effect = _sr_effect_heatmap(
        baseline_np,
        sr_np,
        hr_np,
        gain=error_gain,
    )
    attribution = np.concatenate(
        [
            _label_tile(err_baseline, "ERROR BEFORE SR"),
            _label_tile(err_sr, "ERROR AFTER SR"),
            _label_tile(sr_effect, "SR EFFECT: GREEN + / RED -"),
        ],
        axis=1,
    )

    # Use one shared crop where SR changes the baseline error most. Comparing
    # different crops would make attribution ambiguous.
    baseline_error = _mean_abs_diff_map(baseline_np, hr_np)
    sr_error = _mean_abs_diff_map(sr_np, hr_np)
    crop_box = _best_patch_crop_box(
        np.abs(baseline_error - sr_error),
        fraction=zoom_fraction,
    )
    col_w = baseline_np.shape[1]
    row_h = baseline_np.shape[0]

    def _zoom_slot(a: np.ndarray, b: np.ndarray, label: str) -> np.ndarray:
        return _label_tile(
            _paste_zoom_in_column(
                _crop_zoom_pair(
                    a,
                    b,
                    fraction=zoom_fraction,
                    scale=zoom_scale,
                    target_width=col_w,
                    crop_box=crop_box,
                ),
                col_w=col_w,
                row_h=row_h,
            ),
            label,
        )

    zoom = np.concatenate(
        [
            _zoom_slot(baseline_np, hr_np, "ZOOM: PRE-SR | HR"),
            _zoom_slot(sr_np, hr_np, "ZOOM: SR | HR"),
            _zoom_slot(baseline_np, sr_np, "ZOOM: PRE-SR | SR"),
        ],
        axis=1,
    )
    return np.concatenate([top, attribution, zoom], axis=0)


def rgb_diff_stats(a_hwc: np.ndarray, b_hwc: np.ndarray) -> dict[str, float]:
    """Mean/max abs RGB diff and PSNR between two HWC uint8 images."""
    diff = np.abs(a_hwc.astype(np.float32) - b_hwc.astype(np.float32))
    mse = float(np.mean(diff**2))
    psnr = 100.0 if mse <= 0.0 else float(10.0 * np.log10(255.0 * 255.0 / mse))
    return {
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "psnr": psnr,
    }


def make_data_preview_panel(
    lr_rgb: torch.Tensor,
    hr_rgb: torch.Tensor,
    *,
    colorspace: str = "yuv",
    error_gain: float = 8.0,
    zoom_fraction: float = 0.25,
    zoom_scale: int = 4,
) -> np.ndarray:
    """Build an MLVC reconstruction preview with a colorspace roundtrip check.

    Top row: MLVC LR (NN x3) | HR target | HR after train colorspace roundtrip.
    Bottom row: MLVC error | colorspace error | max-error detail crop.
    """
    hr_hw = hr_rgb.shape[-2:]
    lr_up = upscale_lr_canvas(lr_rgb, hr_hw)
    lr_np = chw_tensor_to_uint8_hwc(lr_up)
    hr_np = chw_tensor_to_uint8_hwc(hr_rgb)
    hr_roundtrip = colorspace_roundtrip_rgb(hr_rgb, colorspace)
    hr_rt_np = chw_tensor_to_uint8_hwc(hr_roundtrip)

    top = np.concatenate([lr_np, hr_np, hr_rt_np], axis=1)
    err_lr = _abs_diff_heatmap(lr_np, hr_np, gain=error_gain)
    err_rt = _abs_diff_heatmap(hr_np, hr_rt_np, gain=error_gain)
    col_w = lr_np.shape[1]
    row_h = err_lr.shape[0]
    zoom_hr_rt = _paste_zoom_in_column(
        _crop_zoom_pair(
            hr_np,
            hr_rt_np,
            fraction=zoom_fraction,
            scale=zoom_scale,
            target_width=col_w,
            crop_box=_max_error_crop_box(hr_np, hr_rt_np, fraction=zoom_fraction),
        ),
        col_w=col_w,
        row_h=row_h,
    )
    bottom = np.concatenate([err_lr, err_rt, zoom_hr_rt], axis=1)
    return np.concatenate([top, bottom], axis=0)


def data_preview_caption(*, colorspace: str) -> str:
    roundtrip = "YUV↔RGB roundtrip" if colorspace == "yuv" else "RGB identity"
    return (
        "MLVC LR (NN x3) | HR target | HR after colorspace roundtrip\n"
        f"err(MLVC LR↑) | err({roundtrip}) | zoom(HR|roundtrip)@max-error"
    )


@dataclass(frozen=True)
class DataPreviewSample:
    index: int
    panel: np.ndarray
    stats: dict[str, float]


def collect_data_preview_samples(
    val_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    *,
    num_samples: int,
    colorspace: str = "yuv",
    error_gain: float = 8.0,
) -> list[DataPreviewSample]:
    """Build preview panels from frozen-MLVC validation batches."""
    if num_samples <= 0:
        return []

    samples: list[DataPreviewSample] = []
    for model_input, hr in val_loader:
        lr, _codec_feature = split_sr_input(model_input)
        for offset in range(lr.shape[0]):
            lr_rgb = colorspace_to_rgb(lr[offset, :3], colorspace)
            hr_rgb = colorspace_to_rgb(hr[offset], colorspace)
            panel = make_data_preview_panel(
                lr_rgb,
                hr_rgb,
                colorspace=colorspace,
                error_gain=error_gain,
            )
            lr_np = chw_tensor_to_uint8_hwc(upscale_lr_canvas(lr_rgb, hr_rgb.shape[-2:]))
            hr_np = chw_tensor_to_uint8_hwc(hr_rgb)
            lr_stats = rgb_diff_stats(lr_np, hr_np)
            samples.append(
                DataPreviewSample(
                    index=len(samples),
                    panel=panel,
                    stats={
                        "mlvc_lr_psnr": lr_stats["psnr"],
                        "mlvc_lr_mean_abs": lr_stats["mean_abs"],
                    },
                )
            )
            if len(samples) >= num_samples:
                return samples
    return samples


def _save_preview_png(panel: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(path)


def log_data_preview_samples(
    samples: list[DataPreviewSample],
    *,
    step: int = 0,
    max_size: int = 768,
    save_dir: Path | None = None,
    colorspace: str = "yuv",
) -> dict[str, float]:
    """Upload frozen-MLVC input previews and aggregate degradation metrics."""
    if not samples:
        return {}

    aggregate = {
        "data_preview/mlvc_lr_psnr": float(
            np.mean([sample.stats["mlvc_lr_psnr"] for sample in samples])
        ),
        "data_preview/mlvc_lr_mean_abs": float(
            np.mean([sample.stats["mlvc_lr_mean_abs"] for sample in samples])
        ),
    }

    if save_dir is not None:
        preview_dir = save_dir / "data_preview"
        for sample in samples:
            _save_preview_png(sample.panel, preview_dir / f"sample_{sample.index:03d}.png")

    if _active:
        payload: dict[str, Any] = {}
        for sample in samples:
            payload[f"data_preview/sample_{sample.index:03d}"] = swanlab.Image(
                sample.panel,
                caption=data_preview_caption(colorspace=colorspace),
                size=max_size,
            )
        swanlab.log(payload, step=step)
        log_metrics(aggregate, step=step)

    return aggregate


def run_training_data_preview(
    val_loader: DataLoader,
    *,
    colorspace: str,
    num_samples: int = 4,
    max_size: int = 768,
    error_gain: float = 8.0,
    save_dir: Path | None = None,
    step: int = 0,
) -> dict[str, float]:
    """Collect and log frozen-MLVC reconstructions before training."""
    samples = collect_data_preview_samples(
        iter_val_batches(val_loader),
        num_samples=num_samples,
        colorspace=colorspace,
        error_gain=error_gain,
    )
    metrics = log_data_preview_samples(
        samples,
        step=step,
        max_size=max_size,
        save_dir=save_dir,
        colorspace=colorspace,
    )
    if metrics:
        logger.info(
            "data preview ({} samples): MLVC LR PSNR={:.2f} dB mean_abs={:.3f}",
            len(samples),
            metrics["data_preview/mlvc_lr_psnr"],
            metrics["data_preview/mlvc_lr_mean_abs"],
        )
    return metrics


@torch.no_grad()
def collect_sr_validation_panels(
    model: nn.Module,
    val_loader: Iterator[tuple[SRInput, torch.Tensor]],
    device: torch.device,
    *,
    num_samples: int = 4,
    error_gain: float = 8.0,
    include_detail: bool = True,
    colorspace: str = "rgb",
) -> list[np.ndarray]:
    """Collect Phase-RLFN panels from processed MLVC validation batches."""
    if num_samples <= 0:
        return []

    unwrap = getattr(model, "module", model)
    was_training = unwrap.training
    unwrap.eval()

    panels: list[np.ndarray] = []
    for model_input, hr in val_loader:
        lr, codec_feature = split_sr_input(model_input)
        lr = lr.to(device, non_blocking=True)
        if codec_feature is not None:
            model_input = (lr, codec_feature.to(device, non_blocking=True))
        else:
            model_input = lr
        hr = hr.to(device, non_blocking=True)
        bicubic_base = getattr(unwrap, "bicubic_base", None)
        baseline = (
            bicubic_base(lr) if callable(bicubic_base) else bicubic_lr_canvas(lr, hr.shape[-2:])
        )
        sr = torch.clamp(forward_sr(unwrap, model_input), 0.0, 255.0)
        for i in range(lr.shape[0]):
            lr_rgb = colorspace_to_rgb(lr[i, :3], colorspace)
            baseline_rgb = colorspace_to_rgb(baseline[i], colorspace)
            sr_rgb = colorspace_to_rgb(sr[i], colorspace)
            hr_rgb = colorspace_to_rgb(hr[i], colorspace)
            panels.append(
                make_sr_panel(
                    lr_rgb,
                    sr_rgb,
                    hr_rgb,
                    baseline=baseline_rgb,
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


def save_sr_panels(
    panels: list[np.ndarray],
    save_dir: Path,
    *,
    subdir: str = "sr_preview",
) -> Path:
    """Write pre-SR/SR/HR attribution panels under ``save_dir/subdir``."""
    preview_dir = Path(save_dir) / subdir
    for index, panel in enumerate(panels):
        _save_preview_png(panel, preview_dir / f"sample_{index:03d}.png")
    return preview_dir


def log_sr_panels(
    panels: list[np.ndarray],
    *,
    step: int,
    key_prefix: str = "val/sample",
    max_size: int = 768,
    include_detail: bool = True,
    save_dir: Path | None = None,
    save_subdir: str = "sr_preview",
    colorspace: str | None = None,
) -> Path | None:
    """Upload generic SR panels to SwanLab."""
    preview_dir = (
        save_sr_panels(panels, save_dir, subdir=save_subdir)
        if save_dir is not None and panels
        else None
    )
    if not _active or not panels:
        return preview_dir

    detail_note = (
        "\nerror before SR | error after SR | SR effect (green=better, red=worse)"
        "\nzoom(pre-SR|HR) | zoom(SR|HR) | zoom(pre-SR|SR) @ max SR impact"
        if include_detail
        else ""
    )
    space = f" [{colorspace}]" if colorspace else ""
    caption = f"pre-SR MLVC+bicubic | Phase-RLFN | HR{space}{detail_note}"

    payload: dict[str, Any] = {}
    for i, panel in enumerate(panels):
        payload[f"{key_prefix}_{i}"] = swanlab.Image(
            panel,
            caption=caption,
            size=max_size,
        )
    swanlab.log(payload, step=step)
    return preview_dir


def log_validation_sr_images(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    step: int,
    num_samples: int = 4,
    max_size: int = 768,
    error_gain: float = 8.0,
    include_detail: bool = True,
    colorspace: str = "rgb",
    save_dir: Path | None = None,
    key_prefix: str = "val",
    save_subdir: str = "sr_preview",
) -> None:
    """Collect and upload validation visualizations to SwanLab."""
    panels = collect_sr_validation_panels(
        model,
        iter_val_batches(val_loader),
        device,
        num_samples=num_samples,
        error_gain=error_gain,
        include_detail=include_detail,
        colorspace=colorspace,
    )
    log_sr_panels(
        panels,
        step=step,
        key_prefix=f"{key_prefix}/sample",
        max_size=max_size,
        include_detail=include_detail,
        save_dir=save_dir,
        save_subdir=save_subdir,
        colorspace=colorspace,
    )


def run_final_sr_preview(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    colorspace: str,
    num_samples: int = 8,
    max_size: int = 768,
    save_dir: Path,
    step: int = 0,
    checkpoint: Path | None = None,
) -> Path | None:
    """After training, write LR|SR|HR panels under ``{save_dir}/sr_preview``.

    Prefers ``checkpoint`` (usually ``best.pth``) when present so the panels
    reflect the best validation weights rather than the last step.
    """
    from rknn_super_resolution.utils.train_framework import load_training_module_state_dict

    if checkpoint is not None and checkpoint.is_file():
        raw = torch.load(checkpoint, map_location=device, weights_only=False)
        if not isinstance(raw, dict) or not {"graph_format", "state_dict"}.issubset(raw):
            raise TypeError(f"Expected a versioned model checkpoint: {checkpoint}")
        load_training_module_state_dict(model, raw["state_dict"])
        step = int(raw.get("step", step))
        logger.info("final sr preview loading {}", checkpoint)

    panels = collect_sr_validation_panels(
        model,
        iter_val_batches(val_loader),
        device,
        num_samples=num_samples,
        colorspace=colorspace,
    )
    preview_dir = log_sr_panels(
        panels,
        step=step,
        key_prefix="sr_preview/sample",
        max_size=max_size,
        save_dir=save_dir,
        save_subdir="sr_preview",
        colorspace=colorspace,
    )
    if preview_dir is not None:
        logger.info(
            "final sr preview ({} samples @ step {}): {}",
            len(panels),
            step,
            preview_dir,
        )
    return preview_dir
