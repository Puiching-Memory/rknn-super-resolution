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

from rk3588_mobile_sr.data.val_loader import (
    FixedValDataset,
    iter_val_batches,
    select_val_vis_indices,
    val_sample_meta,
)
from rk3588_mobile_sr.data.yuv_utils import colorspace_roundtrip_rgb
from rk3588_mobile_sr.data.types import ValSampleMeta
from rk3588_mobile_sr.utils.run_logger import logger

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
    from rk3588_mobile_sr.data.yuv_utils import yuv444_to_rgb

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


def upscale_lr_canvas(lr: torch.Tensor, hr_hw: tuple[int, int]) -> torch.Tensor:
    """Upscale codec LR canvas to HR size for side-by-side visualization."""
    return F.interpolate(
        lr.unsqueeze(0),
        size=hr_hw,
        mode="nearest",
    )[0]


def make_codec_canvas_panel(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    *,
    error_gain: float = 8.0,
    zoom_fraction: float = 0.25,
    zoom_scale: int = 4,
    include_detail: bool = True,
) -> np.ndarray:
    """Build codec-canvas validation panel: LR(NN x3) | SR | HR (+ optional error row)."""
    hr_hw = hr.shape[-2:]
    lr_up = upscale_lr_canvas(lr, hr_hw)
    lr_np = chw_tensor_to_uint8_hwc(lr_up)
    sr_np = chw_tensor_to_uint8_hwc(sr)
    hr_np = chw_tensor_to_uint8_hwc(hr)

    top = np.concatenate([lr_np, sr_np, hr_np], axis=1)
    if not include_detail:
        return top

    err_lr = _abs_diff_heatmap(lr_np, hr_np, gain=error_gain)
    err_sr = _abs_diff_heatmap(sr_np, hr_np, gain=error_gain)
    col_w = lr_np.shape[1]
    row_h = err_lr.shape[0]
    zoom_sr_hr = _paste_zoom_in_column(
        _crop_zoom_pair(
            sr_np,
            hr_np,
            fraction=zoom_fraction,
            scale=zoom_scale,
            target_width=col_w,
            crop_box=_max_error_crop_box(sr_np, hr_np, fraction=zoom_fraction),
        ),
        col_w=col_w,
        row_h=row_h,
    )
    bottom = np.concatenate([err_lr, err_sr, zoom_sr_hr], axis=1)
    return np.concatenate([top, bottom], axis=0)


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
    """Build pre-training codec downsample preview with colorspace roundtrip check.

    Top row: codec LR (NN x3) | HR decode | HR after train colorspace roundtrip
    Bottom row: err(codec LR↑) | err(colorspace roundtrip) | zoom(HR|roundtrip)@max-error
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
        "codec LR (NN x3) | HR decode | HR after colorspace roundtrip\n"
        f"err(codec LR↑) | err({roundtrip}) | zoom(HR|roundtrip)@max-error"
    )


def make_sr_panel(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    **kwargs: Any,
) -> np.ndarray:
    """Backward-compatible alias for codec canvas panels."""
    return make_codec_canvas_panel(lr, sr, hr, **kwargs)


@dataclass(frozen=True)
class DataPreviewSample:
    meta: ValSampleMeta
    panel: np.ndarray
    stats: dict[str, float]
    lr_video: Path | None = None
    hr_video: Path | None = None


def collect_data_preview_samples(
    dataset: FixedValDataset,
    *,
    indices: list[int],
    colorspace: str = "yuv",
    error_gain: float = 8.0,
) -> list[DataPreviewSample]:
    """Build codec downsample preview panels before training (no model inference)."""
    if not indices:
        return []

    lr_h, lr_w = dataset.settings.lr_size
    hr_h, hr_w = dataset.settings.hr_size
    samples: list[DataPreviewSample] = []
    for index in indices:
        lr_rgb, hr_rgb = dataset.decode_rgb_pair(index)
        panel = make_data_preview_panel(
            lr_rgb,
            hr_rgb,
            colorspace=colorspace,
            error_gain=error_gain,
        )
        lr_up = upscale_lr_canvas(lr_rgb, hr_rgb.shape[-2:])
        lr_np = chw_tensor_to_uint8_hwc(lr_up)
        hr_np = chw_tensor_to_uint8_hwc(hr_rgb)
        hr_rt_np = chw_tensor_to_uint8_hwc(colorspace_roundtrip_rgb(hr_rgb, colorspace))
        codec_stats = rgb_diff_stats(lr_np, hr_np)
        roundtrip_stats = rgb_diff_stats(hr_np, hr_rt_np)
        meta = val_sample_meta(
            dataset.specs[index],
            lr_size=(lr_h, lr_w),
            hr_size=(hr_h, hr_w),
        )
        clip_paths = dataset.codec_clip_paths_for_index(index)
        lr_video = clip_paths[0] if clip_paths else None
        hr_video = clip_paths[1] if clip_paths else None
        samples.append(
            DataPreviewSample(
                meta=meta,
                panel=panel,
                stats={
                    "codec_lr_psnr": codec_stats["psnr"],
                    "codec_lr_mean_abs": codec_stats["mean_abs"],
                    "colorspace_roundtrip_psnr": roundtrip_stats["psnr"],
                    "colorspace_roundtrip_mean_abs": roundtrip_stats["mean_abs"],
                    "colorspace_roundtrip_max_abs": roundtrip_stats["max_abs"],
                },
                lr_video=lr_video,
                hr_video=hr_video,
            )
        )
    return samples


def _save_preview_png(panel: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).save(path)


def _append_codec_clip_videos(
    payload: dict[str, Any],
    *,
    key_prefix: str,
    slug: str,
    lr_video: Path | None,
    hr_video: Path | None,
    caption: str,
) -> None:
    """Attach LR/HR codec MP4 paths to a SwanLab log payload."""
    if lr_video is not None and lr_video.is_file():
        payload[f"{key_prefix}/{slug}/lr_clip"] = swanlab.Video(
            str(lr_video),
            caption=f"{caption}\nLR codec clip",
        )
    if hr_video is not None and hr_video.is_file():
        payload[f"{key_prefix}/{slug}/hr_clip"] = swanlab.Video(
            str(hr_video),
            caption=f"{caption}\nHR mezzanine clip",
        )


def log_data_preview_samples(
    samples: list[DataPreviewSample],
    *,
    step: int = 0,
    max_size: int = 768,
    colorspace: str = "yuv",
    save_dir: Path | None = None,
) -> dict[str, float]:
    """Upload data preview panels and aggregate colorspace diagnostic metrics."""
    if not samples:
        return {}

    aggregate = {
        "data_preview/codec_lr_psnr": float(
            np.mean([sample.stats["codec_lr_psnr"] for sample in samples])
        ),
        "data_preview/codec_lr_mean_abs": float(
            np.mean([sample.stats["codec_lr_mean_abs"] for sample in samples])
        ),
        "data_preview/colorspace_roundtrip_psnr": float(
            np.mean([sample.stats["colorspace_roundtrip_psnr"] for sample in samples])
        ),
        "data_preview/colorspace_roundtrip_mean_abs": float(
            np.mean([sample.stats["colorspace_roundtrip_mean_abs"] for sample in samples])
        ),
        "data_preview/colorspace_roundtrip_max_abs": float(
            np.max([sample.stats["colorspace_roundtrip_max_abs"] for sample in samples])
        ),
    }

    if save_dir is not None:
        preview_dir = save_dir / "data_preview"
        for sample in samples:
            _save_preview_png(sample.panel, preview_dir / f"{sample.meta.slug}.png")

    if _active:
        payload: dict[str, Any] = {}
        for sample in samples:
            header = sample.meta.data_preview_header(colorspace=colorspace)
            caption = (
                header
                + "\n"
                + data_preview_caption(colorspace=colorspace).split("\n", 1)[1]
            )
            payload[f"data_preview/{sample.meta.slug}"] = swanlab.Image(
                sample.panel,
                caption=caption,
                size=max_size,
            )
            _append_codec_clip_videos(
                payload,
                key_prefix="data_preview",
                slug=sample.meta.slug,
                lr_video=sample.lr_video,
                hr_video=sample.hr_video,
                caption=header,
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
    """Collect and log codec downsample previews from the fixed val set."""
    dataset = val_loader.dataset
    if not isinstance(dataset, FixedValDataset):
        logger.warning("data preview skipped: val loader is not FixedValDataset")
        return {}

    indices = select_val_vis_indices(dataset.specs, num_samples)
    samples = collect_data_preview_samples(
        dataset,
        indices=indices,
        colorspace=colorspace,
        error_gain=error_gain,
    )
    metrics = log_data_preview_samples(
        samples,
        step=step,
        max_size=max_size,
        colorspace=colorspace,
        save_dir=save_dir,
    )
    if metrics:
        logger.info(
            "data preview ({} samples): codec LR PSNR={:.2f} dB | "
            "colorspace roundtrip PSNR={:.2f} dB mean_abs={:.3f} max_abs={:.1f}",
            len(samples),
            metrics["data_preview/codec_lr_psnr"],
            metrics["data_preview/colorspace_roundtrip_psnr"],
            metrics["data_preview/colorspace_roundtrip_mean_abs"],
            metrics["data_preview/colorspace_roundtrip_max_abs"],
        )
    return metrics


@dataclass(frozen=True)
class ValVisSample:
    meta: ValSampleMeta
    panel: np.ndarray
    lr_video: Path | None = None
    hr_video: Path | None = None


@torch.no_grad()
def collect_codec_val_vis_samples(
    model: nn.Module,
    dataset: FixedValDataset,
    device: torch.device,
    *,
    indices: list[int],
    colorspace: str = "yuv",
    error_gain: float = 8.0,
    include_detail: bool = True,
) -> list[ValVisSample]:
    """Run inference on selected fixed-val indices and build labeled panels."""
    if not indices:
        return []

    unwrap = getattr(model, "module", model)
    was_training = unwrap.training
    unwrap.eval()

    lr_h, lr_w = dataset.settings.lr_size
    hr_h, hr_w = dataset.settings.hr_size
    samples: list[ValVisSample] = []
    for index in indices:
        lr, hr = dataset[index]
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        sr = torch.clamp(unwrap(lr.unsqueeze(0)), 0.0, 255.0)[0]
        lr_rgb = colorspace_to_rgb(lr, colorspace)
        sr_rgb = colorspace_to_rgb(sr, colorspace)
        hr_rgb = colorspace_to_rgb(hr, colorspace)
        meta = val_sample_meta(
            dataset.specs[index],
            lr_size=(lr_h, lr_w),
            hr_size=(hr_h, hr_w),
        )
        panel = make_codec_canvas_panel(
            lr_rgb,
            sr_rgb,
            hr_rgb,
            error_gain=error_gain,
            include_detail=include_detail,
        )
        clip_paths = dataset.codec_clip_paths_for_index(index)
        lr_video = clip_paths[0] if clip_paths else None
        hr_video = clip_paths[1] if clip_paths else None
        samples.append(
            ValVisSample(
                meta=meta,
                panel=panel,
                lr_video=lr_video,
                hr_video=hr_video,
            )
        )

    if was_training:
        unwrap.train()
    return samples


@torch.no_grad()
def collect_sr_validation_panels(
    model: nn.Module,
    val_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    num_samples: int = 4,
    error_gain: float = 8.0,
    include_detail: bool = True,
    colorspace: str = "rgb",
) -> list[np.ndarray]:
    """Legacy iterator-based panel collection for simple tensor loaders."""
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
            lr_rgb = colorspace_to_rgb(lr[i], colorspace)
            sr_rgb = colorspace_to_rgb(sr[i], colorspace)
            hr_rgb = colorspace_to_rgb(hr[i], colorspace)
            panels.append(
                make_codec_canvas_panel(
                    lr_rgb,
                    sr_rgb,
                    hr_rgb,
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


def log_val_vis_samples(
    samples: list[ValVisSample],
    *,
    step: int,
    max_size: int = 768,
    include_detail: bool = True,
) -> None:
    """Upload codec-canvas validation panels to SwanLab with per-sample keys."""
    if not _active or not samples:
        return

    detail_note = (
        "\nerr(codec LR↑) | err(SR) | zoom(SR|HR)@max-error"
        if include_detail
        else ""
    )
    payload: dict[str, Any] = {}
    for sample in samples:
        caption = sample.meta.caption() + detail_note
        payload[f"val/{sample.meta.slug}"] = swanlab.Image(
            sample.panel,
            caption=caption,
            size=max_size,
        )
        _append_codec_clip_videos(
            payload,
            key_prefix="val",
            slug=sample.meta.slug,
            lr_video=sample.lr_video,
            hr_video=sample.hr_video,
            caption=sample.meta.caption(),
        )
    swanlab.log(payload, step=step)


def log_sr_panels(
    panels: list[np.ndarray],
    *,
    step: int,
    key_prefix: str = "val/sample",
    max_size: int = 768,
    include_detail: bool = True,
) -> None:
    """Upload generic SR panels to SwanLab."""
    if not _active or not panels:
        return

    detail_note = (
        "\nerr(codec LR↑) | err(SR) | zoom(SR|HR)@max-error"
        if include_detail
        else ""
    )
    caption = f"codec LR (NN x3) | SR | HR{detail_note}"

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
    val_loader: DataLoader,
    device: torch.device,
    *,
    step: int,
    num_samples: int = 4,
    max_size: int = 768,
    error_gain: float = 8.0,
    include_detail: bool = True,
    colorspace: str = "rgb",
) -> None:
    """Collect and upload validation visualizations to SwanLab."""
    dataset = val_loader.dataset
    if isinstance(dataset, FixedValDataset):
        indices = select_val_vis_indices(dataset.specs, num_samples)
        samples = collect_codec_val_vis_samples(
            model,
            dataset,
            device,
            indices=indices,
            colorspace=colorspace,
            error_gain=error_gain,
            include_detail=include_detail,
        )
        log_val_vis_samples(
            samples,
            step=step,
            max_size=max_size,
            include_detail=include_detail,
        )
        return

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
        max_size=max_size,
        include_detail=include_detail,
    )
