"""Super-resolution validation metrics (PSNR, Y-PSNR, SSIM, L1)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rk3588_mobile_sr.data.val_loader import FixedValDataset, iter_val_batches, select_val_vis_indices, val_spec_slug

# UVG canvas samples (diverse sequences) tracked each validation step.
DEFAULT_FIXED_VAL_TRACKS: int = 3


@dataclass
class ValidationMetrics:
    """Aggregated validation metrics over the full val set."""

    psnr: float
    y_psnr: float
    ssim: float
    l1: float
    psnr_min: float
    psnr_p10: float
    psnr_p50: float
    psnr_p90: float
    dists: float | None = None
    fixed_psnr: dict[str, float] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "val/psnr": self.psnr,
            "val/y_psnr": self.y_psnr,
            "val/ssim": self.ssim,
            "val/l1": self.l1,
            "val/psnr_min": self.psnr_min,
            "val/psnr_p10": self.psnr_p10,
            "val/psnr_p50": self.psnr_p50,
            "val/psnr_p90": self.psnr_p90,
        }
        if self.dists is not None:
            metrics["val/dists"] = self.dists
        for slug, psnr in self.fixed_psnr.items():
            metrics[f"val/fixed_{slug}_psnr"] = psnr
        return metrics


def shave_borders(tensor: torch.Tensor, shave: int) -> torch.Tensor:
    """Crop ``shave`` pixels from each spatial border."""
    if shave <= 0:
        return tensor
    if tensor.shape[-1] <= 2 * shave or tensor.shape[-2] <= 2 * shave:
        return tensor
    return tensor[..., shave:-shave, shave:-shave]


def rgb_to_y_batch(bchw: torch.Tensor) -> torch.Tensor:
    """Convert BCHW RGB in [0, 255] to luminance Y (MATLAB-style)."""
    weights = torch.tensor([65.481, 128.553, 24.966], device=bchw.device, dtype=bchw.dtype)
    return (bchw * weights.view(1, 3, 1, 1)).sum(dim=1) / 255.0 + 16.0


def batch_psnr(pred: torch.Tensor, target: torch.Tensor, *, shave: int = 0) -> torch.Tensor:
    """Per-sample PSNR for BCHW tensors in [0, 255]."""
    pred = shave_borders(pred, shave)
    target = shave_borders(target, shave)
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return 10.0 * torch.log10(255.0 * 255.0 / mse.clamp_min(1e-12))


def batch_y_psnr(pred: torch.Tensor, target: torch.Tensor, *, shave: int = 0) -> torch.Tensor:
    """Per-sample Y-channel PSNR for BCHW RGB tensors in [0, 255]."""
    pred_y = rgb_to_y_batch(pred)
    target_y = rgb_to_y_batch(target)
    pred_y = shave_borders(pred_y.unsqueeze(1), shave).squeeze(1)
    target_y = shave_borders(target_y.unsqueeze(1), shave).squeeze(1)
    mse = torch.mean((pred_y - target_y) ** 2, dim=[1, 2])
    return 10.0 * torch.log10(255.0 * 255.0 / mse.clamp_min(1e-12))


def batch_l1(pred: torch.Tensor, target: torch.Tensor, *, shave: int = 0) -> torch.Tensor:
    """Per-sample mean L1 for BCHW tensors in [0, 255]."""
    pred = shave_borders(pred, shave)
    target = shave_borders(target, shave)
    return torch.mean(torch.abs(pred - target), dim=[1, 2, 3])


def ssim_rgb(
    pred_hwc: np.ndarray,
    target_hwc: np.ndarray,
    *,
    shave: int = 0,
) -> float:
    """SSIM for HWC uint8/float RGB arrays in [0, 255]."""
    pred = pred_hwc.astype(np.float32)
    target = target_hwc.astype(np.float32)
    if shave > 0 and pred.shape[0] > 2 * shave and pred.shape[1] > 2 * shave:
        pred = pred[shave:-shave, shave:-shave]
        target = target[shave:-shave, shave:-shave]
    win = min(7, min(pred.shape[0], pred.shape[1]))
    if win % 2 == 0:
        win -= 1
    win = max(3, win)
    return float(
        ssim_metric(
            pred,
            target,
            data_range=255.0,
            channel_axis=-1,
            win_size=win,
        )
    )


def _rank_sample_indices(val_loader: DataLoader) -> list[int]:
    sampler = val_loader.sampler
    if isinstance(sampler, DistributedSampler):
        return list(sampler)
    return list(range(len(val_loader.dataset)))


def _aggregate_per_sample(
    records: list[tuple[int, float, float, float, float, float]],
    *,
    fixed_indices: tuple[int, ...],
    specs: list | None,
    has_dists: bool,
) -> ValidationMetrics:
    if not records:
        return ValidationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    records.sort(key=lambda row: row[0])
    psnrs = np.array([row[1] for row in records], dtype=np.float64)
    y_psnrs = np.array([row[2] for row in records], dtype=np.float64)
    ssims = np.array([row[3] for row in records], dtype=np.float64)
    l1s = np.array([row[4] for row in records], dtype=np.float64)
    dists_vals = np.array([row[5] for row in records], dtype=np.float64) if has_dists else None

    by_index = {row[0]: row[1] for row in records}
    fixed_psnr: dict[str, float] = {}
    for idx in fixed_indices:
        if idx not in by_index:
            continue
        if specs is not None and 0 <= idx < len(specs):
            slug = val_spec_slug(specs[idx])
        else:
            slug = str(idx)
        fixed_psnr[slug] = by_index[idx]

    return ValidationMetrics(
        psnr=float(psnrs.mean()),
        y_psnr=float(y_psnrs.mean()),
        ssim=float(ssims.mean()),
        l1=float(l1s.mean()),
        psnr_min=float(psnrs.min()),
        psnr_p10=float(np.percentile(psnrs, 10)),
        psnr_p50=float(np.percentile(psnrs, 50)),
        psnr_p90=float(np.percentile(psnrs, 90)),
        dists=float(dists_vals.mean()) if dists_vals is not None else None,
        fixed_psnr=fixed_psnr,
    )


def _broadcast_scalar(value: float, rank: int, world_size: int, device: torch.device) -> float:
    if world_size <= 1:
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.broadcast(tensor, src=0)
    return float(tensor.item())


@torch.no_grad()
def validate_ddp_extended(
    model: nn.Module,
    val_loader: DataLoader,
    rank: int,
    world_size: int,
    *,
    scale: int = 3,
    fixed_indices: tuple[int, ...] | None = None,
    fixed_tracks: int = DEFAULT_FIXED_VAL_TRACKS,
    compute_dists: bool = False,
    colorspace: str = "rgb",
) -> tuple[float, ValidationMetrics | None]:
    """Run full validation; all ranks receive mean PSNR, rank 0 also gets full metrics."""
    from rk3588_mobile_sr.utils.pyiqa_metric import batch_perceptual_metric

    dataset = val_loader.dataset
    specs = dataset.specs if isinstance(dataset, FixedValDataset) else None
    if fixed_indices is None and specs is not None:
        fixed_indices = tuple(select_val_vis_indices(specs, fixed_tracks))
    elif fixed_indices is None:
        fixed_indices = ()

    if colorspace == "yuv":
        from rk3588_mobile_sr.data.yuv_utils import yuv444_to_rgb

    unwrap = getattr(model, "module", model)
    was_training = unwrap.training
    unwrap.eval()

    device = torch.device(f"cuda:{rank}")
    shave = scale
    sample_indices = _rank_sample_indices(val_loader)
    local_records: list[tuple[int, float, float, float, float, float]] = []
    offset = 0

    for lr, hr in iter_val_batches(val_loader):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        out = torch.clamp(unwrap(lr), 0.0, 255.0)

        if colorspace == "yuv":
            out_rgb = yuv444_to_rgb(out)
            hr_rgb = yuv444_to_rgb(hr)
            psnr_b = batch_psnr(out_rgb, hr_rgb, shave=shave)
            y_psnr_b = batch_psnr(out[:, :1], hr[:, :1], shave=shave)
            l1_b = batch_l1(out, hr, shave=shave)
            dists_b = (
                batch_perceptual_metric(out_rgb, hr_rgb, device=device, shave=shave)
                if compute_dists
                else None
            )
        else:
            psnr_b = batch_psnr(out, hr, shave=shave)
            y_psnr_b = batch_y_psnr(out, hr, shave=shave)
            l1_b = batch_l1(out, hr, shave=shave)
            dists_b = (
                batch_perceptual_metric(out, hr, device=device, shave=shave)
                if compute_dists
                else None
            )

        batch_size = out.shape[0]
        batch_indices = sample_indices[offset : offset + batch_size]
        offset += batch_size

        for i, global_idx in enumerate(batch_indices):
            if colorspace == "yuv":
                sr_np = yuv444_to_rgb(out[i].detach().float()).cpu().permute(1, 2, 0).numpy()
                hr_np = yuv444_to_rgb(hr[i].detach().float()).cpu().permute(1, 2, 0).numpy()
            else:
                sr_np = out[i].detach().float().cpu().permute(1, 2, 0).numpy()
                hr_np = hr[i].detach().float().cpu().permute(1, 2, 0).numpy()
            ssim_val = ssim_rgb(sr_np, hr_np, shave=shave)
            dists_val = float(dists_b[i].item()) if dists_b is not None else 0.0
            local_records.append(
                (
                    global_idx,
                    float(psnr_b[i].item()),
                    float(y_psnr_b[i].item()),
                    ssim_val,
                    float(l1_b[i].item()),
                    dists_val,
                )
            )

    if was_training:
        unwrap.train()

    if world_size > 1:
        gathered: list[list[tuple[int, float, float, float, float, float]] | None] = [
            None
        ] * world_size
        dist.all_gather_object(gathered, local_records)
    else:
        gathered = [local_records]

    metrics: ValidationMetrics | None = None
    psnr = 0.0
    if rank == 0:
        merged: list[tuple[int, float, float, float, float, float]] = []
        for part in gathered:
            if part:
                merged.extend(part)
        metrics = _aggregate_per_sample(
            merged,
            fixed_indices=fixed_indices,
            specs=specs,
            has_dists=compute_dists,
        )
        psnr = metrics.psnr

    psnr = _broadcast_scalar(psnr, rank, world_size, device)
    return psnr, metrics


@torch.no_grad()
def validate_ddp(
    model: nn.Module,
    val_loader: DataLoader,
    rank: int,
    world_size: int,
    *,
    scale: int = 3,
) -> float:
    """Evaluate mean RGB PSNR on the validation set (DDP-safe, border shave)."""
    psnr, _ = validate_ddp_extended(
        model,
        val_loader,
        rank,
        world_size,
        scale=scale,
        fixed_indices=(),
    )
    return psnr
