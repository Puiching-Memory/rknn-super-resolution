"""Super-resolution validation metrics (PSNR, Y-PSNR, SSIM, L1)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


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
    vmaf: float | None = None

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
        if self.vmaf is not None:
            metrics["val/vmaf"] = self.vmaf
        return metrics


def iter_val_batches(
    val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield already-processed MLVC validation batches."""
    yield from val_loader


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


def _module_device(module: nn.Module, rank: int) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def _rank_sample_indices(val_loader: DataLoader | object) -> list[int]:
    sampler = getattr(val_loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        return list(sampler)
    dataset = getattr(val_loader, "dataset", None)
    if dataset is not None:
        return list(range(len(dataset)))
    return list(range(len(val_loader)))


def _aggregate_per_sample(
    records: list[tuple[int, float, float, float, float, float, float]],
    *,
    has_dists: bool,
    has_vmaf: bool,
) -> ValidationMetrics:
    if not records:
        return ValidationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    records = sorted({row[0]: row for row in records}.values(), key=lambda row: row[0])
    psnrs = np.array([row[1] for row in records], dtype=np.float64)
    y_psnrs = np.array([row[2] for row in records], dtype=np.float64)
    ssims = np.array([row[3] for row in records], dtype=np.float64)
    l1s = np.array([row[4] for row in records], dtype=np.float64)
    dists_vals = np.array([row[5] for row in records], dtype=np.float64) if has_dists else None
    vmaf_vals = np.array([row[6] for row in records], dtype=np.float64) if has_vmaf else None

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
        vmaf=float(vmaf_vals.mean()) if vmaf_vals is not None else None,
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
    val_loader: DataLoader | object,
    rank: int,
    world_size: int,
    *,
    scale: int = 3,
    compute_dists: bool = False,
    compute_vmaf: bool = True,
    vmaf_model: str = "1080p",
    vmaf_enc_size: tuple[int, int] | None = (360, 640),
    colorspace: str = "yuv",
) -> tuple[float, ValidationMetrics | None]:
    """Run full validation; primary score is VMAF when enabled, else PSNR."""
    from rk3588_mobile_sr.utils.pyiqa_metric import batch_perceptual_metric
    from rk3588_mobile_sr.utils.vmaf_metric import batch_vmaf

    if colorspace == "yuv":
        from rk3588_mobile_sr.data.yuv_utils import yuv444_to_rgb

    unwrap = getattr(model, "module", model)
    was_training = unwrap.training
    unwrap.eval()

    device = _module_device(unwrap, rank)
    shave = scale
    sample_indices = _rank_sample_indices(val_loader)
    local_records: list[tuple[int, float, float, float, float, float, float]] = []
    offset = 0
    enc_h, enc_w = vmaf_enc_size if vmaf_enc_size is not None else (None, None)

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
            vmaf_b = (
                batch_vmaf(
                    out_rgb.cpu(),
                    hr_rgb.cpu(),
                    model=vmaf_model,
                    enc_width=enc_w,
                    enc_height=enc_h,
                )
                if compute_vmaf
                else None
            )
        else:
            out_rgb = out
            hr_rgb = hr
            psnr_b = batch_psnr(out, hr, shave=shave)
            y_psnr_b = batch_y_psnr(out, hr, shave=shave)
            l1_b = batch_l1(out, hr, shave=shave)
            dists_b = (
                batch_perceptual_metric(out, hr, device=device, shave=shave)
                if compute_dists
                else None
            )
            vmaf_b = (
                batch_vmaf(
                    out_rgb.cpu(),
                    hr_rgb.cpu(),
                    model=vmaf_model,
                    enc_width=enc_w,
                    enc_height=enc_h,
                )
                if compute_vmaf
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
            vmaf_val = float(vmaf_b[i].item()) if vmaf_b is not None else 0.0
            local_records.append(
                (
                    global_idx,
                    float(psnr_b[i].item()),
                    float(y_psnr_b[i].item()),
                    ssim_val,
                    float(l1_b[i].item()),
                    dists_val,
                    vmaf_val,
                )
            )

    if was_training:
        unwrap.train()

    if world_size > 1:
        gathered: list[list[tuple[int, float, float, float, float, float, float]] | None] = [
            None
        ] * world_size
        dist.all_gather_object(gathered, local_records)
    else:
        gathered = [local_records]

    metrics: ValidationMetrics | None = None
    primary = 0.0
    if rank == 0:
        merged: list[tuple[int, float, float, float, float, float, float]] = []
        for part in gathered:
            if part:
                merged.extend(part)
        metrics = _aggregate_per_sample(
            merged,
            has_dists=compute_dists,
            has_vmaf=compute_vmaf,
        )
        primary = (
            float(metrics.vmaf)
            if compute_vmaf and metrics.vmaf is not None
            else metrics.psnr
        )

    primary = _broadcast_scalar(primary, rank, world_size, device)
    return primary, metrics


@torch.no_grad()
def validate_ddp(
    model: nn.Module,
    val_loader: DataLoader | object,
    rank: int,
    world_size: int,
    *,
    scale: int = 3,
    colorspace: str = "yuv",
) -> float:
    """Evaluate mean primary val score (VMAF by default) on the validation set."""
    score, _ = validate_ddp_extended(
        model,
        val_loader,
        rank,
        world_size,
        scale=scale,
        colorspace=colorspace,
    )
    return score
