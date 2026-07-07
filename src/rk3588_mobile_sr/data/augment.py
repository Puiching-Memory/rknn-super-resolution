"""Online LR/HR paired augmentations via Kornia."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import kornia.augmentation as K
import torch


@dataclass(frozen=True)
class AugmentConfig:
    """Training augmentations for canvas codec LR/HR pairs."""

    # Paired geometric (LR + HR, same random params).
    flip_h: bool = True
    flip_v: bool = True
    rot180: bool = True
    rot90: bool = False
    rot90_resize_back: bool = True

    # LR-only degradations (HR stays clean supervision).
    lr_gaussian_blur: bool = False
    lr_motion_blur: bool = False
    lr_gaussian_noise: bool = False
    lr_jpeg: bool = False
    blur_p: float = 0.3
    motion_blur_p: float = 0.2
    noise_p: float = 0.3
    jpeg_p: float = 0.3
    noise_std: float = 5.0
    jpeg_quality: tuple[float, float] = (40.0, 85.0)


@lru_cache(maxsize=32)
def build_paired_augment(config: AugmentConfig) -> K.AugmentationSequential:
    """Geometric augmentations with shared random params across LR and HR."""
    ops: list[K.Augmentation] = []
    if config.flip_h:
        ops.append(K.RandomHorizontalFlip(p=0.5))
    if config.flip_v:
        ops.append(K.RandomVerticalFlip(p=0.5))
    if config.rot90:
        ops.append(
            K.RandomRotation90(
                times=(1, 3),
                p=0.5,
                keepdim=config.rot90_resize_back,
            )
        )
    elif config.rot180:
        ops.append(K.RandomRotation(degrees=180.0, p=0.5))
    return K.AugmentationSequential(
        *ops,
        data_keys=["input", "input"],
        same_on_batch=False,
    )


@lru_cache(maxsize=32)
def build_lr_degrade_augment(config: AugmentConfig) -> K.AugmentationSequential:
    """Blur / noise / JPEG on LR input only (simulates extra capture/codec stress)."""
    ops: list[K.Augmentation] = []
    if config.lr_gaussian_blur:
        ops.append(
            K.RandomGaussianBlur(
                kernel_size=(3, 7),
                sigma=(0.5, 2.0),
                p=config.blur_p,
            )
        )
    if config.lr_motion_blur:
        ops.append(
            K.RandomMotionBlur(
                kernel_size=5,
                angle=(-15.0, 15.0),
                direction=(-1.0, 1.0),
                p=config.motion_blur_p,
            )
        )
    if config.lr_gaussian_noise:
        ops.append(
            K.RandomGaussianNoise(
                mean=0.0,
                std=config.noise_std,
                p=config.noise_p,
            )
        )
    if config.lr_jpeg:
        ops.append(
            K.RandomJPEG(
                jpeg_quality=config.jpeg_quality,
                p=config.jpeg_p,
            )
        )
    return K.AugmentationSequential(*ops)


def batch_random_crop_pair(
    lr: torch.Tensor,
    hr: torch.Tensor,
    *,
    lr_crop_h: int,
    lr_crop_w: int,
    scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample random crop with LR/HR alignment (BCHW).

    Kornia ``RandomCrop`` applies the same pixel size to both tensors; for x3 SR
    we crop ``scale``-larger regions on HR, so this stays explicit.
    """
    b, _, lr_h, lr_w = lr.shape
    if lr_crop_h > lr_h or lr_crop_w > lr_w:
        raise ValueError(f"crop {lr_crop_h}x{lr_crop_w} exceeds LR {lr_h}x{lr_w}")

    hr_crop_h, hr_crop_w = lr_crop_h * scale, lr_crop_w * scale
    tops = torch.randint(0, lr_h - lr_crop_h + 1, (b,), device=lr.device)
    lefts = torch.randint(0, lr_w - lr_crop_w + 1, (b,), device=lr.device)

    out_lr = torch.empty((b, lr.shape[1], lr_crop_h, lr_crop_w), device=lr.device, dtype=lr.dtype)
    out_hr = torch.empty((b, hr.shape[1], hr_crop_h, hr_crop_w), device=hr.device, dtype=hr.dtype)
    for i in range(b):
        top = int(tops[i])
        left = int(lefts[i])
        out_lr[i] = lr[i, :, top : top + lr_crop_h, left : left + lr_crop_w]
        hr_top, hr_left = top * scale, left * scale
        out_hr[i] = hr[i, :, hr_top : hr_top + hr_crop_h, hr_left : hr_left + hr_crop_w]
    return out_lr, out_hr


def augment_config_for_canvas(
    *,
    augment_rot90: bool = False,
    patch_size: int | None = None,
    augment_lr_blur: bool = False,
    augment_lr_motion_blur: bool = False,
    augment_lr_noise: bool = False,
    augment_lr_jpeg: bool = False,
) -> AugmentConfig:
    """Build AugmentConfig for full-canvas or patch training."""
    use_rot90 = augment_rot90 or patch_size is not None
    return AugmentConfig(
        rot90=use_rot90,
        rot90_resize_back=patch_size is None,
        lr_gaussian_blur=augment_lr_blur,
        lr_motion_blur=augment_lr_motion_blur,
        lr_gaussian_noise=augment_lr_noise,
        lr_jpeg=augment_lr_jpeg,
    )


def batch_augment_pair(
    lr: torch.Tensor,
    hr: torch.Tensor,
    *,
    config: AugmentConfig,
    lr_canvas: tuple[int, int],
    hr_canvas: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply geometric (paired) then degradation (LR-only) augmentations."""
    del lr_canvas, hr_canvas
    lr, hr = build_paired_augment(config)(lr, hr)
    lr = build_lr_degrade_augment(config)(lr)
    lr = lr.clamp(0.0, 255.0)
    return lr, hr
