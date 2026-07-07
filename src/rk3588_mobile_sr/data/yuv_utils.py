"""BT.601 full-range RGB <-> YUV444 conversions for training in YUV space."""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Coefficients aligned with rknn_eval.rgb_to_nv12_planes (OpenCV BT.601 when available).


def rgb_to_yuv444(bchw: torch.Tensor) -> torch.Tensor:
    """Convert BCHW RGB in [0, 255] to BCHW YUV444 in [0, 255]."""
    r = bchw[:, 0]
    g = bchw[:, 1]
    b = bchw[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.169 * r - 0.331 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.419 * g - 0.081 * b + 128.0
    return torch.stack([y, u, v], dim=1).clamp(0.0, 255.0)


def yuv444_to_rgb(tensor: torch.Tensor) -> torch.Tensor:
    """Convert BCHW or CHW YUV444 in [0, 255] to RGB in [0, 255]."""
    if tensor.ndim == 3:
        return yuv444_to_rgb(tensor.unsqueeze(0)).squeeze(0)
    y = tensor[:, 0]
    u = tensor[:, 1] - 128.0
    v = tensor[:, 2] - 128.0
    r = y + 1.402 * v
    g = y - 0.344 * u - 0.714 * v
    b = y + 1.772 * u
    return torch.stack([r, g, b], dim=1).clamp(0.0, 255.0)


def simulate_nv12_roundtrip_rgb(bchw: torch.Tensor) -> torch.Tensor:
    """RGB -> YUV444 -> 4:2:0 chroma subsample/upsample -> RGB (NV12-like)."""
    if bchw.ndim == 3:
        return simulate_nv12_roundtrip_rgb(bchw.unsqueeze(0)).squeeze(0)
    h, w = bchw.shape[-2:]
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"NV12 simulation requires even H and W, got {h}x{w}")
    yuv = rgb_to_yuv444(bchw)
    y = yuv[:, 0:1]
    u = yuv[:, 1:2]
    v = yuv[:, 2:3]
    u_ds = F.avg_pool2d(u, kernel_size=2, stride=2)
    v_ds = F.avg_pool2d(v, kernel_size=2, stride=2)
    u_up = F.interpolate(u_ds, size=(h, w), mode="bilinear", align_corners=False)
    v_up = F.interpolate(v_ds, size=(h, w), mode="bilinear", align_corners=False)
    return yuv444_to_rgb(torch.cat([y, u_up, v_up], dim=1))


def rgb_to_model_colorspace(
    bchw: torch.Tensor,
    *,
    colorspace: str,
    nv12_simulate: bool = False,
) -> torch.Tensor:
    """Optional NV12 roundtrip on RGB, then convert to train/val tensor layout."""
    rgb = simulate_nv12_roundtrip_rgb(bchw) if nv12_simulate else bchw
    return maybe_rgb_to_colorspace(rgb, colorspace)


def maybe_rgb_to_colorspace(tensor: torch.Tensor, colorspace: str) -> torch.Tensor:
    if colorspace == "rgb":
        return tensor
    if colorspace == "yuv":
        if tensor.ndim == 3:
            return rgb_to_yuv444(tensor.unsqueeze(0)).squeeze(0)
        return rgb_to_yuv444(tensor)
    raise ValueError(f"Unsupported colorspace {colorspace!r}; expected 'rgb' or 'yuv'.")


def colorspace_roundtrip_rgb(tensor: torch.Tensor, colorspace: str) -> torch.Tensor:
    """Apply train/val colorspace encode+decode and return RGB for diagnostics."""
    if colorspace == "rgb":
        return tensor
    if colorspace == "yuv":
        if tensor.ndim == 3:
            return yuv444_to_rgb(maybe_rgb_to_colorspace(tensor, "yuv"))
        return yuv444_to_rgb(maybe_rgb_to_colorspace(tensor, "yuv"))
    raise ValueError(f"Unsupported colorspace {colorspace!r}; expected 'rgb' or 'yuv'.")
