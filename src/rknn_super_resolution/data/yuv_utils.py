"""MLVC-compatible BT.709 full-range RGB <-> YCbCr444 conversions."""

from __future__ import annotations

import torch

_KR = 0.2126
_KG = 0.7152
_KB = 0.0722
_CENTER = 127.5


def rgb_to_yuv444(bchw: torch.Tensor) -> torch.Tensor:
    """Convert CHW/BCHW RGB in [0, 255] to BT.709 YCbCr in [0, 255]."""
    r, g, b = bchw.chunk(3, dim=-3)
    y = _KR * r + _KG * g + _KB * b
    cb = 0.5 * (b - y) / (1.0 - _KB) + _CENTER
    cr = 0.5 * (r - y) / (1.0 - _KR) + _CENTER
    return torch.cat((y, cb, cr), dim=-3).clamp(0.0, 255.0)


def yuv444_to_rgb(tensor: torch.Tensor) -> torch.Tensor:
    """Convert BCHW or CHW YUV444 in [0, 255] to RGB in [0, 255]."""
    """Convert CHW/BCHW BT.709 YCbCr in [0, 255] to RGB in [0, 255]."""
    y, cb, cr = tensor.chunk(3, dim=-3)
    r = y + (2.0 - 2.0 * _KR) * (cr - _CENTER)
    b = y + (2.0 - 2.0 * _KB) * (cb - _CENTER)
    g = (y - _KR * r - _KB * b) / _KG
    return torch.cat((r, g, b), dim=-3).clamp(0.0, 255.0)


def maybe_rgb_to_colorspace(tensor: torch.Tensor, colorspace: str) -> torch.Tensor:
    if colorspace == "rgb":
        return tensor
    if colorspace == "yuv":
        return rgb_to_yuv444(tensor)
    raise ValueError(f"Unsupported colorspace {colorspace!r}; expected 'rgb' or 'yuv'.")


def colorspace_roundtrip_rgb(tensor: torch.Tensor, colorspace: str) -> torch.Tensor:
    """Apply train/val colorspace encode+decode and return RGB for diagnostics."""
    if colorspace == "rgb":
        return tensor
    if colorspace == "yuv":
        return yuv444_to_rgb(maybe_rgb_to_colorspace(tensor, "yuv"))
    raise ValueError(f"Unsupported colorspace {colorspace!r}; expected 'rgb' or 'yuv'.")
