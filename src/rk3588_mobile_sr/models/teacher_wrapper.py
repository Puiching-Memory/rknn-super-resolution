"""Teacher model wrapper for knowledge distillation."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from rk3588_mobile_sr.models.mambairv2_light import build_mambairv2_light
from rk3588_mobile_sr.models.teacher_checkpoint import load_teacher_state_dict
from rk3588_mobile_sr.utils.train_framework import resolve_amp_dtype

TEACHER_ARCH_CHOICES = ("mambairv2_light", "real_esrgan")


class TeacherWrapper(nn.Module):
    """Wrap an arbitrary teacher network for fast inference in [0, 255]."""

    def __init__(self, model: nn.Module, scale: int = 3):
        super().__init__()
        self.model = model
        self.scale = scale
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MambaIRv2Light expects [0, 1] RGB with internal mean normalization.
        x_in = x / 255.0 if x.max() > 1.0 else x
        if x.is_cuda:
            amp_ctx = torch.amp.autocast("cuda", dtype=resolve_amp_dtype())
        else:
            amp_ctx = nullcontext()
        with amp_ctx:
            out = self.model(x_in)
        return (out.clamp(0.0, 1.0) * 255.0).float()


def load_teacher(
    arch: str,
    weight_path: str,
    scale: int = 3,
    device: str = "cuda",
    *,
    compile_model: bool = False,
) -> TeacherWrapper:
    """Load a frozen teacher for Stage 2 distillation.

    Supported:
      - ``mambairv2_light``: MambaIRv2Light (AIO_MAI default teacher family)
      - ``real_esrgan``: placeholder for Real-ESRGAN integration
    """
    if arch == "real_esrgan":
        raise NotImplementedError(
            "real_esrgan teacher requires RealESRGANer; please wire in your checkpoint."
        )
    if arch == "mambairv2_light":
        teacher = build_mambairv2_light(upscale=scale)
        state = load_teacher_state_dict(weight_path)
        teacher.load_state_dict(state, strict=True)
    else:
        raise ValueError(f"Unknown teacher arch: {arch!r}. Choose from {TEACHER_ARCH_CHOICES}.")

    model = teacher.to(device)
    # Mamba selective_scan + torch.compile is fragile; default off for teachers.
    if compile_model:
        model = torch.compile(model)
    return TeacherWrapper(model, scale=scale)
