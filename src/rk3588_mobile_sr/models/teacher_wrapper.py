"""Teacher model wrapper for knowledge distillation."""

from contextlib import nullcontext

import torch
import torch.nn as nn

from rk3588_mobile_sr.utils.train_framework import resolve_amp_dtype


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
        # torchsr EDSR expects [0, 1] RGB; training loaders provide [0, 255].
        x_in = x / 255.0 if x.max() > 1.0 else x
        if x.is_cuda:
            amp_ctx = torch.amp.autocast("cuda", dtype=resolve_amp_dtype())
        else:
            amp_ctx = nullcontext()
        with amp_ctx:
            out = self.model(x_in)
        if out.max() <= 1.0:
            out = out * 255.0
        return out.float()


def load_teacher(
    arch: str,
    weight_path: str,
    scale: int = 3,
    device: str = "cuda",
    *,
    compile_model: bool = True,
) -> TeacherWrapper:
    """Load a teacher model by architecture name.

    Supported:
      - "real_esrgan": expects basicsr model file
      - "edsr": expects torch state_dict for EDSR-baseline x3
    """
    if arch == "real_esrgan":
        # User can replace with RealESRGANer from basicsr/realesrgan
        raise NotImplementedError(
            "real_esrgan teacher requires RealESRGANer; please wire in your checkpoint."
        )
    elif arch == "edsr":
        from torchsr.models import edsr_baseline

        teacher = edsr_baseline(scale=scale, pretrained=False)
        teacher.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    else:
        raise ValueError(f"Unknown teacher arch: {arch}")

    model = teacher.to(device)
    if compile_model:
        model = torch.compile(model)
    return TeacherWrapper(model, scale=scale)
