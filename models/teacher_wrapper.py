"""Teacher model wrapper for knowledge distillation."""

import torch
import torch.nn as nn


class TeacherWrapper(nn.Module):
    """Wrap an arbitrary teacher network; enforces no_grad and same range [0,255]."""

    def __init__(self, model: nn.Module, scale: int = 3):
        super().__init__()
        self.model = model
        self.scale = scale
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            out = self.model(x)
        # Normalize teacher output to [0, 255] if needed
        if out.max() <= 1.0:
            out = out * 255.0
        return out


def load_teacher(
    arch: str, weight_path: str, scale: int = 3, device: str = "cuda"
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
        from torchsr.models import edsr

        teacher = edsr(scale=scale, pretrained=False)
        teacher.load_state_dict(torch.load(weight_path, map_location="cpu"))
    else:
        raise ValueError(f"Unknown teacher arch: {arch}")

    return TeacherWrapper(teacher.to(device), scale=scale)
