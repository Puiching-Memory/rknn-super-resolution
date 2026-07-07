"""Still-image HR source for offline codec clip generation."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype

from rk3588_mobile_sr.data.types import SourceRecord


def _load_rgb_image(path: Path) -> torch.Tensor:
    """Load an image as CHW float RGB in [0, 255]."""
    tensor = read_image(str(path))
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[0] == 4:
        tensor = tensor[:3]
    return convert_image_dtype(tensor, torch.float32) * 255.0


def _random_crop(
    image: torch.Tensor,
    crop_h: int,
    crop_w: int,
    rng: random.Random,
) -> torch.Tensor:
    """Crop CHW RGB; resize with area if the image is smaller than the crop."""
    _, h, w = image.shape
    if h < crop_h or w < crop_w:
        return F.interpolate(
            image.unsqueeze(0),
            size=(crop_h, crop_w),
            mode="area",
        ).squeeze(0)
    top = rng.randint(0, h - crop_h)
    left = rng.randint(0, w - crop_w)
    return image[:, top : top + crop_h, left : left + crop_w]


class StillImageSource:
    """Single-frame clips from a still HR image with random crops."""

    def __init__(self, record: SourceRecord, project_root: Path) -> None:
        if record.height <= 0 or record.width <= 0:
            raise ValueError(f"image record {record.id} requires width/height")
        self.record = record
        self.canvas_h = record.height
        self.canvas_w = record.width
        image_path = (project_root / record.path).resolve()
        self.image = _load_rgb_image(image_path)

    @property
    def fps(self) -> int:
        return self.record.fps or 30

    def clip_starts(self, clips_per_video: int, *, rng: random.Random) -> list[int]:
        del rng
        return list(range(max(1, clips_per_video)))

    def read_clip(self, clip_start: int) -> torch.Tensor:
        """Return a one-frame NCHW clip (random crop indexed by ``clip_start``)."""
        crop_rng = random.Random(f"{self.record.id}:{clip_start}")
        frame = _random_crop(self.image, self.canvas_h, self.canvas_w, crop_rng)
        return frame.unsqueeze(0)
