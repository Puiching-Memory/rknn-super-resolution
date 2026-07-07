"""Build multi-frame HR mezzanine MP4 from a still image (DIV2K)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype

from rk3588_mobile_sr.data_pipeline.codec_args import encode_rgb_clip_to_mp4


def _load_rgb(path: Path) -> torch.Tensor:
    tensor = read_image(str(path))
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[0] == 4:
        tensor = tensor[:3]
    return convert_image_dtype(tensor, torch.float32) * 255.0


def _random_crop(
    image: torch.Tensor, crop_h: int, crop_w: int, rng: random.Random
) -> torch.Tensor:
    _, h, w = image.shape
    if h < crop_h or w < crop_w:
        return F.interpolate(
            image.unsqueeze(0), size=(crop_h, crop_w), mode="area"
        ).squeeze(0)
    top = rng.randint(0, h - crop_h)
    left = rng.randint(0, w - crop_w)
    return image[:, top : top + crop_h, left : left + crop_w]


def build_image_mezzanine(
    *,
    image_path: Path,
    output_path: Path,
    source_id: str,
    canvas_h: int,
    canvas_w: int,
    clips_per_video: int,
    fps: int,
    mezzanine_gop: int,
) -> None:
    image = _load_rgb(image_path)
    crops = []
    for clip_start in range(max(1, clips_per_video)):
        crop_rng = random.Random(f"{source_id}:{clip_start}")
        frame = _random_crop(image, canvas_h, canvas_w, crop_rng)
        crops.append(frame.unsqueeze(0))
    hr_clip = torch.cat(crops, dim=0)
    encode_rgb_clip_to_mp4(
        hr_clip,
        output_path,
        fps=fps,
        codec="libx264",
        gop=max(mezzanine_gop, len(crops)),
        bitrate_kbps=20_000,
        preset="veryfast",
        tune=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--canvas-h", type=int, required=True)
    parser.add_argument("--canvas-w", type=int, required=True)
    parser.add_argument("--clips-per-video", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mezzanine-gop", type=int, default=1)
    args = parser.parse_args()
    build_image_mezzanine(
        image_path=args.image_path.resolve(),
        output_path=args.output_path.resolve(),
        source_id=args.source_id,
        canvas_h=args.canvas_h,
        canvas_w=args.canvas_w,
        clips_per_video=args.clips_per_video,
        fps=args.fps,
        mezzanine_gop=args.mezzanine_gop,
    )


if __name__ == "__main__":
    main()
