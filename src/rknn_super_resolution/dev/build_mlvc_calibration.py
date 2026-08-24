"""Build RKNN calibration inputs from frozen-MLVC OpenVidHD reconstructions."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rknn_super_resolution.config import load_config
from rknn_super_resolution.data.mlvc_loader import build_mlvc_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset_description", type=str, default=None)
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--mlvc_repo", type=str, default=None)
    parser.add_argument("--mlvc_checkpoint", type=str, default=None)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--codec-context",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include the MLVC decoder-feature input",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("data/rknn_calib_ycbcr"))
    parser.add_argument("--output_list", type=Path, default=Path("data/rknn_calib.txt"))
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MLVC calibration generation requires CUDA")

    project_root = Path(__file__).resolve().parents[3]
    app_cfg = load_config(args.config)
    data = app_cfg.data
    model_cfg = app_cfg.model
    overrides = {
        "dataset_description": args.dataset_description,
        "video_root": args.video_root,
        "mlvc_repo": args.mlvc_repo,
        "mlvc_checkpoint": args.mlvc_checkpoint,
        "codec_context": args.codec_context,
    }
    data = replace(
        data,
        val_samples=args.samples,
        **{name: value for name, value in overrides.items() if value is not None},
    )
    train_loader, val_loader = build_mlvc_loaders(
        data,
        device=device,
        batch_size=1,
        scale=3,
        colorspace="yuv",
        train_aug=False,
        val_batch_size=1,
        rank=0,
        world_size=1,
        project_root=project_root,
    )

    output_dir = args.output_dir.resolve()
    output_list = args.output_list.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_list.parent.mkdir(parents=True, exist_ok=True)
    paths: list[tuple[Path, Path | None]] = []
    try:
        for model_input, _hr in val_loader:
            if isinstance(model_input, torch.Tensor):
                lr_ycbcr, codec_features = model_input, None
            else:
                lr_ycbcr, codec_features = model_input
            for offset, sample in enumerate(lr_ycbcr):
                sample = sample.clamp(0.0, 255.0).round().byte()
                packed = (
                    torch.nn.functional.pixel_unshuffle(sample.unsqueeze(0), model_cfg.phase_factor)
                    .cpu()
                    .numpy()
                )
                phase_path = output_dir / f"mlvc_{len(paths):04d}_phases.npy"
                np.save(phase_path, packed)
                codec_path = None
                if codec_features is not None:
                    codec_path = output_dir / f"mlvc_{len(paths):04d}_codec.npy"
                    np.save(codec_path, codec_features[offset : offset + 1].cpu().numpy())
                paths.append((phase_path, codec_path))
                if len(paths) >= args.samples:
                    break
            if len(paths) >= args.samples:
                break
    finally:
        train_loader.close()

    lines = [f"{phase} {codec}\n" if codec is not None else f"{phase}\n" for phase, codec in paths]
    output_list.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {len(paths)} MLVC YCbCr444 calibration inputs to {output_list}")


if __name__ == "__main__":
    main()
