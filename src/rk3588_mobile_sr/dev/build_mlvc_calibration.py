"""Build RKNN calibration inputs from frozen-MLVC OpenVidHD reconstructions."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image

from rk3588_mobile_sr.config import load_config
from rk3588_mobile_sr.data.mlvc_loader import build_mlvc_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset_description", type=str, default=None)
    parser.add_argument("--mlvc_repo", type=str, default=None)
    parser.add_argument("--mlvc_checkpoint", type=str, default=None)
    parser.add_argument("--samples", type=int, default=100)
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
    data = load_config(args.config).data
    overrides = {
        "dataset_description": args.dataset_description,
        "mlvc_repo": args.mlvc_repo,
        "mlvc_checkpoint": args.mlvc_checkpoint,
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
        patch_size=None,
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
    paths: list[Path] = []
    try:
        for lr_ycbcr, _hr in val_loader:
            for sample in lr_ycbcr:
                array = (
                    sample.clamp(0.0, 255.0)
                    .round()
                    .byte()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                path = output_dir / f"mlvc_{len(paths):04d}.png"
                Image.fromarray(array).save(path)
                paths.append(path)
                if len(paths) >= args.samples:
                    break
            if len(paths) >= args.samples:
                break
    finally:
        train_loader.close()

    output_list.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
    print(f"wrote {len(paths)} MLVC YCbCr444 calibration inputs to {output_list}")


if __name__ == "__main__":
    main()
