"""Evaluate PSNR/SSIM on validation set."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from rknn_super_resolution.config import load_config
from rknn_super_resolution.models import PhaseRLFNSR


def parse_args():
    cfg = load_config().model
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--lr_dir", type=str, default=None)
    parser.add_argument("--scale", type=int, default=cfg.scale)
    parser.add_argument("--num_channels", type=int, default=cfg.num_channels)
    parser.add_argument("--num_blocks", type=int, default=cfg.num_blocks)
    parser.add_argument("--phase_factor", type=int, default=cfg.phase_factor)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save_dir", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    cfg = load_config().model
    model = PhaseRLFNSR(
        scale=args.scale,
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        phase_factor=args.phase_factor,
        codec_feature_channels=cfg.codec_feature_channels,
        codec_project_channels=cfg.codec_project_channels,
        codec_upsample_factor=cfg.codec_upsample_factor,
    ).to(device)
    raw = torch.load(args.weight, map_location=device, weights_only=False)
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    model.load_state_dict(state_dict)
    model.switch_to_deploy()
    model.eval()

    hr_paths = sorted(Path(args.hr_dir).glob("*.png"))
    if len(hr_paths) == 0:
        hr_paths = sorted(Path(args.hr_dir).glob("*.jpg"))

    psnr_list = []
    ssim_list = []
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for hr_path in hr_paths:
        hr = Image.open(hr_path).convert("RGB")
        if args.lr_dir:
            lr = Image.open(Path(args.lr_dir) / hr_path.name).convert("RGB")
        else:
            w, h = hr.size
            lr = hr.resize((w // args.scale, h // args.scale), Image.BICUBIC)

        hr_t = TF.to_tensor(hr).unsqueeze(0).to(device) * 255.0
        lr_t = TF.to_tensor(lr).unsqueeze(0).to(device) * 255.0

        sr_t = torch.clamp(model(lr_t), 0.0, 255.0)

        mse = torch.mean((sr_t - hr_t) ** 2).item()
        psnr = 10 * np.log10(255.0 * 255.0 / mse)
        psnr_list.append(psnr)

        sr_np = sr_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        hr_np = hr_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        ssim_val = ssim(
            sr_np,
            hr_np,
            data_range=255.0,
            channel_axis=-1,
            win_size=min(7, min(sr_np.shape[:2]) | 1),
        )
        ssim_list.append(ssim_val)

        if save_dir:
            sr_img = Image.fromarray(sr_np.astype(np.uint8))
            sr_img.save(save_dir / hr_path.name)

    print(f"PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"SSIM: {np.mean(ssim_list):.4f}")


def evaluate_cli() -> None:
    """Console entry point for PSNR/SSIM evaluation."""
    evaluate(parse_args())


if __name__ == "__main__":
    evaluate_cli()
