"""Compare FP32 reference vs RKNN simulator outputs (PSNR / SSIM)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rknn_super_resolution.config import ModelConfig

try:
    import cv2
except ImportError:  # pragma: no cover - optional in minimal envs
    cv2 = None  # type: ignore[assignment]

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except ImportError:  # pragma: no cover
    skimage_ssim = None


class _RknnRuntime(Protocol):
    def inference(self, inputs: list[Any], data_format: str | None = None) -> list[Any]: ...


@dataclass(frozen=True)
class ImagePair:
    name: str
    lr_rgb: np.ndarray
    hr_rgb: np.ndarray


@dataclass
class AccuracyRow:
    label: str
    psnr_vs_hr: float
    ssim_vs_hr: float
    psnr_min: float
    psnr_max: float


@dataclass
class AccuracyReport:
    num_images: int
    input_h: int
    input_w: int
    fp32: AccuracyRow | None
    rknn: AccuracyRow
    match_psnr: float
    match_psnr_min: float
    quant_mode: str
    scale: int = 3

    @property
    def psnr_drop(self) -> float | None:
        if self.fp32 is None:
            return None
        return self.fp32.psnr_vs_hr - self.rknn.psnr_vs_hr

    @property
    def ssim_drop(self) -> float | None:
        if self.fp32 is None:
            return None
        return self.fp32.ssim_vs_hr - self.rknn.ssim_vs_hr


def psnr_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """RGB HWC float arrays in [0, 255]."""
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def ssim_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """RGB HWC float arrays in [0, 255]."""
    if skimage_ssim is not None:
        h, w = pred.shape[:2]
        win = min(7, h, w)
        if win % 2 == 0:
            win -= 1
        win = max(3, win)
        return float(
            skimage_ssim(
                pred,
                target,
                data_range=255.0,
                channel_axis=-1,
                win_size=win,
            )
        )

    # Fallback: luminance-only SSIM (no skimage).
    def _to_y(img: np.ndarray) -> np.ndarray:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        return 0.299 * r + 0.587 * g + 0.114 * b

    x = _to_y(pred)
    y = _to_y(target)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float(num / den) if den > 0 else 1.0


def _lr_path_for_hr(hr_path: Path, lr_dir: Path, scale: int) -> Path:
    scaled = lr_dir / f"{hr_path.stem}x{scale}{hr_path.suffix}"
    if scaled.exists():
        return scaled
    return lr_dir / hr_path.name


def _read_rgb(path: Path, size_wh: tuple[int, int] | None) -> np.ndarray:
    if cv2 is None:
        raise ImportError("opencv-python is required for RKNN eval image loading.")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if size_wh is not None:
        rgb = cv2.resize(rgb, size_wh, interpolation=cv2.INTER_CUBIC)
    return rgb


def collect_image_pairs(
    hr_dir: Path,
    lr_dir: Path | None,
    *,
    scale: int,
    input_h: int,
    input_w: int,
    max_images: int | None,
) -> list[ImagePair]:
    hr_paths = sorted(hr_dir.glob("*.png"))
    if not hr_paths:
        hr_paths = sorted(hr_dir.glob("*.jpg"))
    if max_images is not None:
        hr_paths = hr_paths[:max_images]

    out_h, out_w = input_h * scale, input_w * scale
    pairs: list[ImagePair] = []
    for hr_path in hr_paths:
        if lr_dir is not None:
            lr_path = _lr_path_for_hr(hr_path, lr_dir, scale)
            lr_rgb = _read_rgb(lr_path, (input_w, input_h))
        else:
            hr_full = _read_rgb(hr_path, None)
            hr_h, hr_w = hr_full.shape[:2]
            lr_rgb = _read_rgb(hr_path, (hr_w // scale, hr_h // scale))

        hr_rgb = _read_rgb(hr_path, (out_w, out_h)).astype(np.float32)
        pairs.append(ImagePair(name=hr_path.name, lr_rgb=lr_rgb, hr_rgb=hr_rgb))
    return pairs


def _aggregate(label: str, psnrs: list[float], ssims: list[float]) -> AccuracyRow:
    arr = np.asarray(psnrs, dtype=np.float64)
    return AccuracyRow(
        label=label,
        psnr_vs_hr=float(arr.mean()),
        ssim_vs_hr=float(np.mean(ssims)),
        psnr_min=float(arr.min()),
        psnr_max=float(arr.max()),
    )


def _rknn_output_to_hwc(output: np.ndarray) -> np.ndarray:
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0.0, 255.0)


def rgb_to_mlvc_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """Convert HWC RGB [0,255] to MLVC BT.709 full-range YCbCr."""
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = 0.5 * (b - y) / (1.0 - 0.0722) + 127.5
    cr = 0.5 * (r - y) / (1.0 - 0.2126) + 127.5
    return np.stack((y, cb, cr), axis=-1).clip(0.0, 255.0)


def mlvc_ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """Convert HWC MLVC BT.709 full-range YCbCr [0,255] to RGB."""
    values = ycbcr.astype(np.float32)
    y, cb, cr = values[..., 0], values[..., 1], values[..., 2]
    r = y + (2.0 - 2.0 * 0.2126) * (cr - 127.5)
    b = y + (2.0 - 2.0 * 0.0722) * (cb - 127.5)
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.stack((r, g, b), axis=-1).clip(0.0, 255.0)


def infer_rknn_rgb(
    runtime: _RknnRuntime,
    lr_rgb: np.ndarray,
    *,
    phase_factor: int = 2,
    scale: int = 3,
    codec_feature_channels: int = 0,
    codec_feature: np.ndarray | None = None,
) -> np.ndarray:
    """Combine an RGA-bicubic reference with RKNN phase residuals."""
    ycbcr = np.rint(rgb_to_mlvc_ycbcr(lr_rgb)).astype(np.uint8)
    base = resize_bicubic_hwc(ycbcr, scale)
    packed = pixel_unshuffle_hwc_to_nchw(ycbcr, phase_factor)
    inputs = [packed]
    if codec_feature_channels:
        codec_h = ((ycbcr.shape[0] + 15) // 16) * 2
        codec_w = ((ycbcr.shape[1] + 15) // 16) * 2
        if codec_feature is None:
            codec_feature = np.zeros(
                (1, codec_feature_channels, codec_h, codec_w), dtype=np.float32
            )
        inputs.append(np.asarray(codec_feature, dtype=np.float32))
    outputs = runtime.inference(inputs=inputs, data_format="nchw")
    residual = pixel_shuffle_nchw_to_hwc(outputs[0], scale * phase_factor)
    return mlvc_ycbcr_to_rgb(np.clip(base + residual, 0.0, 255.0))


def resize_bicubic_hwc(image: np.ndarray, scale: int) -> np.ndarray:
    """OpenCV reference for the board-side RGA bicubic base path."""
    if cv2 is None:
        raise ImportError("opencv-python is required for RKNN bicubic reconstruction.")
    if scale < 1:
        raise ValueError("scale must be positive")
    height, width = image.shape[:2]
    return np.clip(
        cv2.resize(
            image.astype(np.float32),
            (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC,
        ),
        0.0,
        255.0,
    )


def pixel_unshuffle_hwc_to_nchw(image: np.ndarray, factor: int) -> np.ndarray:
    """Pack an HWC image using PyTorch-compatible PixelUnshuffle ordering."""
    height, width, channels = image.shape
    if height % factor or width % factor:
        raise ValueError("image dimensions must be divisible by factor")
    nchw = np.transpose(image, (2, 0, 1))[np.newaxis]
    packed = nchw.reshape(1, channels, height // factor, factor, width // factor, factor).transpose(
        0, 1, 3, 5, 2, 4
    )
    return packed.reshape(1, channels * factor * factor, height // factor, width // factor)


def pixel_shuffle_nchw_to_hwc(phases: np.ndarray, factor: int) -> np.ndarray:
    """Unpack an NCHW tensor using PyTorch-compatible PixelShuffle ordering."""
    array = np.asarray(phases)
    if array.ndim == 3:
        array = array[np.newaxis]
    if array.ndim != 4 or array.shape[0] != 1:
        raise ValueError(f"expected a single NCHW tensor, got shape {array.shape}")
    _, packed_channels, height, width = array.shape
    divisor = factor * factor
    if packed_channels % divisor:
        raise ValueError("channel count must be divisible by factor squared")
    channels = packed_channels // divisor
    unpacked = array.reshape(1, channels, factor, factor, height, width).transpose(0, 1, 4, 2, 5, 3)
    return np.transpose(
        unpacked.reshape(1, channels, height * factor, width * factor)[0],
        (1, 2, 0),
    )


def load_fp32_predictor(
    weight: Path,
    *,
    scale: int,
    num_channels: int,
    num_blocks: int,
    device: str,
    phase_factor: int = 2,
    codec_feature_channels: int = 96,
    codec_project_channels: int = 16,
    codec_upsample_factor: int = 4,
):
    import torch

    from rknn_super_resolution.models import PhaseRLFNSR

    dev = torch.device(device)
    model = PhaseRLFNSR(
        scale=scale,
        num_channels=num_channels,
        num_blocks=num_blocks,
        phase_factor=phase_factor,
        codec_feature_channels=codec_feature_channels,
        codec_project_channels=codec_project_channels,
        codec_upsample_factor=codec_upsample_factor,
    ).to(dev)
    raw = torch.load(weight, map_location=dev, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise TypeError(f"Unsupported checkpoint format in {weight}")
    model.load_state_dict(state_dict)
    model.switch_to_deploy()
    model.eval()

    def predict(lr_rgb: np.ndarray) -> np.ndarray:
        from rknn_super_resolution.data.yuv_utils import rgb_to_yuv444, yuv444_to_rgb

        lr = torch.from_numpy(lr_rgb).permute(2, 0, 1).unsqueeze(0).to(dev).float()
        lr = rgb_to_yuv444(lr)
        with torch.no_grad():
            sr = torch.clamp(model(lr), 0.0, 255.0)
        return yuv444_to_rgb(sr).squeeze(0).permute(1, 2, 0).cpu().numpy()

    return predict


def evaluate_accuracy(
    runtime: _RknnRuntime,
    pairs: list[ImagePair],
    *,
    fp32_predictor,
    quant_mode: str,
    phase_factor: int = 2,
    scale: int = 3,
    codec_feature_channels: int = 0,
) -> AccuracyReport:
    if not pairs:
        raise ValueError("No validation images found for RKNN accuracy eval.")

    input_h, input_w = pairs[0].lr_rgb.shape[:2]
    fp32_psnrs: list[float] = []
    fp32_ssims: list[float] = []
    rknn_psnrs: list[float] = []
    rknn_ssims: list[float] = []
    match_psnrs: list[float] = []

    for pair in pairs:
        sr_rknn = infer_rknn_rgb(
            runtime,
            pair.lr_rgb,
            phase_factor=phase_factor,
            scale=scale,
            codec_feature_channels=codec_feature_channels,
        )
        rknn_psnrs.append(psnr_numpy(sr_rknn, pair.hr_rgb))
        rknn_ssims.append(ssim_numpy(sr_rknn, pair.hr_rgb))

        if fp32_predictor is not None:
            sr_fp32 = fp32_predictor(pair.lr_rgb)
            fp32_psnrs.append(psnr_numpy(sr_fp32, pair.hr_rgb))
            fp32_ssims.append(ssim_numpy(sr_fp32, pair.hr_rgb))
            match_psnrs.append(psnr_numpy(sr_rknn, sr_fp32))

    fp32_row = _aggregate("FP32 (PyTorch)", fp32_psnrs, fp32_ssims) if fp32_predictor else None
    rknn_row = _aggregate(f"RKNN ({quant_mode})", rknn_psnrs, rknn_ssims)
    match_arr = np.asarray(match_psnrs, dtype=np.float64)

    return AccuracyReport(
        num_images=len(pairs),
        input_h=input_h,
        input_w=input_w,
        fp32=fp32_row,
        rknn=rknn_row,
        match_psnr=float(match_arr.mean()) if match_psnrs else float("nan"),
        match_psnr_min=float(match_arr.min()) if match_psnrs else float("nan"),
        quant_mode=quant_mode,
        scale=scale,
    )


def format_accuracy_table(report: AccuracyReport) -> str:
    hr_h = report.input_h * report.scale
    hr_w = report.input_w * report.scale
    header = (
        f"=== RKNN accuracy ({report.num_images} images, "
        f"LR {report.input_h}x{report.input_w} -> HR {hr_h}x{hr_w}) ==="
    )

    def fmt(v: float) -> str:
        if np.isinf(v):
            return "inf"
        return f"{v:.3f}"

    col_w = 18
    lines = [
        header,
        "",
        f"{'Metric':<28} {'FP32 (PyTorch)':>{col_w}} {report.rknn.label:>{col_w}} {'Delta':>{col_w}}",
        "-" * (28 + col_w * 3 + 2),
    ]

    if report.fp32 is None:
        lines.extend(
            [
                f"{'PSNR vs HR (dB)':<28} {'—':>{col_w}} {fmt(report.rknn.psnr_vs_hr):>{col_w}} {'—':>{col_w}}",
                f"{'SSIM vs HR':<28} {'—':>{col_w}} {fmt(report.rknn.ssim_vs_hr):>{col_w}} {'—':>{col_w}}",
                "",
                "FP32 column skipped: install PyTorch and pass --weight for full comparison.",
            ]
        )
        return "\n".join(lines)

    drop_psnr = report.psnr_drop or 0.0
    drop_ssim = report.ssim_drop or 0.0
    lines.extend(
        [
            f"{'PSNR vs HR (dB)':<28} {fmt(report.fp32.psnr_vs_hr):>{col_w}} "
            f"{fmt(report.rknn.psnr_vs_hr):>{col_w}} {drop_psnr:+.3f}",
            f"{'SSIM vs HR':<28} {fmt(report.fp32.ssim_vs_hr):>{col_w}} "
            f"{fmt(report.rknn.ssim_vs_hr):>{col_w}} {drop_ssim:+.4f}",
            f"{'PSNR range (min..max)':<28} {report.fp32.psnr_min:.2f}..{report.fp32.psnr_max:.2f}",
            f"{'':>{col_w}} {report.rknn.psnr_min:.2f}..{report.rknn.psnr_max:.2f}",
            "",
            f"{'FP32 vs RKNN output PSNR (dB)':<28} {fmt(report.match_psnr):>18} "
            f"(worst image {report.match_psnr_min:.2f} dB)",
        ]
    )
    return "\n".join(lines)


def add_eval_args(parser: argparse.ArgumentParser, model_config: ModelConfig) -> None:
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run SR-fallback FP32 vs RKNN PSNR/SSIM comparison after build.",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default=None,
        help="PyTorch checkpoint for FP32 reference (recommended for eval table).",
    )
    parser.add_argument("--hr_dir", type=str, default="data/DIV2K_valid_HR")
    parser.add_argument("--lr_dir", type=str, default="data/DIV2K_valid_LR_bicubic/X3")
    parser.add_argument("--scale", type=int, default=model_config.scale)
    parser.add_argument("--num_channels", type=int, default=model_config.num_channels)
    parser.add_argument("--num_blocks", type=int, default=model_config.num_blocks)
    parser.add_argument("--phase_factor", type=int, default=model_config.phase_factor)
    parser.add_argument(
        "--codec_feature_channels", type=int, default=model_config.codec_feature_channels
    )
    parser.add_argument(
        "--codec_project_channels", type=int, default=model_config.codec_project_channels
    )
    parser.add_argument(
        "--codec_upsample_factor", type=int, default=model_config.codec_upsample_factor
    )
    parser.add_argument("--eval_device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max_images", type=int, default=100)


def run_post_build_eval(
    args: argparse.Namespace, runtime: _RknnRuntime, *, quant_mode: str
) -> None:
    hr_dir = Path(args.hr_dir)
    lr_dir = Path(args.lr_dir) if args.lr_dir else None
    if not hr_dir.is_dir():
        print(f"--> Eval skipped: HR directory not found: {hr_dir}")
        return

    size_specs = [[int(x.strip()) for x in item.split(",")] for item in args.input_size.split(";")]
    _input_channels, input_h, input_w = size_specs[0]
    input_h *= args.phase_factor
    input_w *= args.phase_factor
    runtime_codec_channels = size_specs[1][0] if len(size_specs) > 1 else 0

    fp32_predictor = None
    if args.weight:
        try:
            fp32_predictor = load_fp32_predictor(
                Path(args.weight),
                scale=args.scale,
                num_channels=args.num_channels,
                num_blocks=args.num_blocks,
                phase_factor=args.phase_factor,
                codec_feature_channels=args.codec_feature_channels,
                codec_project_channels=args.codec_project_channels,
                codec_upsample_factor=args.codec_upsample_factor,
                device=args.eval_device,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"--> Eval warning: FP32 reference unavailable ({exc})")
    else:
        print("--> Eval warning: --weight not set; RKNN-only metrics vs HR will be shown.")

    pairs = collect_image_pairs(
        hr_dir,
        lr_dir,
        scale=args.scale,
        input_h=input_h,
        input_w=input_w,
        max_images=args.max_images,
    )
    report = evaluate_accuracy(
        runtime,
        pairs,
        fp32_predictor=fp32_predictor,
        quant_mode=quant_mode,
        phase_factor=args.phase_factor,
        scale=args.scale,
        codec_feature_channels=runtime_codec_channels,
    )
    print()
    print(format_accuracy_table(report))
