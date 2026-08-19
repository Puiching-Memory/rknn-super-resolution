"""Netflix VMAF v1 scoring via the local ``vmaf`` CLI (libvmaf).

Models follow https://github.com/Netflix/vmaf/blob/master/resource/doc/models_v1.md
Default for this project: standard 1080p / 3H (``vmaf_v1.0.16_3d0h``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rk3588_mobile_sr.data.yuv_utils import rgb_to_yuv444

# Alias -> built-in libvmaf version name (preferred) or on-disk JSON under third_party.
VMAF_MODEL_ALIASES: dict[str, str] = {
    "phone": "vmaf_v1.0.16_5d0h",
    "1080p": "vmaf_v1.0.16_3d0h",
    "4k": "vmaf_v1.0.16_1d5h_2160",
    "4k_3h": "vmaf_v1.0.16_3d0h_2160",
    "phone_hfr": "vmaf_v1.0.16_hfr_5d0h",
    "1080p_hfr": "vmaf_v1.0.16_hfr_3d0h",
}

DEFAULT_VMAF_MODEL = "1080p"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_vmaf_runtime_env() -> None:
    """Prepend project-local libvmaf install to PATH / LD_LIBRARY_PATH."""
    prefix = _repo_root() / ".local"
    bin_dir = prefix / "bin"
    lib_dir = prefix / "lib" / "x86_64-linux-gnu"
    if bin_dir.is_dir():
        path = os.environ.get("PATH", "")
        prefix_s = str(bin_dir)
        if prefix_s not in path.split(":"):
            os.environ["PATH"] = f"{prefix_s}:{path}" if path else prefix_s
    if lib_dir.is_dir():
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        lib_s = str(lib_dir)
        if lib_s not in ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{lib_s}:{ld}" if ld else lib_s


def resolve_vmaf_binary() -> str:
    ensure_vmaf_runtime_env()
    binary = shutil.which("vmaf")
    if binary is None:
        raise FileNotFoundError(
            "vmaf CLI not found. Build Netflix libvmaf first:\n"
            "  ./scripts/setup_vmaf.sh"
        )
    return binary


def resolve_vmaf_model_arg(model: str = DEFAULT_VMAF_MODEL) -> str:
    """Return a ``vmaf --model`` argument (``version=...`` or ``path=...``)."""
    key = model.strip()
    if key.startswith("path=") or key.startswith("version="):
        return key
    alias = VMAF_MODEL_ALIASES.get(key.lower(), key)
    # Prefer built-in compiled models from libvmaf 3.2+ / VMAF v1.
    return f"version={alias}"


def rgb_chw_to_yuv420_bytes(rgb: torch.Tensor) -> bytes:
    """CHW RGB float/uint in [0, 255] -> planar YUV420p 8-bit bytes."""
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"expected CHW RGB, got {tuple(rgb.shape)}")
    yuv = rgb_to_yuv444(rgb.unsqueeze(0).float()).squeeze(0)
    y = yuv[0].round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    u = yuv[1].unsqueeze(0).unsqueeze(0)
    v = yuv[2].unsqueeze(0).unsqueeze(0)
    u_ds = F.avg_pool2d(u, kernel_size=2, stride=2).round().clamp(0, 255).to(torch.uint8)
    v_ds = F.avg_pool2d(v, kernel_size=2, stride=2).round().clamp(0, 255).to(torch.uint8)
    return (
        y.tobytes()
        + u_ds.cpu().numpy().tobytes()
        + v_ds.cpu().numpy().tobytes()
    )


def _parse_vmaf_json(payload: dict) -> tuple[list[float], float]:
    frames = payload.get("frames") or []
    per_frame: list[float] = []
    for frame in frames:
        metrics = frame.get("metrics") or {}
        if "vmaf" not in metrics:
            raise KeyError(f"VMAF JSON frame missing 'vmaf' metric: {list(metrics)[:8]}")
        per_frame.append(float(metrics["vmaf"]))
    pooled = (payload.get("pooled_metrics") or {}).get("vmaf") or {}
    mean = float(pooled["mean"]) if "mean" in pooled else float(np.mean(per_frame))
    return per_frame, mean


def run_vmaf_yuv_files(
    *,
    reference_yuv: Path,
    distorted_yuv: Path,
    width: int,
    height: int,
    model: str = DEFAULT_VMAF_MODEL,
    threads: int = 0,
    enc_width: int | None = None,
    enc_height: int | None = None,
    enc_bitdepth: int = 8,
    bitdepth: int = 8,
) -> tuple[list[float], float]:
    """Score planar YUV420 files; return (per-frame scores, pooled mean)."""
    binary = resolve_vmaf_binary()
    model_arg = resolve_vmaf_model_arg(model)
    if enc_width is not None and enc_height is not None:
        model_arg = (
            f"{model_arg}:cambi.enc_width={enc_width}"
            f":cambi.enc_height={enc_height}:cambi.enc_bitdepth={enc_bitdepth}"
        )
    with tempfile.TemporaryDirectory(prefix="vmaf_") as tmp:
        out_json = Path(tmp) / "out.json"
        cmd = [
            binary,
            "--reference",
            str(reference_yuv),
            "--distorted",
            str(distorted_yuv),
            "--width",
            str(width),
            "--height",
            str(height),
            "--pixel_format",
            "420",
            "--bitdepth",
            str(bitdepth),
            "--model",
            model_arg,
            "--json",
            "--output",
            str(out_json),
            "--quiet",
        ]
        if threads > 0:
            cmd.extend(["--threads", str(threads)])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out_json.is_file():
            raise RuntimeError(
                f"vmaf failed (rc={proc.returncode}): {proc.stderr[-800:] or proc.stdout[-800:]}"
            )
        payload = json.loads(out_json.read_text(encoding="utf-8"))
    return _parse_vmaf_json(payload)


@torch.no_grad()
def batch_vmaf(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    model: str = DEFAULT_VMAF_MODEL,
    threads: int = 0,
    enc_width: int | None = None,
    enc_height: int | None = None,
) -> torch.Tensor:
    """Per-sample VMAF for BCHW RGB in [0, 255]. Higher is better."""
    if pred_rgb.shape != target_rgb.shape:
        raise ValueError(f"shape mismatch {tuple(pred_rgb.shape)} vs {tuple(target_rgb.shape)}")
    if pred_rgb.ndim != 4:
        raise ValueError(f"expected BCHW, got {tuple(pred_rgb.shape)}")
    batch, _, height, width = pred_rgb.shape
    if height % 2 or width % 2:
        raise ValueError(f"VMAF YUV420 requires even HxW, got {height}x{width}")

    with tempfile.TemporaryDirectory(prefix="vmaf_batch_") as tmp:
        tmp_path = Path(tmp)
        ref_path = tmp_path / "ref.yuv"
        dis_path = tmp_path / "dis.yuv"
        with ref_path.open("wb") as ref_f, dis_path.open("wb") as dis_f:
            for i in range(batch):
                ref_f.write(rgb_chw_to_yuv420_bytes(target_rgb[i]))
                dis_f.write(rgb_chw_to_yuv420_bytes(pred_rgb[i]))
        per_frame, _ = run_vmaf_yuv_files(
            reference_yuv=ref_path,
            distorted_yuv=dis_path,
            width=width,
            height=height,
            model=model,
            threads=threads,
            enc_width=enc_width,
            enc_height=enc_height,
        )
    if len(per_frame) != batch:
        raise RuntimeError(f"VMAF returned {len(per_frame)} frames, expected {batch}")
    return torch.tensor(per_frame, dtype=torch.float32, device=pred_rgb.device)
