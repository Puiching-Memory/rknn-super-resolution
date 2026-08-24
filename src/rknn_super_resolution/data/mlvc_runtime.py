"""Frozen MLVC-S no-bitstream reconstruction runtime."""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class MLVCReconstruction:
    """Reconstructed P-frames and their decoder-side recurrent features."""

    frames: torch.Tensor
    features: torch.Tensor


def mlvc_model_config(variant: str) -> dict[str, Any]:
    """Return the public MLVC/MLVC-S DMC-6.1sb model configuration."""
    common: dict[str, Any] = {
        "type": "DMC-6.1sb",
        "activation": "LeakyReLU",
        "input_offset": -0.5,
        "memory_activation": "identity",
        "zero_init_residual": True,
        "chunk_mode": "gated",
        "ffn_gate_activation": "ReLU1",
        "chain_feature_adaptors": True,
    }
    if variant == "full":
        return {
            **common,
            "feature_channels": 128,
            "spatial_prior_channels": 256,
        }
    if variant == "small":
        return {
            **common,
            "feature_channels": 48,
            "spatial_prior_channels": 128,
            "recon_channels": 192,
            "hidden_channels": 192,
            "hyperprior_num_blocks": 2,
            "y_scale_repeat": 4,
            "z_channels": 48,
            "y_channels": 48,
            "hyperprior_variant": "mini",
            "feature_extractor_num_conv1_layers": 1,
            "feature_extractor_num_conv2_layers": 1,
        }
    raise ValueError(f"unsupported MLVC variant {variant!r}; expected 'small' or 'full'")


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    if isinstance(raw, dict) and "net" in raw:
        raw = raw["net"]
    if not isinstance(raw, dict):
        raise TypeError(f"unsupported MLVC checkpoint format: {path}")
    state: dict[str, torch.Tensor] = {}
    for name, value in raw.items():
        if not isinstance(value, torch.Tensor):
            continue
        state[name.removeprefix("module.")] = value
    return state


def _import_mlvc_factory(repo: Path):
    video_root = repo / "video"
    package_root = video_root / "src"
    if not package_root.is_dir():
        raise FileNotFoundError(
            f"MLVC source not found at {package_root}; clone submodules or run ./scripts/setup_mlvc.sh"
        )
    existing = sys.modules.get("src")
    if existing is not None:
        roots = {Path(item).resolve() for item in getattr(existing, "__path__", ())}
        if package_root.resolve() not in roots:
            raise RuntimeError("top-level Python package 'src' is already owned by another project")
    video_str = str(video_root.resolve())
    if video_str not in sys.path:
        sys.path.insert(0, video_str)
    return importlib.import_module("src.utils.model_factory").create_video_model


class FrozenMLVCRuntime:
    """Execute MLVC analysis, true rounding and synthesis without rANS."""

    def __init__(
        self,
        *,
        repo: str | Path,
        checkpoint: str | Path,
        variant: str,
        device: torch.device,
        amp: bool,
    ) -> None:
        repo_path = Path(repo).resolve()
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"MLVC checkpoint not found at {checkpoint_path}; run ./scripts/setup_mlvc.sh"
            )
        create_video_model = _import_mlvc_factory(repo_path)
        model = create_video_model(mlvc_model_config(variant))
        model.load_state_dict(_checkpoint_state(checkpoint_path), strict=True)
        model.to(device).eval().requires_grad_(False)

        self.model = model
        self.device = device
        self.amp = amp and device.type == "cuda"
        self._lock = Lock()

    def reconstruct(self, sequence: torch.Tensor, q_index: torch.Tensor) -> MLVCReconstruction:
        """Return reconstructed P-frames and decoder features.

        Frame 0 is the uncompressed DPB reference and is not returned. The output
        layout is ``B x (T-1) x 3 x H x W`` in ``[0, 1]``.
        """
        if sequence.ndim != 5 or sequence.shape[1] < 2 or sequence.shape[2] != 3:
            raise ValueError("MLVC input must be BxTx3xHxW with T >= 2")
        if q_index.ndim != 1 or q_index.shape[0] != sequence.shape[0]:
            raise ValueError("q_index must contain one value per sequence")
        if int(q_index.min()) < 0 or int(q_index.max()) >= 64:
            raise ValueError("MLVC q_index must be in [0, 63]")

        height, width = sequence.shape[-2:]
        alignment = int(getattr(self.model, "padding_size", 1))
        pad_h = (-height) % alignment
        pad_w = (-width) % alignment
        if pad_h or pad_w:
            flat = sequence.flatten(0, 1)
            sequence = F.pad(flat, (0, pad_w, 0, pad_h), mode="replicate").unflatten(
                0, (sequence.shape[0], sequence.shape[1])
            )

        device_context = (
            torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext()
        )
        with self._lock, device_context, torch.inference_mode():
            autocast = (
                torch.amp.autocast("cuda", dtype=torch.float16) if self.amp else nullcontext()
            )
            with autocast:
                dpb: dict[str, torch.Tensor | None] = {
                    "ref_frame": sequence[:, 0],
                    "ref_feature": None,
                }
                frames: list[torch.Tensor] = []
                features: list[torch.Tensor] = []
                frame_map = tuple(getattr(self.model, "frame_index_map", (0, 1, 0, 2, 0, 2, 0, 2)))
                for frame_index in range(1, sequence.shape[1]):
                    result = self.model.compress_core(
                        sequence[:, frame_index],
                        dpb,
                        q_index=q_index,
                        fa_idx=frame_map[frame_index % len(frame_map)],
                    )
                    dpb = result["dpb"]
                    ref = dpb["ref_frame"]
                    feature = dpb["ref_feature"]
                    assert ref is not None
                    assert feature is not None
                    frames.append(ref[..., :height, :width])
                    features.append(feature)
        output = torch.stack(frames, dim=1)
        feature_output = torch.stack(features, dim=1)
        return MLVCReconstruction(
            frames=torch.nan_to_num(output.float(), nan=0.0, posinf=1.0).clamp_(0.0, 1.0),
            features=torch.nan_to_num(feature_output.float(), nan=0.0),
        )
