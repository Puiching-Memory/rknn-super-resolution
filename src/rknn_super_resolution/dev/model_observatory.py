"""Generate statistical and visual evidence from checkpoints and saved tensors."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rknn_super_resolution.utils.model_observatory import (
    TensorObservatory,
    analyze_state_dict,
    render_channel_atlas,
    render_checkpoint_overview,
    render_frequency_atlas,
    render_quantization_risk,
    render_spectra,
    render_tensor_observatory,
    save_png,
    summarize_transition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--state_key",
        type=str,
        default=None,
        help="checkpoint mapping containing tensors (auto-detects state_dict/ema_state_dict)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="default: CHECKPOINT.parent/model_observatory/CHECKPOINT.stem",
    )
    parser.add_argument(
        "--clip_abs",
        type=float,
        default=1.0,
        help="simulate symmetric weight clipping to [-clip_abs, clip_abs]",
    )
    parser.add_argument("--max_layers", type=int, default=96)
    parser.add_argument(
        "--tensor_npz",
        type=Path,
        default=None,
        help="optional named activation/state arrays captured from a real forward pass",
    )
    parser.add_argument(
        "--previous_npz",
        type=Path,
        default=None,
        help="optional previous-frame arrays; matching names receive transition statistics",
    )
    return parser.parse_args()


def _tensor_mapping(raw: Any, state_key: str | None) -> tuple[str, dict[str, torch.Tensor]]:
    if not isinstance(raw, dict):
        raise TypeError("checkpoint must be a mapping")
    candidates = [state_key] if state_key else ["state_dict", "ema_state_dict"]
    candidates.append("<root>")
    for candidate in candidates:
        value = raw if candidate == "<root>" else raw.get(candidate)
        if not isinstance(value, dict) or not value:
            continue
        tensors = {
            str(name): tensor
            for name, tensor in value.items()
            if isinstance(name, str) and isinstance(tensor, torch.Tensor)
        }
        if tensors and len(tensors) == len(value):
            return candidate, tensors
    requested = f" {state_key!r}" if state_key else ""
    raise TypeError(f"could not find an all-tensor state mapping{requested} in checkpoint")


def load_checkpoint_tensors(
    checkpoint: Path,
    *,
    state_key: str | None = None,
) -> tuple[str, dict[str, torch.Tensor], dict[str, Any]]:
    """Load a generic PyTorch checkpoint without reconstructing the model class."""
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    selected_key, tensors = _tensor_mapping(raw, state_key)
    metadata: dict[str, Any] = {}
    if isinstance(raw, dict):
        for key in ("phase", "step", "graph_format", "swanlab_run_id"):
            value = raw.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
    return selected_key, tensors, metadata


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _checkpoint_report(
    checkpoint: Path,
    output_dir: Path,
    *,
    state_key: str | None,
    clip_abs: float,
    max_layers: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_key, tensors, metadata = load_checkpoint_tensors(checkpoint, state_key=state_key)
    observations, spectra, frequencies = analyze_state_dict(tensors, clip_abs=clip_abs)
    summaries = [asdict(item.summary) for item in observations]
    matrices = [item for item in summaries if len(item["shape"]) >= 2]
    ranked = sorted(matrices, key=lambda item: item["clip_relative_rmse"], reverse=True)
    payload = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "selected_state_key": selected_key,
        "metadata": metadata,
        "clip_abs": clip_abs,
        "floating_tensor_count": len(observations),
        "floating_value_count": sum(item["numel"] for item in summaries),
        "matrix_count": len(matrices),
        "risk_summary": {
            "layers_over_1pct_clip_rmse": sum(
                item["clip_relative_rmse"] > 0.01 for item in matrices
            ),
            "layers_over_10pct_clip_rmse": sum(
                item["clip_relative_rmse"] > 0.10 for item in matrices
            ),
            "highest_clip_risk": [
                {
                    "name": item["name"],
                    "clip_relative_rmse": item["clip_relative_rmse"],
                    "clip_ratio": item["clip_ratio"],
                    "abs_max": item["abs_max"],
                }
                for item in ranked[:10]
            ],
        },
        "tensors": summaries,
    }
    paths = {
        "summary": output_dir / "checkpoint_summary.json",
        "overview": output_dir / "checkpoint_overview.png",
        "quantization": output_dir / "quantization_risk.png",
        "spectra": output_dir / "weight_spectra.png",
        "channels": output_dir / "channel_energy_atlas.png",
        "frequency": output_dir / "kernel_frequency_atlas.png",
    }
    _write_json(paths["summary"], payload)
    save_png(
        render_checkpoint_overview(
            observations,
            checkpoint_name=checkpoint.name,
            clip_abs=clip_abs,
            max_layers=max_layers,
        ),
        paths["overview"],
    )
    save_png(
        render_quantization_risk(observations, clip_abs=clip_abs, max_layers=max_layers),
        paths["quantization"],
    )
    save_png(render_spectra(spectra, max_layers=max_layers), paths["spectra"])
    save_png(render_channel_atlas(observations, max_layers=max_layers), paths["channels"])
    save_png(render_frequency_atlas(frequencies, max_layers=max_layers), paths["frequency"])
    return paths


def _load_npz(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: torch.from_numpy(np.array(archive[name], copy=True))
            for name in archive.files
            if np.issubdtype(archive[name].dtype, np.number)
        }


def _activation_report(
    tensor_npz: Path,
    previous_npz: Path | None,
    output_dir: Path,
    *,
    clip_abs: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    current = _load_npz(tensor_npz)
    previous = _load_npz(previous_npz) if previous_npz is not None else {}
    observatory = TensorObservatory(clip_abs=clip_abs)
    for name, tensor in current.items():
        channel_dim = 1 if tensor.ndim >= 2 else 0 if tensor.ndim else None
        observatory.observe(name, tensor, channel_dim=channel_dim)
        if name in previous and previous[name].shape == tensor.shape:
            observatory.transitions[name] = summarize_transition(name, previous[name], tensor)
    paths = {
        "activation_summary": output_dir / "activation_summary.json",
        "activation_panel": output_dir / "activation_observatory.png",
    }
    _write_json(
        paths["activation_summary"],
        {
            "schema_version": 1,
            "tensor_npz": str(tensor_npz.resolve()),
            "previous_npz": str(previous_npz.resolve()) if previous_npz else None,
            **observatory.to_dict(),
        },
    )
    save_png(render_tensor_observatory(observatory), paths["activation_panel"])
    return paths


def main() -> None:
    args = parse_args()
    if args.clip_abs <= 0:
        raise ValueError("clip_abs must be positive")
    if args.max_layers <= 0:
        raise ValueError("max_layers must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.previous_npz is not None and args.tensor_npz is None:
        raise ValueError("--previous_npz requires --tensor_npz")
    output_dir = args.output_dir or (
        args.checkpoint.parent / "model_observatory" / args.checkpoint.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _checkpoint_report(
        args.checkpoint,
        output_dir,
        state_key=args.state_key,
        clip_abs=args.clip_abs,
        max_layers=args.max_layers,
    )
    if args.tensor_npz is not None:
        paths.update(
            _activation_report(
                args.tensor_npz,
                args.previous_npz,
                output_dir,
                clip_abs=args.clip_abs,
            )
        )
    print(f"model observatory report: {output_dir.resolve()}")
    for name, path in paths.items():
        print(f"  {name}: {path.name}")


if __name__ == "__main__":
    main()
