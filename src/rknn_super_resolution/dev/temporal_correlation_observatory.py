"""Observe sparse motion-correlation coverage on real MLVC reconstructions."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from rknn_super_resolution.config import load_config
from rknn_super_resolution.data.mlvc_loader import (
    build_mlvc_evaluation_loader,
    rgb_to_mlvc_ycbcr,
)
from rknn_super_resolution.dev.cst_oracle import translate

ROOT = Path(__file__).resolve().parents[3]

BANKS = {
    "k5_integer_cross": ((0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)),
    "k13_half_cross": (
        (0.0, 0.0),
        (-0.5, 0.0),
        (0.5, 0.0),
        (0.0, -0.5),
        (0.0, 0.5),
        (-0.5, -0.5),
        (-0.5, 0.5),
        (0.5, -0.5),
        (0.5, 0.5),
        (-1.0, 0.0),
        (1.0, 0.0),
        (0.0, -1.0),
        (0.0, 1.0),
    ),
    "k25_half_grid": tuple(
        (dx, dy)
        for dx in (-1.0, -0.5, 0.0, 0.5, 1.0)
        for dy in (-1.0, -0.5, 0.0, 0.5, 1.0)
    ),
}


@dataclass(frozen=True)
class ObservatoryConfig:
    output_dir: Path
    config: Path | None
    clips: int
    q_indices: tuple[int, ...]
    block_size: int
    device: str


def parse_args() -> ObservatoryConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--clips", type=int, default=4)
    parser.add_argument("--q_indices", type=int, nargs="+", default=(21, 42))
    parser.add_argument("--block_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    ns = parser.parse_args()
    return ObservatoryConfig(
        output_dir=ns.output_dir,
        config=ns.config,
        clips=ns.clips,
        q_indices=tuple(ns.q_indices),
        block_size=ns.block_size,
        device=ns.device,
    )


def analyze_bank(
    current: torch.Tensor,
    history: torch.Tensor,
    offsets: tuple[tuple[float, float], ...],
    *,
    block_size: int,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Return correlation statistics and first-frame diagnostic maps."""
    batch, frames = history.shape[:2]
    flat_history = history.flatten(0, 1)
    reference = current[:, None].expand(-1, frames, -1, -1, -1).flatten(0, 1)
    errors = []
    for dx, dy in offsets:
        shift = torch.tensor((dx, dy), device=current.device)[None].expand(
            batch * frames, -1
        )
        aligned = translate(flat_history, shift)
        squared = (aligned - reference).square().mean(dim=1, keepdim=True)
        errors.append(F.avg_pool2d(squared, block_size, stride=block_size))
    error = torch.cat(errors, dim=1)
    order = error.argsort(dim=1)
    best = error.gather(1, order[:, :1])
    second = error.gather(1, order[:, 1:2])
    center_index = offsets.index((0.0, 0.0))
    center = error[:, center_index : center_index + 1]
    selection = order[:, 0]
    histogram = torch.bincount(selection.flatten(), minlength=len(offsets)).float()
    probabilities = histogram / histogram.sum().clamp_min(1.0)
    entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
    normalized_entropy = entropy / math.log(len(offsets)) if len(offsets) > 1 else entropy
    offset_tensor = torch.tensor(offsets, device=current.device)
    chosen_offsets = offset_tensor[selection]
    improvement = (center - best) / center.clamp_min(1e-8)
    confidence = (second - best) / (second + best).clamp_min(1e-8)
    metrics = {
        "relative_error_reduction_mean": float(improvement.mean().item()),
        "relative_error_reduction_p50": float(improvement.median().item()),
        "confidence_margin_mean": float(confidence.mean().item()),
        "noncenter_ratio": float((selection != center_index).float().mean().item()),
        "selection_entropy": float(normalized_entropy.item()),
        "chosen_offset_l1_mean": float(chosen_offsets.abs().sum(dim=-1).mean().item()),
        "best_mse": float(best.mean().item()),
        "center_mse": float(center.mean().item()),
    }
    maps = {
        "selection": selection[0].float(),
        "confidence": confidence[0, 0],
        "improvement": improvement[0, 0],
    }
    return metrics, maps


def _color_map(values: torch.Tensor, *, categorical: bool) -> Image.Image:
    array = values.detach().float().cpu().numpy()
    if categorical:
        normalized = (array % 12) / 11.0
    else:
        low, high = np.quantile(array, [0.02, 0.98])
        normalized = np.clip((array - low) / max(float(high - low), 1e-8), 0.0, 1.0)
    red = np.clip(1.8 * normalized, 0.0, 1.0)
    green = np.clip(1.8 * (1.0 - np.abs(normalized - 0.5) * 2.0), 0.0, 1.0)
    blue = np.clip(1.8 * (1.0 - normalized), 0.0, 1.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def save_maps(path: Path, maps: dict[str, dict[str, torch.Tensor]]) -> None:
    tile_width, tile_height = 320, 180
    label_height = 24
    panel = Image.new("RGB", (3 * tile_width, len(maps) * (tile_height + label_height)), (15, 23, 42))
    draw = ImageDraw.Draw(panel)
    for row, (bank, bank_maps) in enumerate(maps.items()):
        for column, name in enumerate(("selection", "confidence", "improvement")):
            x = column * tile_width
            y = row * (tile_height + label_height)
            draw.text((x + 4, y + 4), f"{bank} / {name}", fill=(226, 232, 240))
            image = _color_map(bank_maps[name], categorical=name == "selection")
            panel.paste(image.resize((tile_width, tile_height), Image.Resampling.NEAREST), (x, y + label_height))
    panel.save(path)


def _describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def run(config: ObservatoryConfig) -> dict:
    if config.clips < 1 or config.block_size < 1:
        raise ValueError("clips and block_size must be positive")
    device = torch.device(config.device)
    app = load_config(config.config)
    app.data.val_samples = config.clips
    app.data.q_indices = config.q_indices
    app.data.num_workers = 0
    loader = build_mlvc_evaluation_loader(
        app.data,
        split="validation",
        device=device,
        scale=app.model.scale,
        colorspace="yuv",
        batch_size=1,
        rank=0,
        world_size=1,
        project_root=ROOT,
    )
    rows: list[dict] = []
    first_maps: dict[str, dict[str, torch.Tensor]] = {}
    try:
        for sample_index, raw_batch in enumerate(loader.loader):
            sequence_u8, _hr_u8 = loader.processor.decoder.decode_batch(raw_batch)
            sequence = sequence_u8.to(device).float().div_(255.0)
            batch, time, channels, height, width = sequence.shape
            yuv = rgb_to_mlvc_ycbcr(sequence.reshape(batch * time, channels, height, width))
            yuv = yuv.reshape(batch, time, channels, height, width)
            q_index = raw_batch["q_index"].to(device)
            reconstruction = loader.processor.runtime.reconstruct(yuv, q_index).frames
            luma = reconstruction[:, :, :1]
            current, history = luma[:, -1], luma[:, :-1]
            bank_rows = {}
            for bank, offsets in BANKS.items():
                metrics, maps = analyze_bank(
                    current,
                    history,
                    offsets,
                    block_size=config.block_size,
                )
                bank_rows[bank] = metrics
                if sample_index == 0:
                    first_maps[bank] = maps
            row = {
                "sample": sample_index,
                "q_index": int(q_index[0].item()),
                "source": str(raw_batch["source"][0]),
                "banks": bank_rows,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        loader.close()

    aggregate = {
        bank: {
            metric: _describe([float(row["banks"][bank][metric]) for row in rows])
            for metric in rows[0]["banks"][bank]
        }
        for bank in BANKS
    }
    result = {
        "schema_version": 1,
        "status": "completed",
        "evidence": "real frozen-MLVC validation reconstructions",
        "config": {**asdict(config), "output_dir": str(config.output_dir), "config": str(config.config) if config.config else None},
        "samples": rows,
        "aggregate": aggregate,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    save_maps(config.output_dir / "correlation_maps.png", first_maps)
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
