"""Direct inverse-problem oracle for information in synthetic VSR observations.

This deliberately bypasses the neural transport implementation.  It asks whether
the known multi-frame image-formation equations contain recoverable information
that is absent from the current frame, using target-aware early stopping as a
generous upper bound rather than as a deployable reconstruction method.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from rknn_super_resolution.dev.cst_oracle import SHIFT_VALUES, generate_batch, translate
from rknn_super_resolution.utils.swanlab_logging import (
    finish_swanlab,
    log_metrics,
    setup_swanlab,
)


@dataclass(frozen=True)
class InformationOracleConfig:
    output_dir: Path
    batches: int
    batch_size: int
    hr_size: int
    history_frames: int
    optimization_steps: int
    learning_rate: float
    tv_weight: float
    crop: int
    seed: int
    device: str
    swanlab_project: str
    swanlab_experiment: str
    disable_swanlab: bool


def parse_args() -> InformationOracleConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hr_size", type=int, default=96)
    parser.add_argument("--history_frames", type=int, default=4)
    parser.add_argument("--optimization_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.04)
    parser.add_argument("--tv_weight", type=float, default=2e-4)
    parser.add_argument("--crop", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--swanlab_project", default="rknn-super-resolution")
    parser.add_argument("--swanlab_experiment", default="cst-a0-information-oracle")
    parser.add_argument("--disable_swanlab", action="store_true")
    ns = parser.parse_args()
    return InformationOracleConfig(**vars(ns))


def _validate_config(config: InformationOracleConfig) -> None:
    if min(config.batches, config.batch_size, config.optimization_steps) < 1:
        raise ValueError("batches, batch_size and optimization_steps must be positive")
    if config.hr_size % 3:
        raise ValueError("hr_size must be divisible by the 3x scale")
    if config.history_frames < 1:
        raise ValueError("history_frames must be positive")
    if config.crop < 0 or config.crop * 2 >= config.hr_size:
        raise ValueError("crop must leave a non-empty image interior")


def clean_downsample(hr: torch.Tensor) -> torch.Tensor:
    """Apply the differentiable, noise-free part of the synthetic observation model."""
    return F.interpolate(
        hr,
        scale_factor=1 / 3,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


def predict_observations(
    estimate: torch.Tensor,
    shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict current and shifted-history LR observations from an HR estimate."""
    current = clean_downsample(estimate)
    history = []
    for index in range(shifts.shape[1]):
        history.append(clean_downsample(translate(estimate, shifts[:, index] * 3.0)))
    return current, torch.stack(history, dim=1)


@torch.no_grad()
def evaluate_shift_retrieval(
    current: torch.Tensor,
    history: torch.Tensor,
    shifts: torch.Tensor,
    *,
    crop: int = 3,
) -> dict[str, float]:
    """Measure whether raw LR correlation identifies the known half-pixel motion."""
    candidates = torch.tensor(
        [(dx, dy) for dx in SHIFT_VALUES for dy in SHIFT_VALUES],
        device=current.device,
        dtype=current.dtype,
    )
    batch, frames = history.shape[:2]
    flat_history = history.flatten(0, 1)
    reference = current[:, None].expand(-1, frames, -1, -1, -1).flatten(0, 1)
    errors = []
    for candidate in candidates:
        candidate_shifts = candidate[None].expand(batch * frames, -1)
        aligned = translate(flat_history, candidate_shifts)
        if crop:
            aligned = aligned[..., crop:-crop, crop:-crop]
            compared = reference[..., crop:-crop, crop:-crop]
        else:
            compared = reference
        errors.append((aligned - compared).square().mean(dim=(1, 2, 3)))
    error_tensor = torch.stack(errors, dim=1)
    order = error_tensor.argsort(dim=1)
    target_offsets = -shifts.flatten(0, 1)
    target_indices = (target_offsets[:, None] == candidates[None]).all(dim=2).float().argmax(dim=1)
    predicted = order[:, 0]
    predicted_offsets = candidates[predicted]
    exact = predicted == target_indices
    top3 = (order[:, :3] == target_indices[:, None]).any(dim=1)
    margin = error_tensor.gather(1, order[:, 1:2]) - error_tensor.gather(1, order[:, :1])
    return {
        "exact_accuracy": float(exact.float().mean().item()),
        "top3_accuracy": float(top3.float().mean().item()),
        "offset_mae": float((predicted_offsets - target_offsets).abs().mean().item()),
        "best_second_margin": float(margin.mean().item()),
    }


def _psnr(prediction: torch.Tensor, target: torch.Tensor, crop: int) -> torch.Tensor:
    if crop:
        prediction = prediction[..., crop:-crop, crop:-crop]
        target = target[..., crop:-crop, crop:-crop]
    mse = (prediction - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def _total_variation(tensor: torch.Tensor) -> torch.Tensor:
    vertical = (tensor[..., 1:, :] - tensor[..., :-1, :]).abs().mean()
    horizontal = (tensor[..., :, 1:] - tensor[..., :, :-1]).abs().mean()
    return vertical + horizontal


def _high_frequency(tensor: torch.Tensor) -> torch.Tensor:
    channels = tensor.shape[1]
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=tensor.device,
        dtype=tensor.dtype,
    )[None, None]
    return F.conv2d(tensor, kernel.expand(channels, 1, -1, -1), padding=1, groups=channels)


def _image(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def save_reconstruction_panel(
    path: Path,
    *,
    target: torch.Tensor,
    current: torch.Tensor,
    spatial: torch.Tensor,
    temporal: torch.Tensor,
) -> None:
    """Save a compact visual audit of one paired oracle reconstruction."""
    bicubic = F.interpolate(
        current[:1], scale_factor=3, mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
    target_image = target[0]
    entries = [
        ("target", target_image),
        ("bicubic", bicubic[0]),
        ("spatial inverse", spatial[0]),
        ("temporal inverse", temporal[0]),
        ("spatial error x4", (spatial[0] - target_image).abs() * 4),
        ("temporal error x4", (temporal[0] - target_image).abs() * 4),
    ]
    tile_width, tile_height = target.shape[-1], target.shape[-2]
    label_height = 24
    panel = Image.new("RGB", (3 * tile_width, 2 * (tile_height + label_height)), (15, 23, 42))
    draw = ImageDraw.Draw(panel)
    for index, (label, tensor) in enumerate(entries):
        row, column = divmod(index, 3)
        x = column * tile_width
        y = row * (tile_height + label_height)
        draw.text((x + 4, y + 4), label, fill=(226, 232, 240))
        panel.paste(_image(tensor), (x, y + label_height))
    panel.save(path)


def reconstruct(
    current: torch.Tensor,
    history: torch.Tensor,
    shifts: torch.Tensor,
    target: torch.Tensor,
    *,
    use_history: bool,
    steps: int,
    learning_rate: float,
    tv_weight: float,
    crop: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Solve the observation equations and retain the target-best iterate.

    Target-aware iterate selection makes this an intentionally optimistic
    information ceiling.  It must not be reported as a realizable model score.
    """
    initial = F.interpolate(
        current,
        scale_factor=3,
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)
    estimate = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([estimate], lr=learning_rate)
    best = initial.detach().clone()
    best_mean = float(_psnr(best, target, crop).mean().item())
    best_step = 0
    final_data_loss = math.inf

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted_current, predicted_history = predict_observations(estimate, shifts)
        data_loss = F.smooth_l1_loss(predicted_current, current, beta=1 / 255)
        if use_history:
            data_loss = data_loss + F.smooth_l1_loss(
                predicted_history,
                history,
                beta=1 / 255,
            )
        loss = data_loss + tv_weight * _total_variation(estimate)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            estimate.clamp_(0.0, 1.0)
            mean_psnr = float(_psnr(estimate, target, crop).mean().item())
            if mean_psnr > best_mean:
                best_mean = mean_psnr
                best_step = step
                best.copy_(estimate)
        final_data_loss = float(data_loss.item())

    initial_psnr = float(_psnr(initial, target, crop).mean().item())
    return best, {
        "initial_psnr": initial_psnr,
        "best_psnr": best_mean,
        "gain_over_bicubic": best_mean - initial_psnr,
        "best_step": float(best_step),
        "final_data_loss": final_data_loss,
    }


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=True).item()) if len(values) > 1 else 0.0,
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(config: InformationOracleConfig) -> dict:
    _validate_config(config)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested device")
    device = torch.device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    setup_swanlab(
        rank=0,
        save_dir=config.output_dir,
        project=config.swanlab_project,
        experiment_name=config.swanlab_experiment,
        config={**asdict(config), "output_dir": str(config.output_dir)},
        disabled=config.disable_swanlab,
        mode="offline",
    )
    generator = torch.Generator(device=config.device).manual_seed(config.seed)
    paired_rows: list[dict[str, float | int]] = []
    start = time.monotonic()
    try:
        for batch_index in range(config.batches):
            current, history, shifts, target = generate_batch(
                config.batch_size,
                config.hr_size,
                config.history_frames,
                device=device,
                generator=generator,
            )
            retrieval = evaluate_shift_retrieval(
                current,
                history,
                shifts,
                crop=max(config.crop // 3, 1),
            )
            spatial_image, spatial = reconstruct(
                current,
                history,
                shifts,
                target,
                use_history=False,
                steps=config.optimization_steps,
                learning_rate=config.learning_rate,
                tv_weight=config.tv_weight,
                crop=config.crop,
            )
            temporal_image, temporal = reconstruct(
                current,
                history,
                shifts,
                target,
                use_history=True,
                steps=config.optimization_steps,
                learning_rate=config.learning_rate,
                tv_weight=config.tv_weight,
                crop=config.crop,
            )
            delta = temporal["best_psnr"] - spatial["best_psnr"]
            spatial_hf_psnr = float(
                _psnr(_high_frequency(spatial_image), _high_frequency(target), config.crop)
                .mean()
                .item()
            )
            temporal_hf_psnr = float(
                _psnr(_high_frequency(temporal_image), _high_frequency(target), config.crop)
                .mean()
                .item()
            )
            row: dict[str, float | int] = {
                "batch": batch_index,
                "bicubic_psnr": spatial["initial_psnr"],
                "spatial_oracle_psnr": spatial["best_psnr"],
                "temporal_oracle_psnr": temporal["best_psnr"],
                "temporal_minus_spatial_psnr": delta,
                "spatial_hf_psnr": spatial_hf_psnr,
                "temporal_hf_psnr": temporal_hf_psnr,
                "temporal_minus_spatial_hf_psnr": temporal_hf_psnr - spatial_hf_psnr,
                "spatial_best_step": int(spatial["best_step"]),
                "temporal_best_step": int(temporal["best_step"]),
                **{f"shift_{key}": value for key, value in retrieval.items()},
            }
            paired_rows.append(row)
            if batch_index == 0:
                save_reconstruction_panel(
                    config.output_dir / "reconstruction_panel.png",
                    target=target,
                    current=current,
                    spatial=spatial_image,
                    temporal=temporal_image,
                )
            log_metrics(
                {
                    "oracle/spatial_psnr": spatial["best_psnr"],
                    "oracle/temporal_psnr": temporal["best_psnr"],
                    "oracle/temporal_minus_spatial_psnr": delta,
                },
                step=batch_index + 1,
            )
            print(json.dumps(row), flush=True)

        deltas = [float(row["temporal_minus_spatial_psnr"]) for row in paired_rows]
        exact_accuracies = [float(row["shift_exact_accuracy"]) for row in paired_rows]
        top3_accuracies = [float(row["shift_top3_accuracy"]) for row in paired_rows]
        offset_maes = [float(row["shift_offset_mae"]) for row in paired_rows]
        result = {
            "schema_version": 1,
            "status": "completed",
            "interpretation": "target-aware inverse-problem information upper bound",
            "config": {**asdict(config), "output_dir": str(config.output_dir)},
            "paired_rows": paired_rows,
            "temporal_minus_spatial_psnr": _summary(deltas),
            "positive_batches": sum(delta > 0 for delta in deltas),
            "shift_retrieval": {
                "exact_accuracy": _summary(exact_accuracies),
                "top3_accuracy": _summary(top3_accuracies),
                "offset_mae": _summary(offset_maes),
            },
            "elapsed_seconds": time.monotonic() - start,
        }
        _atomic_json(config.output_dir / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        finish_swanlab()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
