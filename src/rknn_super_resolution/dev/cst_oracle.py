"""Stage A0 oracle probe for the information ceiling of coarse temporal transport."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rknn_super_resolution.models import ResidualLocalBlock
from rknn_super_resolution.utils.swanlab_logging import (
    finish_swanlab,
    log_metrics,
    setup_swanlab,
)

type Condition = Literal[
    "spatial",
    "center",
    "uniform",
    "wrong",
    "full",
    "half",
    "coarse",
    "attention",
    "retrieved",
]
type InputLayout = Literal["lr", "phase", "hr", "splat"]

CONDITIONS: tuple[Condition, ...] = (
    "spatial",
    "center",
    "uniform",
    "wrong",
    "full",
    "half",
    "coarse",
    "attention",
    "retrieved",
)
SHIFT_VALUES = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)


@dataclass(frozen=True)
class OracleConfig:
    condition: Condition
    input_layout: InputLayout
    output_dir: Path
    steps: int
    val_every: int
    log_every: int
    batch_size: int
    val_batches: int
    hr_size: int
    history_frames: int
    channels: int
    blocks: int
    learning_rate: float
    seed: int
    device: str
    swanlab_project: str
    swanlab_experiment: str
    swanlab_mode: Literal["online", "offline", "local"]
    disable_swanlab: bool


def parse_args() -> OracleConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument(
        "--input_layout",
        choices=("lr", "phase", "hr", "splat"),
        default="phase",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--val_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batches", type=int, default=16)
    parser.add_argument("--hr_size", type=int, default=96)
    parser.add_argument("--history_frames", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--swanlab_project", type=str, default="rknn-super-resolution")
    parser.add_argument("--swanlab_experiment", type=str, default=None)
    parser.add_argument(
        "--swanlab_mode",
        choices=("online", "offline", "local"),
        default="offline",
    )
    parser.add_argument("--disable_swanlab", action="store_true")
    ns = parser.parse_args()
    experiment = ns.swanlab_experiment or f"cst-a0-{ns.condition}-s{ns.seed}"
    return OracleConfig(
        condition=ns.condition,
        input_layout=ns.input_layout,
        output_dir=ns.output_dir,
        steps=ns.steps,
        val_every=ns.val_every,
        log_every=ns.log_every,
        batch_size=ns.batch_size,
        val_batches=ns.val_batches,
        hr_size=ns.hr_size,
        history_frames=ns.history_frames,
        channels=ns.channels,
        blocks=ns.blocks,
        learning_rate=ns.learning_rate,
        seed=ns.seed,
        device=ns.device,
        swanlab_project=ns.swanlab_project,
        swanlab_experiment=experiment,
        swanlab_mode=ns.swanlab_mode,
        disable_swanlab=ns.disable_swanlab,
    )


def _validate_config(config: OracleConfig) -> None:
    if config.steps < 1 or config.val_every < 1 or config.log_every < 1:
        raise ValueError("steps, val_every and log_every must be positive")
    if config.batch_size < 1 or config.val_batches < 1:
        raise ValueError("batch_size and val_batches must be positive")
    if config.hr_size % 12:
        raise ValueError("hr_size must be divisible by 12 (3x SR and 4x coarse state)")
    if config.history_frames < 1:
        raise ValueError("history_frames must be positive")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _coordinate_grid(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return yy[None, None], xx[None, None]


def generate_texture_batch(
    batch_size: int,
    size: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate repeatable multi-scale textures with strong aliasing content."""
    yy, xx = _coordinate_grid(size, device)
    low = torch.rand(
        batch_size,
        3,
        max(size // 12, 2),
        max(size // 12, 2),
        device=device,
        generator=generator,
    )
    low = F.interpolate(low, size=(size, size), mode="bicubic", align_corners=False)
    medium = torch.rand(
        batch_size,
        3,
        max(size // 4, 2),
        max(size // 4, 2),
        device=device,
        generator=generator,
    )
    medium = F.interpolate(medium, size=(size, size), mode="bicubic", align_corners=False)
    high = torch.rand(batch_size, 3, size, size, device=device, generator=generator)

    frequency = torch.randint(
        3,
        max(size // 3, 4),
        (batch_size, 3, 2),
        device=device,
        generator=generator,
    ).float()
    phase = torch.rand(batch_size, 3, 1, 1, device=device, generator=generator) * (2 * math.pi)
    waves = torch.sin(
        math.pi
        * (
            frequency[..., 0, None, None] * xx
            + frequency[..., 1, None, None] * yy
        )
        + phase
    )
    waves = 0.5 + 0.5 * waves

    texture = 0.25 * low + 0.25 * medium + 0.20 * high + 0.30 * waves
    minimum = texture.amin(dim=(-2, -1), keepdim=True)
    maximum = texture.amax(dim=(-2, -1), keepdim=True)
    return (texture - minimum) / (maximum - minimum).clamp_min(1e-6)


def _sampling_grid(
    batch_size: int,
    height: int,
    width: int,
    shifts: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)[None].expand(batch_size, -1, -1, -1).clone()
    grid[..., 0] += 2.0 * shifts[:, 0, None, None] / max(width - 1, 1)
    grid[..., 1] += 2.0 * shifts[:, 1, None, None] / max(height - 1, 1)
    return grid


def translate(
    tensor: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Return ``output(x,y) = tensor(x+dx,y+dy)`` for pixel-unit shifts."""
    grid = _sampling_grid(
        tensor.shape[0],
        tensor.shape[-2],
        tensor.shape[-1],
        shifts,
        device=tensor.device,
    )
    return F.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )


def _translate_zero(
    tensor: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    grid = _sampling_grid(
        tensor.shape[0],
        tensor.shape[-2],
        tensor.shape[-1],
        shifts,
        device=tensor.device,
    )
    return F.grid_sample(
        tensor,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )


def build_polyphase_observation(
    current: torch.Tensor,
    history: torch.Tensor,
    shifts: torch.Tensor,
    condition: Condition,
) -> torch.Tensor:
    """Pack registered half-LR samples without interpolating their values.

    Samples are first placed on a 2x LR lattice.  The synthetic shifts are
    multiples of half an LR pixel, so exact registration is an integer shift on
    that lattice.  PixelUnshuffle then stores the four sampling phases as
    channels at LR resolution; four additional channels encode observation
    counts so missing phases are distinguishable from black pixels.
    """
    batch, _channels, height, width = current.shape
    sources = [current]
    source_shifts = [torch.zeros(batch, 2, device=current.device, dtype=shifts.dtype)]
    if condition not in {"spatial", "center"}:
        direction = 0.0 if condition == "uniform" else 1.0 if condition == "wrong" else -1.0
        for index in range(history.shape[1]):
            sources.append(history[:, index])
            source_shifts.append(direction * shifts[:, index])

    value_sum = current.new_zeros(batch, 3, height * 2, width * 2)
    count_sum = current.new_zeros(batch, 1, height * 2, width * 2)
    for source, source_shift in zip(sources, source_shifts, strict=True):
        sparse = current.new_zeros(batch, 3, height * 2, width * 2)
        mask = current.new_zeros(batch, 1, height * 2, width * 2)
        sparse[..., ::2, ::2] = source
        mask[..., ::2, ::2] = 1.0
        lattice_shift = source_shift * 2.0
        value_sum = value_sum + _translate_zero(sparse, lattice_shift)
        count_sum = count_sum + _translate_zero(mask, lattice_shift)

    fused = value_sum / count_sum.clamp_min(1.0)
    packed_values = F.pixel_unshuffle(fused, 2)
    packed_counts = F.pixel_unshuffle(count_sum / len(sources), 2)
    return torch.cat((packed_values, packed_counts), dim=1)


@torch.no_grad()
def retrieve_global_shifts(
    current: torch.Tensor,
    history: torch.Tensor,
    *,
    crop: int = 2,
) -> torch.Tensor:
    """Retrieve synthetic global motion using only LR observation correlation."""
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
        alignment = candidate[None].expand(batch * frames, -1)
        aligned = translate(flat_history, alignment)
        if crop:
            aligned = aligned[..., crop:-crop, crop:-crop]
            compared = reference[..., crop:-crop, crop:-crop]
        else:
            compared = reference
        errors.append((aligned - compared).square().mean(dim=(1, 2, 3)))
    best_alignment = candidates[torch.stack(errors, dim=1).argmin(dim=1)]
    return (-best_alignment).unflatten(0, (batch, frames))


def _sample_shifts(
    batch_size: int,
    history_frames: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    values = torch.tensor(SHIFT_VALUES, device=device)
    indices = torch.randint(
        0,
        len(SHIFT_VALUES),
        (batch_size, history_frames, 2),
        device=device,
        generator=generator,
    )
    return values[indices]


def _degrade_lr(
    hr: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    lr = F.interpolate(
        hr,
        scale_factor=1 / 3,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    batch = lr.shape[0]
    quant_step = torch.randint(
        1,
        6,
        (batch, 1, 1, 1),
        device=lr.device,
        generator=generator,
    ).float()
    lr_255 = torch.round(lr * 255.0 / quant_step) * quant_step
    block_noise = torch.randn(
        batch,
        3,
        max(lr.shape[-2] // 8, 1),
        max(lr.shape[-1] // 8, 1),
        device=lr.device,
        generator=generator,
    )
    block_noise = F.interpolate(block_noise, size=lr.shape[-2:], mode="nearest")
    sigma = torch.rand(batch, 1, 1, 1, device=lr.device, generator=generator) * 1.5
    lr_255 = lr_255 + sigma * block_noise
    return (lr_255 / 255.0).clamp(0.0, 1.0)


@torch.no_grad()
def generate_batch(
    batch_size: int,
    hr_size: int,
    history_frames: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate current LR, history LR, known LR shifts and current HR target."""
    target = generate_texture_batch(
        batch_size,
        hr_size,
        device=device,
        generator=generator,
    )
    current = _degrade_lr(target, generator=generator)
    shifts = _sample_shifts(
        batch_size,
        history_frames,
        device=device,
        generator=generator,
    )
    shifted_hr = []
    for index in range(history_frames):
        shifted_hr.append(translate(target, shifts[:, index] * 3.0))
    history_hr = torch.stack(shifted_hr, dim=1)
    flat = history_hr.flatten(0, 1)
    history = _degrade_lr(flat, generator=generator).unflatten(0, (batch_size, history_frames))
    return current, history, shifts, target


class OracleProbe(nn.Module):
    """Equal-capacity SR probe whose only variable is temporal context construction."""

    def __init__(
        self,
        *,
        channels: int = 32,
        blocks: int = 4,
        scale: int = 3,
        input_layout: InputLayout = "lr",
    ) -> None:
        super().__init__()
        self.scale = scale
        self.input_layout = input_layout
        self.phase_factor = 2 if input_layout == "phase" else 1
        self.core_scale = 1 if input_layout == "hr" else scale * self.phase_factor
        input_channels = 16 if input_layout == "splat" else 3 * self.phase_factor**2
        self.encoder = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.temporal_inject = nn.Conv2d(channels, channels, 1, bias=False)
        self.blocks = nn.Sequential(*(ResidualLocalBlock(channels) for _ in range(blocks)))
        head_channels = 3 if input_layout == "hr" else 3 * self.core_scale**2
        self.head = nn.Conv2d(channels, head_channels, 1)
        self.shuffle = nn.Identity() if input_layout == "hr" else nn.PixelShuffle(self.core_scale)
        nn.init.zeros_(self.temporal_inject.weight)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _context(
        self,
        shallow: torch.Tensor,
        history: torch.Tensor,
        shifts: torch.Tensor,
        condition: Condition,
    ) -> torch.Tensor:
        if condition == "spatial":
            return torch.zeros_like(shallow)
        if condition == "center":
            return shallow

        batch, frames, _channels, lr_height, lr_width = history.shape
        flat_history = history.flatten(0, 1)
        if self.input_layout == "phase":
            flat_history = F.pixel_unshuffle(flat_history, self.phase_factor)
        elif self.input_layout == "hr":
            flat_history = F.interpolate(
                flat_history,
                scale_factor=self.scale,
                mode="bicubic",
                align_corners=False,
            )
        encoded = self.encoder(flat_history).unflatten(0, (batch, frames))
        height, width = encoded.shape[-2:]
        if condition == "uniform":
            return encoded.mean(dim=1)

        direction = 1.0 if condition == "wrong" else -1.0
        if self.input_layout == "phase":
            aligned_lr = []
            for index in range(frames):
                aligned_lr.append(
                    translate(history[:, index], direction * shifts[:, index])
                )
            aligned_flat = torch.stack(aligned_lr, dim=1).flatten(0, 1)
            aligned_phases = F.pixel_unshuffle(aligned_flat, self.phase_factor)
            encoded = self.encoder(aligned_phases).unflatten(0, (batch, frames))
            if condition in {"half", "coarse"}:
                factor = 2 if condition == "half" else 4
                context = F.avg_pool2d(encoded.flatten(0, 1), factor).unflatten(
                    0, (batch, frames)
                )
                context = context.mean(dim=1)
                return F.interpolate(context, size=(height, width), mode="nearest")
            if condition == "attention":
                scores = (shallow[:, None] * encoded).mean(dim=2, keepdim=True)
                weights = torch.softmax(scores, dim=1)
                return (weights * encoded).sum(dim=1)
            return encoded.mean(dim=1)

        if self.input_layout == "hr":
            aligned = []
            flat_hr = flat_history.unflatten(0, (batch, frames))
            for index in range(frames):
                aligned.append(
                    translate(
                        flat_hr[:, index],
                        direction * shifts[:, index] * self.scale,
                    )
                )
            aligned_flat = torch.stack(aligned, dim=1).flatten(0, 1)
            encoded = self.encoder(aligned_flat).unflatten(0, (batch, frames))
            if condition in {"half", "coarse"}:
                factor = 2 if condition == "half" else 4
                context = F.avg_pool2d(encoded.flatten(0, 1), factor).unflatten(
                    0, (batch, frames)
                )
                context = context.mean(dim=1)
                return F.interpolate(context, size=(height, width), mode="nearest")
            if condition in {"attention", "wrong"}:
                scores = (shallow[:, None] * encoded).mean(dim=2, keepdim=True)
                weights = torch.softmax(scores, dim=1)
                return (weights * encoded).sum(dim=1)
            return encoded.mean(dim=1)

        if condition in {"half", "coarse"}:
            factor = 2 if condition == "half" else 4
            coarse = F.avg_pool2d(encoded.flatten(0, 1), factor).unflatten(
                0, (batch, frames)
            )
            aligned = []
            for index in range(frames):
                aligned.append(
                    translate(coarse[:, index], direction * shifts[:, index] / factor)
                )
            context = torch.stack(aligned, dim=1).mean(dim=1)
            return F.interpolate(context, size=(height, width), mode="nearest")

        if condition == "attention":
            aligned = []
            for index in range(frames):
                aligned.append(translate(encoded[:, index], -shifts[:, index]))
            aligned_tensor = torch.stack(aligned, dim=1)
            scores = (shallow[:, None] * aligned_tensor).mean(dim=2, keepdim=True)
            weights = torch.softmax(scores, dim=1)
            return (weights * aligned_tensor).sum(dim=1)

        aligned = []
        for index in range(frames):
            aligned.append(translate(encoded[:, index], direction * shifts[:, index]))
        return torch.stack(aligned, dim=1).mean(dim=1)

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        shifts: torch.Tensor,
        condition: Condition,
    ) -> torch.Tensor:
        if self.input_layout == "splat":
            if condition == "retrieved":
                retrieved_shifts = retrieve_global_shifts(current, history)
                core_input = build_polyphase_observation(
                    current,
                    history,
                    retrieved_shifts,
                    "full",
                )
            else:
                core_input = build_polyphase_observation(current, history, shifts, condition)
        elif self.input_layout == "phase":
            core_input = F.pixel_unshuffle(current, self.phase_factor)
        elif self.input_layout == "hr":
            core_input = F.interpolate(
                current,
                scale_factor=self.scale,
                mode="bicubic",
                align_corners=False,
            )
        else:
            core_input = current
        shallow = F.leaky_relu(self.encoder(core_input), negative_slope=0.1)
        context = (
            torch.zeros_like(shallow)
            if self.input_layout == "splat"
            else self._context(shallow, history, shifts, condition)
        )
        feature = shallow + self.temporal_inject(context)
        feature = self.blocks(feature)
        residual = self.shuffle(self.head(feature))
        base = F.interpolate(
            current,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        return (base + residual).clamp(0.0, 1.0)


def _psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((prediction - target).square(), dim=(1, 2, 3)).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


def _high_frequency(tensor: torch.Tensor) -> torch.Tensor:
    channels = tensor.shape[1]
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    kernel = kernel[None, None].expand(channels, 1, -1, -1)
    return F.conv2d(tensor, kernel, padding=1, groups=channels)


@torch.no_grad()
def evaluate(
    model: OracleProbe,
    config: OracleConfig,
    *,
    condition: Condition,
    seed_offset: int = 1_000_000,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device=config.device).manual_seed(config.seed + seed_offset)
    psnr_values: list[torch.Tensor] = []
    hf_psnr_values: list[torch.Tensor] = []
    l1_values: list[torch.Tensor] = []
    for _ in range(config.val_batches):
        current, history, shifts, target = generate_batch(
            config.batch_size,
            config.hr_size,
            config.history_frames,
            device=torch.device(config.device),
            generator=generator,
        )
        prediction = model(current, history, shifts, condition)
        psnr_values.append(_psnr(prediction, target))
        hf_psnr_values.append(_psnr(_high_frequency(prediction), _high_frequency(target)))
        l1_values.append(torch.mean((prediction - target).abs(), dim=(1, 2, 3)))
    model.train()
    return {
        "psnr": float(torch.cat(psnr_values).mean().item()),
        "hf_psnr": float(torch.cat(hf_psnr_values).mean().item()),
        "l1": float(torch.cat(l1_values).mean().item()),
    }


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(
    path: Path,
    model: OracleProbe,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    best_psnr: float,
    config: OracleConfig,
) -> None:
    torch.save(
        {
            "step": step,
            "best_psnr": best_psnr,
            "condition": config.condition,
            "config": {**asdict(config), "output_dir": str(config.output_dir)},
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def run(config: OracleConfig) -> dict:
    _validate_config(config)
    if not torch.cuda.is_available() and config.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the oracle experiment")
    _seed_everything(config.seed)
    device = torch.device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    setup_swanlab(
        rank=0,
        save_dir=config.output_dir,
        project=config.swanlab_project,
        experiment_name=config.swanlab_experiment,
        config={**asdict(config), "output_dir": str(config.output_dir)},
        disabled=config.disable_swanlab,
        mode=config.swanlab_mode,
    )

    model = OracleProbe(
        channels=config.channels,
        blocks=config.blocks,
        input_layout=config.input_layout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=config.device).manual_seed(config.seed)
    best_psnr = -math.inf
    best_step = 0
    start = time.monotonic()
    last_log = start
    window_loss = 0.0

    print(
        f"CST A0 start condition={config.condition} device={device} steps={config.steps} "
        f"batch={config.batch_size} seed={config.seed}",
        flush=True,
    )
    try:
        for step in range(1, config.steps + 1):
            current, history, shifts, target = generate_batch(
                config.batch_size,
                config.hr_size,
                config.history_frames,
                device=device,
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(current, history, shifts, config.condition)
            loss = F.l1_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            window_loss += float(loss.item())

            if step % config.log_every == 0:
                now = time.monotonic()
                steps_per_second = config.log_every / max(now - last_log, 1e-9)
                metrics = {
                    "train/l1": window_loss / config.log_every,
                    "train/steps_per_second": steps_per_second,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                log_metrics(metrics, step=step)
                print(
                    f"step={step} loss={metrics['train/l1']:.6f} "
                    f"rate={steps_per_second:.2f} step/s",
                    flush=True,
                )
                window_loss = 0.0
                last_log = now

            if step % config.val_every == 0 or step == config.steps:
                metrics = evaluate(model, config, condition=config.condition)
                log_metrics({f"val/{key}": value for key, value in metrics.items()}, step=step)
                print(
                    f"validation step={step} psnr={metrics['psnr']:.4f} "
                    f"hf_psnr={metrics['hf_psnr']:.4f} l1={metrics['l1']:.6f}",
                    flush=True,
                )
                if metrics["psnr"] > best_psnr:
                    best_psnr = metrics["psnr"]
                    best_step = step
                    _save_checkpoint(
                        config.output_dir / "best.pth",
                        model,
                        optimizer,
                        step=step,
                        best_psnr=best_psnr,
                        config=config,
                    )
                _atomic_json(
                    config.output_dir / "progress.json",
                    {
                        "condition": config.condition,
                        "step": step,
                        "best_step": best_step,
                        "best_psnr": best_psnr,
                        "latest": metrics,
                        "elapsed_seconds": time.monotonic() - start,
                    },
                )

        raw = torch.load(config.output_dir / "best.pth", map_location=device, weights_only=False)
        model.load_state_dict(raw["state_dict"], strict=True)
        evaluations = {
            condition: evaluate(model, config, condition=condition)
            for condition in CONDITIONS
        }
        result = {
            "schema_version": 1,
            "status": "completed",
            "condition": config.condition,
            "seed": config.seed,
            "best_step": best_step,
            "best_psnr": best_psnr,
            "elapsed_seconds": time.monotonic() - start,
            "config": {**asdict(config), "output_dir": str(config.output_dir)},
            "evaluations": evaluations,
            "history_dependency": {
                "correct_minus_spatial_psnr": evaluations[config.condition]["psnr"]
                - evaluations["spatial"]["psnr"],
                "correct_minus_wrong_psnr": evaluations[config.condition]["psnr"]
                - evaluations["wrong"]["psnr"],
            },
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
