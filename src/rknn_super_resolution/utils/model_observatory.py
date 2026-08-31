"""Bounded statistical and visual observations for small neural networks.

The observatory deliberately separates collection from rendering.  Tensor data is
reduced immediately to summaries, histograms, channel energies and small spatial
maps, so training code never retains a full activation graph or a video sequence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn

_BACKGROUND = (15, 23, 42)
_PANEL = (30, 41, 59)
_TEXT = (226, 232, 240)
_MUTED = (148, 163, 184)
_ACCENT = (56, 189, 248)
_DANGER = (251, 113, 133)
_GOOD = (74, 222, 128)
_COLORS = (
    (56, 189, 248),
    (244, 114, 182),
    (74, 222, 128),
    (250, 204, 21),
    (167, 139, 250),
    (251, 146, 60),
)


@dataclass(frozen=True)
class TensorSummary:
    """Scalar evidence retained for one tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    finite_ratio: float
    mean: float
    std: float
    rms: float
    abs_max: float
    q01: float
    q50: float
    q99: float
    zero_ratio: float
    clip_ratio: float
    clip_relative_rmse: float
    channel_rms_cv: float | None = None
    stable_rank: float | None = None
    effective_rank: float | None = None
    spectral_norm: float | None = None
    condition_number: float | None = None
    high_frequency_ratio: float | None = None


@dataclass(frozen=True)
class TransitionSummary:
    """Statistics describing a recurrent state or activation update."""

    name: str
    previous_rms: float
    current_rms: float
    growth_ratio: float
    relative_delta_rms: float
    cosine_similarity: float
    sign_flip_ratio: float


@dataclass
class TensorObservation:
    """Compact visual evidence for one tensor."""

    summary: TensorSummary
    histogram_edges: np.ndarray
    histogram_density: np.ndarray
    channel_rms: np.ndarray | None
    spatial_map: np.ndarray | None


def _sample_finite(values: torch.Tensor, max_samples: int) -> torch.Tensor:
    flat = values.detach().float().reshape(-1).cpu()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() > max_samples:
        indices = torch.linspace(0, flat.numel() - 1, max_samples).long()
        flat = flat[indices]
    return flat


def _matrix_spectrum(tensor: torch.Tensor) -> np.ndarray | None:
    if tensor.ndim < 2 or tensor.shape[0] < 2:
        return None
    matrix = tensor.detach().float().reshape(tensor.shape[0], -1).cpu()
    try:
        values = torch.linalg.svdvals(matrix).numpy()
    except RuntimeError:
        return None
    return values[np.isfinite(values)]


def _kernel_frequency(
    tensor: torch.Tensor,
    *,
    size: int = 32,
) -> tuple[float | None, np.ndarray | None]:
    if tensor.ndim != 4 or min(tensor.shape[-2:]) < 2:
        return None, None
    kernel = tensor.detach().float().cpu().numpy()
    spectrum = np.fft.fftshift(
        np.fft.fft2(kernel, s=(size, size), axes=(-2, -1)),
        axes=(-2, -1),
    )
    energy = np.mean(np.abs(spectrum) ** 2, axis=(0, 1))
    yy, xx = np.mgrid[-1.0:1.0:complex(size), -1.0:1.0:complex(size)]
    radius = np.sqrt(xx * xx + yy * yy)
    total = float(energy.sum())
    high_ratio = float(energy[radius >= 0.5].sum() / total) if total > 0 else 0.0
    visual = np.log1p(energy)
    visual -= visual.min()
    if visual.max() > 0:
        visual /= visual.max()
    return high_ratio, visual.astype(np.float32)


def summarize_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    clip_abs: float = 1.0,
    channel_dim: int | None = 0,
    include_structure: bool = False,
    max_samples: int = 262_144,
) -> TensorSummary:
    """Compute deterministic scalar statistics without retaining ``tensor``."""
    detached = tensor.detach().float().cpu()
    sampled = _sample_finite(detached, max_samples)
    finite_ratio = float(torch.isfinite(detached).float().mean().item()) if detached.numel() else 1.0
    if sampled.numel() == 0:
        values = dict.fromkeys(("mean", "std", "rms", "abs_max", "q01", "q50", "q99"), math.nan)
        zero_ratio = clip_ratio = clip_relative_rmse = math.nan
    else:
        quantiles = torch.quantile(sampled, torch.tensor([0.01, 0.5, 0.99]))
        rms = torch.sqrt(torch.mean(sampled.square()))
        clipped = sampled.clamp(-clip_abs, clip_abs)
        values = {
            "mean": float(sampled.mean().item()),
            "std": float(sampled.std(unbiased=False).item()),
            "rms": float(rms.item()),
            "abs_max": float(sampled.abs().max().item()),
            "q01": float(quantiles[0].item()),
            "q50": float(quantiles[1].item()),
            "q99": float(quantiles[2].item()),
        }
        zero_ratio = float((sampled.abs() <= 1e-12).float().mean().item())
        clip_ratio = float((sampled.abs() > clip_abs).float().mean().item())
        clip_rmse = torch.sqrt(torch.mean((clipped - sampled).square()))
        clip_relative_rmse = float((clip_rmse / rms.clamp_min(1e-12)).item())

    channel_rms_cv: float | None = None
    if detached.ndim and channel_dim is not None and detached.shape[channel_dim] > 1:
        dims = tuple(index for index in range(detached.ndim) if index != channel_dim)
        energies = (
            torch.sqrt(torch.mean(detached.square(), dim=dims).clamp_min(0.0))
            if dims
            else detached.abs()
        )
        channel_rms_cv = float(
            (energies.std(unbiased=False) / energies.mean().clamp_min(1e-12)).item()
        )

    stable_rank = effective_rank = spectral_norm = condition_number = None
    high_frequency_ratio = None
    if include_structure:
        singular = _matrix_spectrum(detached)
        if singular is not None and singular.size:
            singular = singular.astype(np.float64)
            squared = singular**2
            spectral_norm = float(singular[0])
            stable_rank = float(squared.sum() / max(squared[0], 1e-24))
            probabilities = singular / max(float(singular.sum()), 1e-24)
            effective_rank = float(np.exp(-(probabilities * np.log(probabilities + 1e-24)).sum()))
            positive = singular[singular > singular[0] * 1e-7]
            condition_number = (
                float(singular[0] / positive[-1]) if positive.size else math.inf
            )
        high_frequency_ratio, _ = _kernel_frequency(detached)

    return TensorSummary(
        name=name,
        shape=tuple(detached.shape),
        dtype=str(tensor.dtype).removeprefix("torch."),
        numel=detached.numel(),
        finite_ratio=finite_ratio,
        zero_ratio=zero_ratio,
        clip_ratio=clip_ratio,
        clip_relative_rmse=clip_relative_rmse,
        channel_rms_cv=channel_rms_cv,
        stable_rank=stable_rank,
        effective_rank=effective_rank,
        spectral_norm=spectral_norm,
        condition_number=condition_number,
        high_frequency_ratio=high_frequency_ratio,
        **values,
    )


def observe_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    clip_abs: float = 1.0,
    channel_dim: int | None = 1,
    max_samples: int = 262_144,
    map_size: tuple[int, int] = (96, 64),
) -> TensorObservation:
    """Reduce an activation/state tensor to bounded statistics and thumbnails."""
    summary = summarize_tensor(
        name,
        tensor,
        clip_abs=clip_abs,
        channel_dim=channel_dim,
        max_samples=max_samples,
    )
    sampled = _sample_finite(tensor, max_samples).numpy()
    if sampled.size:
        low, high = np.quantile(sampled, [0.01, 0.99])
        if not high > low:
            low, high = float(sampled.min()) - 0.5, float(sampled.max()) + 0.5
        density, edges = np.histogram(sampled, bins=64, range=(low, high), density=False)
        density = density.astype(np.float32)
        density /= max(float(density.max()), 1.0)
    else:
        edges = np.linspace(-1.0, 1.0, 65, dtype=np.float32)
        density = np.zeros(64, dtype=np.float32)

    detached = tensor.detach().float().cpu()
    channel_rms = None
    if detached.ndim and channel_dim is not None and detached.shape[channel_dim] > 1:
        dims = tuple(index for index in range(detached.ndim) if index != channel_dim)
        energies = (
            torch.sqrt(torch.mean(detached.square(), dim=dims).clamp_min(0.0))
            if dims
            else detached.abs()
        )
        channel_rms = energies.numpy()

    spatial_map = None
    if detached.ndim >= 2:
        reduced = detached.abs()
        if detached.ndim > 2:
            reduced = reduced.mean(dim=tuple(range(detached.ndim - 2)))
        reduced = reduced[None, None]
        reduced = F.interpolate(reduced, size=(map_size[1], map_size[0]), mode="area")[0, 0]
        spatial_map = reduced.numpy()
        low, high = np.quantile(spatial_map, [0.01, 0.99])
        spatial_map = np.clip((spatial_map - low) / max(float(high - low), 1e-12), 0.0, 1.0)

    return TensorObservation(
        summary=summary,
        histogram_edges=edges.astype(np.float32),
        histogram_density=density,
        channel_rms=channel_rms,
        spatial_map=spatial_map,
    )


def summarize_transition(
    name: str,
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    max_samples: int = 262_144,
) -> TransitionSummary:
    """Measure temporal/state update magnitude, direction and stability."""
    if previous.shape != current.shape:
        raise ValueError(f"transition shapes differ: {previous.shape} != {current.shape}")
    prev = previous.detach().float().reshape(-1).cpu()
    curr = current.detach().float().reshape(-1).cpu()
    finite = torch.isfinite(prev) & torch.isfinite(curr)
    prev, curr = prev[finite], curr[finite]
    count = prev.numel()
    if count > max_samples:
        indices = torch.linspace(0, count - 1, max_samples).long()
        prev, curr = prev[indices], curr[indices]
        count = max_samples
    if count == 0:
        return TransitionSummary(name, *(math.nan for _ in range(6)))
    prev_rms = torch.sqrt(torch.mean(prev.square()))
    curr_rms = torch.sqrt(torch.mean(curr.square()))
    delta_rms = torch.sqrt(torch.mean((curr - prev).square()))
    cosine = torch.dot(prev, curr) / (prev.norm() * curr.norm()).clamp_min(1e-12)
    return TransitionSummary(
        name=name,
        previous_rms=float(prev_rms.item()),
        current_rms=float(curr_rms.item()),
        growth_ratio=float((curr_rms / prev_rms.clamp_min(1e-12)).item()),
        relative_delta_rms=float((delta_rms / prev_rms.clamp_min(1e-12)).item()),
        cosine_similarity=float(cosine.item()),
        sign_flip_ratio=float(((prev * curr) < 0).float().mean().item()),
    )


class TensorObservatory:
    """In-memory bounded collector suitable for forward hooks and recurrent states."""

    def __init__(self, *, clip_abs: float = 1.0, max_samples: int = 262_144) -> None:
        self.clip_abs = clip_abs
        self.max_samples = max_samples
        self.observations: dict[str, TensorObservation] = {}
        self.transitions: dict[str, TransitionSummary] = {}

    def observe(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        channel_dim: int | None = 1,
    ) -> None:
        self.observations[name] = observe_tensor(
            name,
            tensor,
            clip_abs=self.clip_abs,
            channel_dim=channel_dim,
            max_samples=self.max_samples,
        )

    def observe_transition(
        self,
        name: str,
        previous: torch.Tensor,
        current: torch.Tensor,
    ) -> None:
        self.transitions[name] = summarize_transition(
            name,
            previous,
            current,
            max_samples=self.max_samples,
        )

    def scalar_metrics(self, *, prefix: str = "observatory") -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, observation in self.observations.items():
            summary = observation.summary
            safe = name.replace("/", "_")
            metrics[f"{prefix}/{safe}/rms"] = summary.rms
            metrics[f"{prefix}/{safe}/abs_max"] = summary.abs_max
            metrics[f"{prefix}/{safe}/zero_ratio"] = summary.zero_ratio
            metrics[f"{prefix}/{safe}/clip_ratio"] = summary.clip_ratio
            if summary.channel_rms_cv is not None:
                metrics[f"{prefix}/{safe}/channel_rms_cv"] = summary.channel_rms_cv
        for name, transition in self.transitions.items():
            safe = name.replace("/", "_")
            metrics[f"{prefix}/{safe}/growth_ratio"] = transition.growth_ratio
            metrics[f"{prefix}/{safe}/relative_delta_rms"] = transition.relative_delta_rms
            metrics[f"{prefix}/{safe}/cosine_similarity"] = transition.cosine_similarity
            metrics[f"{prefix}/{safe}/sign_flip_ratio"] = transition.sign_flip_ratio
        return metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensors": [asdict(item.summary) for item in self.observations.values()],
            "transitions": [asdict(item) for item in self.transitions.values()],
        }


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class ForwardHookObservatory:
    """Opt-in forward-hook collector for direct analysis of a small model.

    Hooks default to leaf modules with trainable parameters. Collection performs
    a CPU reduction during the forward pass, so it is intended for diagnostic
    batches rather than every training step.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        observatory: TensorObservatory | None = None,
        module_names: set[str] | None = None,
        max_modules: int = 96,
    ) -> None:
        self.observatory = observatory or TensorObservatory()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        selected = 0
        for name, module in model.named_modules():
            if not name or selected >= max_modules:
                continue
            if module_names is not None:
                if name not in module_names:
                    continue
            elif any(module.children()) or not any(
                parameter.requires_grad for parameter in module.parameters(recurse=False)
            ):
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
            selected += 1

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is not None and tensor.is_floating_point():
                channel_dim = 1 if tensor.ndim >= 2 else 0 if tensor.ndim else None
                self.observatory.observe(name, tensor, channel_dim=channel_dim)

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> ForwardHookObservatory:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def analyze_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    clip_abs: float = 1.0,
) -> tuple[list[TensorObservation], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Analyze every floating tensor in a checkpoint, including exact spectra."""
    observations: list[TensorObservation] = []
    spectra: dict[str, np.ndarray] = {}
    frequencies: dict[str, np.ndarray] = {}
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        channel_dim = 0 if tensor.ndim else None
        observation = observe_tensor(
            name,
            tensor,
            clip_abs=clip_abs,
            channel_dim=channel_dim,
        )
        structural = summarize_tensor(
            name,
            tensor,
            clip_abs=clip_abs,
            channel_dim=channel_dim,
            include_structure=tensor.ndim >= 2,
        )
        observation.summary = structural
        observations.append(observation)
        singular = _matrix_spectrum(tensor)
        if singular is not None and singular.size:
            spectra[name] = singular.astype(np.float32)
        _, frequency = _kernel_frequency(tensor)
        if frequency is not None:
            frequencies[name] = frequency
    return observations, spectra, frequencies


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    suffix = "Bold" if bold else ""
    candidates = (
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _heat_color(value: float) -> tuple[int, int, int]:
    value = float(np.clip(value, 0.0, 1.0))
    stops = ((15, 23, 42), (30, 64, 175), (14, 165, 233), (250, 204, 21))
    position = value * (len(stops) - 1)
    index = min(int(position), len(stops) - 2)
    amount = position - index
    return tuple(int(stops[index][i] * (1 - amount) + stops[index + 1][i] * amount) for i in range(3))


def _heat_image(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    normalized = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if normalized.ndim == 1:
        normalized = normalized[None, :]
    rgb = np.empty((*normalized.shape, 3), dtype=np.uint8)
    for y in range(normalized.shape[0]):
        for x in range(normalized.shape[1]):
            rgb[y, x] = _heat_color(float(normalized[y, x]))
    return Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST)


def _short_name(name: str, limit: int = 48) -> str:
    return name if len(name) <= limit else "..." + name[-(limit - 3) :]


def render_checkpoint_overview(
    observations: list[TensorObservation],
    *,
    checkpoint_name: str,
    clip_abs: float,
    max_layers: int = 96,
) -> np.ndarray:
    """Render a layer-by-layer checkpoint health table."""
    weights = [item for item in observations if len(item.summary.shape) >= 2]
    weights = weights[:max_layers]
    row_h, width = 34, 1720
    height = 164 + row_h * max(len(weights), 1)
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font, body_font, small_font = _font(28, bold=True), _font(15), _font(13)
    draw.text((28, 20), "Model Observatory / Checkpoint Structure", font=title_font, fill=_TEXT)
    total = sum(item.summary.numel for item in observations)
    risky = sum(item.summary.clip_relative_rmse > 0.01 for item in weights)
    draw.text(
        (28, 62),
        f"{checkpoint_name}  |  {total:,} floating values  |  {len(weights)} matrices  |  "
        f"clip=[-{clip_abs:g}, {clip_abs:g}]  |  {risky} layers >1% clip RMSE",
        font=body_font,
        fill=_MUTED,
    )
    columns = (
        (28, "tensor"),
        (600, "shape"),
        (770, "RMS"),
        (865, "abs max"),
        (965, "clip %"),
        (1065, "clip RMSE"),
        (1190, "eff rank"),
        (1295, "stable rank"),
        (1410, "ch RMS CV"),
        (1530, "HF energy"),
    )
    y0 = 112
    draw.rectangle((20, y0 - 8, width - 20, y0 + 24), fill=_PANEL)
    for x, label in columns:
        draw.text((x, y0), label, font=small_font, fill=_MUTED)
    for index, item in enumerate(weights):
        s = item.summary
        y = y0 + 34 + index * row_h
        if index % 2:
            draw.rectangle((20, y - 5, width - 20, y + row_h - 5), fill=(22, 32, 49))
        risk = s.clip_relative_rmse
        color = _DANGER if risk > 0.01 else (_GOOD if risk == 0 else _TEXT)
        values = (
            (28, _short_name(s.name), color),
            (600, "x".join(map(str, s.shape)), _TEXT),
            (770, f"{s.rms:.3g}", _TEXT),
            (865, f"{s.abs_max:.3g}", _TEXT),
            (965, f"{100*s.clip_ratio:.2f}", color),
            (1065, f"{100*risk:.2f}%", color),
            (1190, "-" if s.effective_rank is None else f"{s.effective_rank:.2f}", _TEXT),
            (1295, "-" if s.stable_rank is None else f"{s.stable_rank:.2f}", _TEXT),
            (1410, "-" if s.channel_rms_cv is None else f"{s.channel_rms_cv:.2f}", _TEXT),
            (1530, "-" if s.high_frequency_ratio is None else f"{s.high_frequency_ratio:.2f}", _TEXT),
        )
        for x, value, fill in values:
            draw.text((x, y), value, font=small_font, fill=fill)
    return np.asarray(image)


def render_quantization_risk(
    observations: list[TensorObservation],
    *,
    clip_abs: float,
    max_layers: int = 32,
) -> np.ndarray:
    """Render hard-clip sensitivity and outlier burden by layer."""
    ranked = sorted(
        (item.summary for item in observations if len(item.summary.shape) >= 2),
        key=lambda item: item.clip_relative_rmse,
        reverse=True,
    )[:max_layers]
    row_h, width = 38, 1400
    height = 130 + row_h * max(len(ranked), 1)
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((28, 18), "Quantization / hard-clip sensitivity", font=_font(27, bold=True), fill=_TEXT)
    draw.text(
        (28, 58),
        f"Relative output-independent weight RMSE after clamp to [-{clip_abs:g}, {clip_abs:g}]",
        font=_font(15),
        fill=_MUTED,
    )
    name_x, bar_x, bar_w = 28, 610, 620
    maximum = max((item.clip_relative_rmse for item in ranked), default=1.0)
    maximum = max(maximum, 0.01)
    for index, item in enumerate(ranked):
        y = 105 + index * row_h
        draw.text((name_x, y), _short_name(item.name), font=_font(13), fill=_TEXT)
        draw.rectangle((bar_x, y + 2, bar_x + bar_w, y + 19), fill=_PANEL)
        amount = min(item.clip_relative_rmse / maximum, 1.0)
        fill = _DANGER if item.clip_relative_rmse > 0.01 else _ACCENT
        draw.rectangle((bar_x, y + 2, bar_x + int(bar_w * amount), y + 19), fill=fill)
        draw.text(
            (bar_x + bar_w + 18, y),
            f"RMSE {100*item.clip_relative_rmse:6.2f}%  |  clipped {100*item.clip_ratio:6.2f}%",
            font=_font(13),
            fill=fill,
        )
    return np.asarray(image)


def render_spectra(
    spectra: dict[str, np.ndarray],
    *,
    max_layers: int = 24,
) -> np.ndarray:
    """Render normalized singular-value spectra for all small weight matrices."""
    width, height = 1450, 840
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((28, 18), "Weight singular spectra", font=_font(27, bold=True), fill=_TEXT)
    draw.text(
        (28, 58),
        "Each curve is normalized by its largest singular value; steep collapse indicates channel redundancy.",
        font=_font(15),
        fill=_MUTED,
    )
    left, top, right, bottom = 82, 105, 1010, 765
    draw.rectangle((left, top, right, bottom), outline=_MUTED, width=1)
    for tick in range(5):
        y = top + tick * (bottom - top) // 4
        draw.line((left, y, right, y), fill=(51, 65, 85), width=1)
        draw.text((25, y - 8), f"{10 ** (-tick):.0e}", font=_font(12), fill=_MUTED)
    entries = list(spectra.items())[:max_layers]
    for index, (name, values) in enumerate(entries):
        if not values.size or values[0] <= 0:
            continue
        normalized = np.clip(values / values[0], 1e-4, 1.0)
        points = []
        for position, value in enumerate(normalized):
            x = left + int((right - left) * position / max(len(normalized) - 1, 1))
            y = top + int((bottom - top) * (-math.log10(float(value))) / 4.0)
            points.append((x, min(y, bottom)))
        color = _COLORS[index % len(_COLORS)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        legend_y = 110 + index * 27
        draw.line((1050, legend_y + 8, 1080, legend_y + 8), fill=color, width=3)
        draw.text((1092, legend_y), _short_name(name, 38), font=_font(12), fill=_TEXT)
    return np.asarray(image)


def render_channel_atlas(
    observations: list[TensorObservation],
    *,
    max_layers: int = 48,
) -> np.ndarray:
    """Render per-output-channel RMS as a layer atlas."""
    entries = [item for item in observations if item.channel_rms is not None][:max_layers]
    row_h, width = 34, 1320
    height = 112 + row_h * max(len(entries), 1)
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((28, 18), "Output-channel energy atlas", font=_font(27, bold=True), fill=_TEXT)
    draw.text((28, 58), "Color is per-layer normalized channel RMS; dark stripes expose unused channels.", font=_font(15), fill=_MUTED)
    for index, item in enumerate(entries):
        y = 96 + index * row_h
        values = np.asarray(item.channel_rms, dtype=np.float32)
        values /= max(float(values.max()), 1e-12)
        draw.text((28, y), _short_name(item.summary.name), font=_font(12), fill=_TEXT)
        heat = _heat_image(values, (690, 21))
        image.paste(heat, (595, y - 2))
    return np.asarray(image)


def render_frequency_atlas(
    frequencies: dict[str, np.ndarray],
    *,
    max_layers: int = 36,
) -> np.ndarray:
    """Render average 2-D kernel frequency response for convolution weights."""
    entries = list(frequencies.items())[:max_layers]
    cols, tile_w, tile_h = 6, 218, 198
    rows = max(math.ceil(len(entries) / cols), 1)
    width, height = 28 + cols * tile_w, 94 + rows * tile_h
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((28, 16), "Convolution kernel frequency atlas", font=_font(27, bold=True), fill=_TEXT)
    draw.text((28, 54), "Center = DC / smooth response; bright edges = high-frequency emphasis.", font=_font(15), fill=_MUTED)
    for index, (name, frequency) in enumerate(entries):
        row, col = divmod(index, cols)
        x, y = 28 + col * tile_w, 91 + row * tile_h
        heat = _heat_image(frequency, (154, 154))
        image.paste(heat, (x, y))
        draw.text((x, y + 158), _short_name(name, 25), font=_font(11), fill=_TEXT)
    return np.asarray(image)


def render_tensor_observatory(observatory: TensorObservatory) -> np.ndarray:
    """Render activation/state distributions, channel energies and spatial maps."""
    entries = list(observatory.observations.values())
    row_h, width = 164, 1580
    height = 110 + row_h * max(len(entries), 1) + 42 * len(observatory.transitions)
    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((28, 18), "Activation and recurrent-state observatory", font=_font(27, bold=True), fill=_TEXT)
    draw.text((28, 58), "Bounded sketches: distribution, channel RMS, spatial |activation| and saturation.", font=_font(15), fill=_MUTED)
    for index, item in enumerate(entries):
        y = 98 + index * row_h
        summary = item.summary
        draw.rectangle((20, y - 5, width - 20, y + row_h - 12), fill=_PANEL)
        draw.text((32, y + 2), _short_name(summary.name, 58), font=_font(16, bold=True), fill=_TEXT)
        draw.text(
            (32, y + 33),
            f"shape={summary.shape}  mean={summary.mean:.3g}  rms={summary.rms:.3g}  "
            f"q01/50/99={summary.q01:.3g}/{summary.q50:.3g}/{summary.q99:.3g}",
            font=_font(12),
            fill=_MUTED,
        )
        draw.text(
            (32, y + 58),
            f"zero={100*summary.zero_ratio:.2f}%  clip={100*summary.clip_ratio:.2f}%  "
            f"channel CV={summary.channel_rms_cv if summary.channel_rms_cv is not None else 0:.3f}",
            font=_font(12),
            fill=_DANGER if summary.clip_ratio > 0.01 else _TEXT,
        )
        hist_left, hist_top, hist_w, hist_h = 560, y + 10, 360, 108
        draw.rectangle((hist_left, hist_top, hist_left + hist_w, hist_top + hist_h), fill=_BACKGROUND)
        density = item.histogram_density
        points = [
            (
                hist_left + int(i * hist_w / max(len(density) - 1, 1)),
                hist_top + hist_h - int(float(value) * (hist_h - 8)),
            )
            for i, value in enumerate(density)
        ]
        if len(points) > 1:
            draw.line(points, fill=_ACCENT, width=2)
        if item.channel_rms is not None:
            values = np.asarray(item.channel_rms, dtype=np.float32)
            values /= max(float(values.max()), 1e-12)
            image.paste(_heat_image(values, (360, 22)), (hist_left, y + 126))
        if item.spatial_map is not None:
            image.paste(_heat_image(item.spatial_map, (450, 138)), (990, y + 5))

    y = 104 + row_h * len(entries)
    for transition in observatory.transitions.values():
        draw.text(
            (32, y),
            f"transition {transition.name}: growth={transition.growth_ratio:.3f}  "
            f"delta/prev={transition.relative_delta_rms:.3f}  cos={transition.cosine_similarity:.3f}  "
            f"sign flips={100*transition.sign_flip_ratio:.2f}%",
            font=_font(14),
            fill=_TEXT,
        )
        y += 42
    return np.asarray(image)


def save_png(image: np.ndarray, path: Path) -> Path:
    """Save an RGB uint8 observation panel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)
    return path
