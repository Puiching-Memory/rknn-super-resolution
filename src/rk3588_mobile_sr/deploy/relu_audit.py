"""Audit ReLU activation sparsity and impact on real DIV2K validation images."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from rk3588_mobile_sr.config import load_config
from rk3588_mobile_sr.deploy.rknn_eval import psnr_numpy
from rk3588_mobile_sr.models.mobileone_block import MobileOneBlock
from rk3588_mobile_sr.models.mobileone_sr import MobileOneSR
from rk3588_mobile_sr.utils.train_framework import _normalize_state_dict, require_cuda


@dataclass(frozen=True)
class ImagePair:
    lr_path: Path
    hr_path: Path


@dataclass
class LayerReLUStats:
    pre_neg_mean: float = 0.0
    post_zero_mean: float = 0.0
    pre_std_mean: float = 0.0
    post_std_mean: float = 0.0
    ch_always_neg_pct: float = 0.0
    ch_gt90_neg_pct: float = 0.0
    ch_gt50_neg_pct: float = 0.0
    ch_always_zero_pct: float = 0.0
    worst_ch_neg: int = 0
    worst_ch_neg_frac: float = 0.0
    worst_ch_zero: int = 0
    worst_ch_zero_frac: float = 0.0


@dataclass
class _LayerAccumulator:
    pre_neg_frac: list[float] = field(default_factory=list)
    post_zero_frac: list[float] = field(default_factory=list)
    pre_std_mean: list[float] = field(default_factory=list)
    post_std_mean: list[float] = field(default_factory=list)
    ch_pre_neg: np.ndarray | None = None
    ch_post_zero: np.ndarray | None = None


def collect_image_pairs(
    hr_dir: Path,
    lr_dir: Path,
    *,
    scale: int,
    max_images: int | None,
) -> list[ImagePair]:
    """Pair HR/LR paths using the same stem convention as ``div2k_loader``."""
    hr_paths = sorted(hr_dir.glob("*.png"))
    if not hr_paths:
        hr_paths = sorted(hr_dir.glob("*.jpg"))

    lr_files = sorted(lr_dir.glob(f"*x{scale}.png"))
    if not lr_files:
        lr_files = sorted(lr_dir.glob("*.png"))
    lr_map = {p.stem.split("x")[0]: p for p in lr_files}

    pairs: list[ImagePair] = []
    for hr_path in hr_paths:
        lr_path = lr_map.get(hr_path.stem)
        if lr_path is None:
            continue
        pairs.append(ImagePair(lr_path=lr_path, hr_path=hr_path))
        if max_images is not None and len(pairs) >= max_images:
            break
    if not pairs:
        raise FileNotFoundError(f"No paired images under {hr_dir} and {lr_dir}")
    return pairs


def load_lr_tensor(
    pair: ImagePair,
    *,
    input_h: int,
    input_w: int,
    device: torch.device,
) -> torch.Tensor:
    lr = resize(read_image(str(pair.lr_path)).float(), [input_h, input_w], antialias=True)
    return lr.unsqueeze(0).to(device)


def load_hr_numpy(
    pair: ImagePair,
    *,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    hr = resize(read_image(str(pair.hr_path)).float(), [out_h, out_w], antialias=True)
    return hr.permute(1, 2, 0).numpy()


def load_checkpoint_model(
    weight: Path,
    device: torch.device,
    *,
    scale: int = 3,
    num_channels: int = 32,
    num_blocks: int = 6,
    num_conv_branches: int = 4,
    deploy: bool = False,
    identity_var_floor: float = 0.0,
) -> MobileOneSR:
    raw = torch.load(weight, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = _normalize_state_dict(raw["state_dict"])
    elif isinstance(raw, dict):
        state_dict = _normalize_state_dict(raw)
    else:
        raise TypeError(f"Unsupported checkpoint format in {weight}")

    model = MobileOneSR(
        scale=scale,
        num_channels=num_channels,
        num_blocks=num_blocks,
        num_conv_branches=num_conv_branches,
    ).to(device)
    model.load_state_dict(state_dict)
    if deploy:
        model.switch_to_deploy(identity_var_floor=identity_var_floor)
    model.eval()
    return model


def audit_per_layer_relu(
    model: MobileOneSR,
    pairs: Sequence[ImagePair],
    *,
    input_h: int,
    input_w: int,
    device: torch.device,
) -> dict[str, LayerReLUStats]:
    """Hook pre-ReLU activations in the train graph (before ``switch_to_deploy``)."""
    if any(isinstance(block, MobileOneBlock) and block.inference_mode for block in model.body):
        raise ValueError("audit_per_layer_relu requires the train graph; do not call switch_to_deploy first.")

    storage: dict[str, _LayerAccumulator] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0].detach()
            pre_neg = (x < 0).float().mean().item()
            post = torch.relu(x)
            post_zero = (post == 0).float().mean().item()
            st = storage.setdefault(name, _LayerAccumulator())
            st.pre_neg_frac.append(pre_neg)
            st.post_zero_frac.append(post_zero)
            st.pre_std_mean.append(float(x.std().item()))
            st.post_std_mean.append(float(post.std().item()))
            ch_neg = (x < 0).float().mean(dim=(0, 2, 3)).cpu().numpy()
            ch_zero = (post == 0).float().mean(dim=(0, 2, 3)).cpu().numpy()
            if st.ch_pre_neg is None:
                st.ch_pre_neg = ch_neg.copy()
                st.ch_post_zero = ch_zero.copy()
            else:
                st.ch_pre_neg += ch_neg
                st.ch_post_zero += ch_zero

        return hook

    handles.append(model.stem[2].register_forward_pre_hook(make_hook("stem.relu")))
    for index, block in enumerate(model.body):
        if isinstance(block, MobileOneBlock) and not block.inference_mode:
            handles.append(block.relu.register_forward_pre_hook(make_hook(f"body.{index}.relu")))

    with torch.no_grad():
        for pair in pairs:
            model(load_lr_tensor(pair, input_h=input_h, input_w=input_w, device=device))

    for handle in handles:
        handle.remove()

    n = len(pairs)
    out: dict[str, LayerReLUStats] = {}
    for name, st in storage.items():
        ch_pre = st.ch_pre_neg / n  # type: ignore[operator]
        ch_post = st.ch_post_zero / n  # type: ignore[operator]
        out[name] = LayerReLUStats(
            pre_neg_mean=float(np.mean(st.pre_neg_frac)),
            post_zero_mean=float(np.mean(st.post_zero_frac)),
            pre_std_mean=float(np.mean(st.pre_std_mean)),
            post_std_mean=float(np.mean(st.post_std_mean)),
            ch_always_neg_pct=float((ch_pre > 0.999).mean() * 100),
            ch_gt90_neg_pct=float((ch_pre > 0.90).mean() * 100),
            ch_gt50_neg_pct=float((ch_pre > 0.50).mean() * 100),
            ch_always_zero_pct=float((ch_post > 0.999).mean() * 100),
            worst_ch_neg=int(ch_pre.argmax()),
            worst_ch_neg_frac=float(ch_pre.max()),
            worst_ch_zero=int(ch_post.argmax()),
            worst_ch_zero_frac=float(ch_post.max()),
        )
    return out


def audit_global_skip(
    model: MobileOneSR,
    pairs: Sequence[ImagePair],
    *,
    input_h: int,
    input_w: int,
    device: torch.device,
) -> dict[str, float]:
    """Measure how ``f_body + f0`` reduces zero activations before ``out_conv``."""
    stem_zero: list[float] = []
    body_zero: list[float] = []
    skip_zero: list[float] = []

    with torch.no_grad():
        for pair in pairs:
            x = load_lr_tensor(pair, input_h=input_h, input_w=input_w, device=device)
            f0 = model.stem(x)
            f_body = model.body(f0)
            f_skip = f_body + f0
            stem_zero.append((f0 == 0).float().mean().item())
            body_zero.append((f_body == 0).float().mean().item())
            skip_zero.append((f_skip == 0).float().mean().item())

    stem_mean = float(np.mean(stem_zero))
    body_mean = float(np.mean(body_zero))
    skip_mean = float(np.mean(skip_zero))
    return {
        "stem_post_relu_zero_mean": stem_mean,
        "body_post_relu_zero_mean": body_mean,
        "after_global_skip_zero_mean": skip_mean,
        "skip_rescue_delta_zero": body_mean - skip_mean,
    }


def identity_var_report(model: MobileOneSR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, block in enumerate(model.body):
        if not isinstance(block, MobileOneBlock) or block.inference_mode:
            continue
        running_var = block.identity.running_var.detach().cpu().numpy()
        rows.append(
            {
                "layer": f"body.{index}",
                "var_min": float(running_var.min()),
                "var_lt_1e-4": int((running_var < 1e-4).sum()),
                "var_lt_0.01": int((running_var < 0.01).sum()),
                "var_mean": float(running_var.mean()),
            }
        )
    return rows


def correlate_var_vs_channel_zero(
    model: MobileOneSR,
    pairs: Sequence[ImagePair],
    *,
    input_h: int,
    input_w: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Pearson correlation between identity ``running_var`` and per-channel ReLU zero rate."""
    ch_zero_acc = {index: np.zeros(model.num_channels, dtype=np.float64) for index in range(len(model.body))}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(block_index: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            post = torch.relu(inputs[0].detach())
            ch_zero_acc[block_index] += (post == 0).float().mean(dim=(0, 2, 3)).cpu().numpy()

        return hook

    for index, block in enumerate(model.body):
        if isinstance(block, MobileOneBlock) and not block.inference_mode:
            handles.append(block.relu.register_forward_pre_hook(make_hook(index)))

    with torch.no_grad():
        for pair in pairs:
            model(load_lr_tensor(pair, input_h=input_h, input_w=input_w, device=device))

    for handle in handles:
        handle.remove()

    rows: list[dict[str, Any]] = []
    n = len(pairs)
    for index, block in enumerate(model.body):
        if not isinstance(block, MobileOneBlock) or block.inference_mode:
            continue
        running_var = block.identity.running_var.detach().cpu().numpy()
        zero_rate = ch_zero_acc[index] / n
        if running_var.std() > 0 and zero_rate.std() > 0:
            corr = float(np.corrcoef(running_var, zero_rate)[0, 1])
        else:
            corr = float("nan")
        worst = int(zero_rate.argmax())
        rows.append(
            {
                "layer": f"body.{index}",
                "corr_var_vs_ch_zero": corr,
                "worst_ch": worst,
                "worst_ch_zero_frac": float(zero_rate[worst]),
                "worst_ch_var": float(running_var[worst]),
            }
        )
    return rows


def channels_gt90_negative(
    model: MobileOneSR,
    pairs: Sequence[ImagePair],
    *,
    input_h: int,
    input_w: int,
    device: torch.device,
    threshold: float = 0.90,
) -> dict[str, list[dict[str, float]]]:
    """List channels whose pre-ReLU activations are negative on >threshold of spatial sites."""
    out: dict[str, list[dict[str, float]]] = {}
    for index, block in enumerate(model.body):
        if not isinstance(block, MobileOneBlock) or block.inference_mode:
            continue
        acc = np.zeros(model.num_channels, dtype=np.float64)

        def make_hook(state: dict[str, Any]):
            def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
                state["acc"] += (inputs[0] < 0).float().mean(dim=(0, 2, 3)).cpu().numpy()
                state["count"] += 1

            return hook

        state: dict[str, Any] = {"acc": acc, "count": 0}
        handle = block.relu.register_forward_pre_hook(make_hook(state))
        with torch.no_grad():
            for pair in pairs:
                model(load_lr_tensor(pair, input_h=input_h, input_w=input_w, device=device))
        handle.remove()

        neg_frac = state["acc"] / state["count"]
        bad = [
            {"channel": int(ch), "pre_neg_frac": float(neg_frac[ch])}
            for ch in np.where(neg_frac > threshold)[0]
        ]
        out[f"body.{index}"] = bad
    return out


def per_channel_block_stats(
    model: MobileOneSR,
    pairs: Sequence[ImagePair],
    *,
    block_index: int,
    input_h: int,
    input_w: int,
    device: torch.device,
) -> list[dict[str, float]]:
    """Per-channel pre-ReLU negativity and post-ReLU std for one body block."""
    block = model.body[block_index]
    if not isinstance(block, MobileOneBlock) or block.inference_mode:
        raise ValueError(f"body.{block_index} is not a train-graph MobileOneBlock")

    state = {
        "n": 0,
        "neg": np.zeros(model.num_channels),
        "std": np.zeros(model.num_channels),
        "mean": np.zeros(model.num_channels),
    }

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        x = inputs[0].detach()
        state["neg"] += (x < 0).float().mean(dim=(0, 2, 3)).cpu().numpy()
        state["std"] += torch.relu(x).std(dim=(0, 2, 3)).cpu().numpy()
        state["mean"] += x.mean(dim=(0, 2, 3)).cpu().numpy()
        state["n"] += 1

    handle = block.relu.register_forward_pre_hook(hook)
    with torch.no_grad():
        for pair in pairs:
            model(load_lr_tensor(pair, input_h=input_h, input_w=input_w, device=device))
    handle.remove()

    n = state["n"]
    running_var = block.identity.running_var.detach().cpu().numpy()
    rows: list[dict[str, float]] = []
    for ch in range(model.num_channels):
        rows.append(
            {
                "channel": ch,
                "identity_var": float(running_var[ch]),
                "pre_mean": float(state["mean"][ch] / n),
                "pre_neg_frac": float(state["neg"][ch] / n),
                "post_std": float(state["std"][ch] / n),
            }
        )
    return rows


def measure_deploy_psnr(
    model: MobileOneSR,
    pairs: Sequence[ImagePair],
    *,
    input_h: int,
    input_w: int,
    out_h: int,
    out_w: int,
    device: torch.device,
) -> float:
    psnrs: list[float] = []
    with torch.no_grad():
        for pair in pairs:
            sr = (
                torch.clamp(
                    model(load_lr_tensor(pair, input_h=input_h, input_w=input_w, device=device)),
                    0.0,
                    255.0,
                )
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            hr = load_hr_numpy(pair, out_h=out_h, out_w=out_w)
            psnrs.append(psnr_numpy(sr, hr))
    return float(np.mean(psnrs))


def oracle_no_body_relu_psnr(
    weight: Path,
    pairs: Sequence[ImagePair],
    *,
    input_h: int,
    input_w: int,
    out_h: int,
    out_w: int,
    device: torch.device,
    identity_var_floor: float,
    scale: int,
    num_channels: int,
    num_blocks: int,
    num_conv_branches: int,
) -> float:
    """Deploy graph with body ReLU removed (invalid oracle; shows training dependence on ReLU)."""

    def no_relu_forward(self: MobileOneBlock, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return self.reparam_conv(x)
        out = 0.0
        for branch in self.conv_branches:
            out += branch(x)
        out += self.conv_1x1(x)
        if self.identity is not None:
            out += self.identity(x)
        return out

    model = load_checkpoint_model(
        weight,
        device,
        scale=scale,
        num_channels=num_channels,
        num_blocks=num_blocks,
        num_conv_branches=num_conv_branches,
        deploy=True,
        identity_var_floor=identity_var_floor,
    )
    for block in model.body:
        block.forward = no_relu_forward.__get__(block, MobileOneBlock)  # type: ignore[method-assign]
    return measure_deploy_psnr(
        model,
        pairs,
        input_h=input_h,
        input_w=input_w,
        out_h=out_h,
        out_w=out_w,
        device=device,
    )


def leaky_relu_sweep_psnr(
    weight: Path,
    pairs: Sequence[ImagePair],
    *,
    slopes: Sequence[float],
    input_h: int,
    input_w: int,
    out_h: int,
    out_w: int,
    device: torch.device,
    identity_var_floor: float,
    scale: int,
    num_channels: int,
    num_blocks: int,
    num_conv_branches: int,
) -> dict[str, float]:
    """Replace ReLU with LeakyReLU(slope) at inference without retraining."""
    results: dict[str, float] = {}
    for slope in slopes:
        model = load_checkpoint_model(
            weight,
            device,
            scale=scale,
            num_channels=num_channels,
            num_blocks=num_blocks,
            num_conv_branches=num_conv_branches,
            deploy=True,
            identity_var_floor=identity_var_floor,
        )
        model.stem[2] = nn.LeakyReLU(slope, inplace=True)
        for block in model.body:
            if isinstance(block, MobileOneBlock) and block.inference_mode:
                reparam_conv = block.reparam_conv

                def fused_forward(
                    x: torch.Tensor,
                    conv: nn.Conv2d = reparam_conv,
                    negative_slope: float = slope,
                ) -> torch.Tensor:
                    return torch.nn.functional.leaky_relu(conv(x), negative_slope=negative_slope)

                block.forward = fused_forward  # type: ignore[method-assign]
        results[str(slope)] = measure_deploy_psnr(
            model,
            pairs,
            input_h=input_h,
            input_w=input_w,
            out_h=out_h,
            out_w=out_w,
            device=device,
        )
    return results


def channel_ablation_psnr(
    model: MobileOneSR,
    pairs: Sequence[ImagePair],
    *,
    block_index: int,
    channel: int,
    input_h: int,
    input_w: int,
    out_h: int,
    out_w: int,
    device: torch.device,
) -> float:
    """Zero one channel at a body block output and measure deploy PSNR."""

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        output[:, channel, :, :] = 0
        return output

    handle = model.body[block_index].register_forward_hook(hook)
    try:
        return measure_deploy_psnr(
            model,
            pairs,
            input_h=input_h,
            input_w=input_w,
            out_h=out_h,
            out_w=out_w,
            device=device,
        )
    finally:
        handle.remove()


@dataclass
class AuditReport:
    num_images: int
    input_h: int
    input_w: int
    baseline_psnr: float
    per_layer_relu: dict[str, LayerReLUStats]
    global_skip: dict[str, float]
    identity_var: list[dict[str, Any]]
    var_vs_ch_zero_corr: list[dict[str, Any]]
    channels_gt90_neg: dict[str, list[dict[str, float]]]
    oracle_no_body_relu_psnr: float | None = None
    leaky_relu_sweep: dict[str, float] | None = None
    oracle_subset_baseline_psnr: float | None = None
    channel_ablation: list[dict[str, Any]] | None = None
    per_channel_blocks: dict[str, list[dict[str, float]]] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "num_images": self.num_images,
            "input_h": self.input_h,
            "input_w": self.input_w,
            "baseline_psnr": self.baseline_psnr,
            "per_layer_relu": {name: asdict(stats) for name, stats in self.per_layer_relu.items()},
            "global_skip": self.global_skip,
            "identity_var": self.identity_var,
            "var_vs_ch_zero_corr": self.var_vs_ch_zero_corr,
            "channels_gt90_neg": self.channels_gt90_neg,
        }
        if self.oracle_no_body_relu_psnr is not None:
            data["oracle_no_body_relu_psnr"] = self.oracle_no_body_relu_psnr
        if self.oracle_subset_baseline_psnr is not None:
            data["oracle_subset_baseline_psnr"] = self.oracle_subset_baseline_psnr
        if self.leaky_relu_sweep is not None:
            data["leaky_relu_sweep"] = self.leaky_relu_sweep
        if self.channel_ablation is not None:
            data["channel_ablation"] = self.channel_ablation
        if self.per_channel_blocks is not None:
            data["per_channel_blocks"] = self.per_channel_blocks
        return data


def run_audit(args: argparse.Namespace) -> AuditReport:
    device = torch.device(args.device)
    if device.type == "cuda":
        require_cuda()

    scale = args.scale
    input_h, input_w = args.input_h, args.input_w
    out_h, out_w = input_h * scale, input_w * scale
    pairs = collect_image_pairs(
        Path(args.hr_dir),
        Path(args.lr_dir),
        scale=scale,
        max_images=args.max_images,
    )

    train_model = load_checkpoint_model(
        Path(args.weight),
        device,
        scale=scale,
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        num_conv_branches=args.num_conv_branches,
        deploy=False,
    )
    deploy_model = load_checkpoint_model(
        Path(args.weight),
        device,
        scale=scale,
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        num_conv_branches=args.num_conv_branches,
        deploy=True,
        identity_var_floor=args.identity_var_floor,
    )

    baseline = measure_deploy_psnr(
        deploy_model,
        pairs,
        input_h=input_h,
        input_w=input_w,
        out_h=out_h,
        out_w=out_w,
        device=device,
    )
    per_layer = audit_per_layer_relu(
        train_model,
        pairs,
        input_h=input_h,
        input_w=input_w,
        device=device,
    )
    global_skip = audit_global_skip(
        deploy_model,
        pairs,
        input_h=input_h,
        input_w=input_w,
        device=device,
    )
    id_vars = identity_var_report(train_model)
    corr = correlate_var_vs_channel_zero(
        train_model,
        pairs,
        input_h=input_h,
        input_w=input_w,
        device=device,
    )
    gt90 = channels_gt90_negative(
        train_model,
        pairs,
        input_h=input_h,
        input_w=input_w,
        device=device,
        threshold=args.neg_threshold,
    )

    oracle_psnr: float | None = None
    leaky: dict[str, float] | None = None
    oracle_subset_baseline: float | None = None
    if not args.skip_oracle:
        oracle_pairs = pairs[: args.oracle_images]
        oracle_subset_baseline = measure_deploy_psnr(
            deploy_model,
            oracle_pairs,
            input_h=input_h,
            input_w=input_w,
            out_h=out_h,
            out_w=out_w,
            device=device,
        )
        oracle_psnr = oracle_no_body_relu_psnr(
            Path(args.weight),
            oracle_pairs,
            input_h=input_h,
            input_w=input_w,
            out_h=out_h,
            out_w=out_w,
            device=device,
            identity_var_floor=args.identity_var_floor,
            scale=scale,
            num_channels=args.num_channels,
            num_blocks=args.num_blocks,
            num_conv_branches=args.num_conv_branches,
        )
        slopes = [float(x.strip()) for x in args.leaky_slopes.split(",") if x.strip()]
        leaky = leaky_relu_sweep_psnr(
            Path(args.weight),
            oracle_pairs,
            slopes=slopes,
            input_h=input_h,
            input_w=input_w,
            out_h=out_h,
            out_w=out_w,
            device=device,
            identity_var_floor=args.identity_var_floor,
            scale=scale,
            num_channels=args.num_channels,
            num_blocks=args.num_blocks,
            num_conv_branches=args.num_conv_branches,
        )

    ablation_rows: list[dict[str, Any]] | None = None
    if args.ablation:
        ablation_pairs = pairs[: args.ablation_images]
        ablation_baseline = measure_deploy_psnr(
            deploy_model,
            ablation_pairs,
            input_h=input_h,
            input_w=input_w,
            out_h=out_h,
            out_w=out_w,
            device=device,
        )
        ablation_rows = []
        for spec in args.ablation:
            block_s, ch_s = spec.split(":")
            block_index = int(block_s)
            channel = int(ch_s)
            psnr = channel_ablation_psnr(
                deploy_model,
                ablation_pairs,
                block_index=block_index,
                channel=channel,
                input_h=input_h,
                input_w=input_w,
                out_h=out_h,
                out_w=out_w,
                device=device,
            )
            ablation_rows.append(
                {
                    "block": block_index,
                    "channel": channel,
                    "psnr": psnr,
                    "baseline_psnr": ablation_baseline,
                    "delta_vs_baseline": psnr - ablation_baseline,
                }
            )

    per_channel_blocks: dict[str, list[dict[str, float]]] | None = None
    if args.per_channel_blocks:
        per_channel_blocks = {}
        for block_index in args.per_channel_blocks:
            per_channel_blocks[f"body.{block_index}"] = per_channel_block_stats(
                train_model,
                pairs,
                block_index=block_index,
                input_h=input_h,
                input_w=input_w,
                device=device,
            )

    return AuditReport(
        num_images=len(pairs),
        input_h=input_h,
        input_w=input_w,
        baseline_psnr=baseline,
        per_layer_relu=per_layer,
        global_skip=global_skip,
        identity_var=id_vars,
        var_vs_ch_zero_corr=corr,
        channels_gt90_neg=gt90,
        oracle_no_body_relu_psnr=oracle_psnr,
        leaky_relu_sweep=leaky,
        oracle_subset_baseline_psnr=oracle_subset_baseline,
        channel_ablation=ablation_rows,
        per_channel_blocks=per_channel_blocks,
    )


def format_report(report: AuditReport) -> str:
    lines = [
        f"=== ReLU audit ({report.num_images} images, LR {report.input_h}x{report.input_w}) ===",
        f"Deploy baseline PSNR vs HR: {report.baseline_psnr:.3f} dB",
        "",
        "Per-layer ReLU (train graph, spatial x channel mean):",
    ]
    for name in sorted(report.per_layer_relu):
        stats = report.per_layer_relu[name]
        lines.append(
            f"  {name:16s} pre_neg={stats.pre_neg_mean * 100:5.1f}%  "
            f"post_zero={stats.post_zero_mean * 100:5.1f}%  "
            f"ch_always_zero={stats.ch_always_zero_pct:4.1f}%  "
            f"worst_ch_zero={stats.worst_ch_zero_frac * 100:.1f}%"
        )

    lines.extend(
        [
            "",
            "Global skip rescue:",
            f"  stem post-ReLU zero: {report.global_skip['stem_post_relu_zero_mean'] * 100:.1f}%",
            f"  body post-ReLU zero: {report.global_skip['body_post_relu_zero_mean'] * 100:.1f}%",
            f"  after f+f0 zero:     {report.global_skip['after_global_skip_zero_mean'] * 100:.1f}%",
            f"  rescued:             {report.global_skip['skip_rescue_delta_zero'] * 100:.1f} pp",
            "",
            "Identity BN running_var:",
        ]
    )
    for row in report.identity_var:
        lines.append(
            f"  {row['layer']}: min={row['var_min']:.2e}  "
            f"var<1e-4={row['var_lt_1e-4']}/32  var<0.01={row['var_lt_0.01']}/32"
        )

    lines.append("")
    lines.append("var vs channel-zero correlation:")
    for row in report.var_vs_ch_zero_corr:
        lines.append(
            f"  {row['layer']}: r={row['corr_var_vs_ch_zero']:.3f}  "
            f"worst_ch={row['worst_ch']} zero={row['worst_ch_zero_frac'] * 100:.1f}%"
        )

    lines.append("")
    lines.append("Channels >90% pre-ReLU negative:")
    for layer, channels in report.channels_gt90_neg.items():
        if channels:
            summary = ", ".join(f"ch{c['channel']}({c['pre_neg_frac'] * 100:.1f}%)" for c in channels)
            lines.append(f"  {layer}: {summary}")
        else:
            lines.append(f"  {layer}: none")

    if report.oracle_no_body_relu_psnr is not None:
        lines.extend(
            [
                "",
                f"Oracle no-body-ReLU PSNR: {report.oracle_no_body_relu_psnr:.2f} dB",
            ]
        )
    if report.leaky_relu_sweep is not None:
        ref = report.oracle_subset_baseline_psnr or report.baseline_psnr
        lines.append(f"LeakyReLU sweep (no retrain, subset baseline {ref:.3f} dB):")
        for slope, psnr in report.leaky_relu_sweep.items():
            lines.append(f"  slope={slope}: {psnr:.3f} dB ({psnr - ref:+.3f})")

    if report.channel_ablation:
        lines.append("")
        lines.append("Channel ablation (zero channel at block output):")
        for row in report.channel_ablation:
            lines.append(
                f"  body.{row['block']} ch{row['channel']:02d}: "
                f"{row['psnr']:.3f} dB ({row['delta_vs_baseline']:+.3f})"
            )

    if report.per_channel_blocks:
        lines.append("")
        lines.append("Per-channel detail:")
        for layer, rows in report.per_channel_blocks.items():
            worst = max(rows, key=lambda r: r["pre_neg_frac"])
            lines.append(
                f"  {layer} worst ch{int(worst['channel']):02d}: "
                f"var={worst['identity_var']:.3e} pre_neg={worst['pre_neg_frac'] * 100:.1f}% "
                f"post_std={worst['post_std']:.3f}"
            )

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Audit ReLU sparsity and output impact on real DIV2K validation images.",
    )
    parser.add_argument("--weight", type=str, default="checkpoints/train/float/best.pth")
    parser.add_argument("--hr_dir", type=str, default="data/DIV2K_valid_HR")
    parser.add_argument("--lr_dir", type=str, default="data/DIV2K_valid_LR_bicubic/X3")
    parser.add_argument("--scale", type=int, default=cfg.model.scale)
    parser.add_argument("--input_h", type=int, default=cfg.deploy.input_h)
    parser.add_argument("--input_w", type=int, default=cfg.deploy.input_w)
    parser.add_argument("--max_images", type=int, default=100)
    parser.add_argument("--identity_var_floor", type=float, default=0.0)
    parser.add_argument("--num_channels", type=int, default=cfg.model.num_channels)
    parser.add_argument("--num_blocks", type=int, default=cfg.model.num_blocks)
    parser.add_argument("--num_conv_branches", type=int, default=cfg.model.num_conv_branches)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--neg_threshold", type=float, default=0.90)
    parser.add_argument("--output", type=str, default="artifacts/relu_audit.json")
    parser.add_argument(
        "--skip_oracle",
        action="store_true",
        help="Skip no-ReLU oracle and LeakyReLU sweep (faster).",
    )
    parser.add_argument(
        "--oracle_images",
        type=int,
        default=30,
        help="Subset size for oracle / LeakyReLU sweep.",
    )
    parser.add_argument(
        "--leaky_slopes",
        type=str,
        default="0.0,0.01,0.05,0.1,0.2",
        help="Comma-separated LeakyReLU negative slopes.",
    )
    parser.add_argument(
        "--ablation",
        nargs="*",
        default=None,
        metavar="BLOCK:CH",
        help="Channel ablation specs, e.g. 2:29 2:1 1:1",
    )
    parser.add_argument("--ablation_images", type=int, default=30)
    parser.add_argument(
        "--per_channel_blocks",
        nargs="*",
        type=int,
        default=None,
        metavar="BLOCK",
        help="Emit per-channel stats for given body block indices.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_audit(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_json_dict(), indent=2))
    print(format_report(report))
    print()
    print(f"JSON report: {output_path}")


if __name__ == "__main__":
    main()
