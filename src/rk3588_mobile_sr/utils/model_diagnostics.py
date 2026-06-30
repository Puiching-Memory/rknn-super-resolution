"""MobileOneSR training diagnostics: norms, activations, deploy consistency."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn

from rk3588_mobile_sr.models.mobileone_sr import MobileOneSR


def _unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


def _group_param_name(name: str) -> str:
    if name.startswith("body."):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return f"body_{parts[1]}"
    if name.startswith("stem"):
        return "stem"
    if name.startswith("out_conv"):
        return "out_conv"
    return "other"


def collect_param_norms(model: nn.Module, *, prefix: str) -> dict[str, float]:
    """L2 norm of parameters grouped by stem / body_i / out_conv."""
    unwrap = _unwrap(model)
    sums: dict[str, float] = defaultdict(float)
    for name, param in unwrap.named_parameters():
        if not param.requires_grad:
            continue
        group = _group_param_name(name)
        sums[f"{prefix}/{group}"] += param.detach().float().norm(2).item() ** 2
    return {key: value**0.5 for key, value in sums.items()}


def collect_grad_norms(model: nn.Module) -> dict[str, float]:
    """L2 norm of gradients grouped by module."""
    unwrap = _unwrap(model)
    sums: dict[str, float] = defaultdict(float)
    for name, param in unwrap.named_parameters():
        if param.grad is None:
            continue
        group = _group_param_name(name)
        sums[f"grad_norm/{group}"] += param.grad.detach().float().norm(2).item() ** 2
    return {key: value**0.5 for key, value in sums.items()}


class ForwardDiagnosticsTracker:
    """Capture stem/body/pre-clip tensors via forward hooks."""

    def __init__(self, model: nn.Module) -> None:
        self._unwrap = _unwrap(model)
        self._storage: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register()

    def _register(self) -> None:
        def save_f0(_module, _inputs, output):
            self._storage["f0"] = output.detach()

        def save_f_body(_module, _inputs, output):
            self._storage["f_body"] = output.detach()

        def save_pre_clip(_module, _inputs, output):
            self._storage["pre_clip"] = output.detach()

        self._handles.append(self._unwrap.stem.register_forward_hook(save_f0))
        self._handles.append(self._unwrap.body.register_forward_hook(save_f_body))
        self._handles.append(self._unwrap.out_conv.register_forward_hook(save_pre_clip))

    def clear(self) -> None:
        self._storage.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._storage.clear()

    def read_stats(self) -> dict[str, float]:
        if not self._storage:
            return {}

        stats: dict[str, float] = {}
        f0 = self._storage.get("f0")
        f_body = self._storage.get("f_body")
        pre_clip = self._storage.get("pre_clip")

        if f0 is not None and f_body is not None:
            f0_mean = f0.abs().mean().clamp_min(1e-6)
            stats["model/skip_ratio"] = float((f_body.abs().mean() / f0_mean).item())

        if pre_clip is not None:
            stats["model/clip_sat_low"] = float((pre_clip <= 0.01).float().mean().item())
            stats["model/clip_sat_high"] = float((pre_clip >= 254.99).float().mean().item())
            stats["model/pre_clip_mean"] = float(pre_clip.mean().item())
            stats["model/pre_clip_std"] = float(pre_clip.std().item())

        if f_body is not None:
            dead = (f_body.abs() < 1e-3).float().mean()
            stats["model/body_dead_ratio"] = float(dead.item())

        return stats


def collect_training_diagnostics(
    model: nn.Module,
    tracker: ForwardDiagnosticsTracker,
) -> dict[str, float]:
    """Merge forward activation stats with grad/weight norms."""
    metrics = tracker.read_stats()
    metrics.update(collect_grad_norms(model))
    metrics.update(collect_param_norms(model, prefix="weight_norm"))
    tracker.clear()
    return metrics


@torch.no_grad()
def check_deploy_consistency(
    model: nn.Module,
    data_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    max_batches: int = 4,
) -> dict[str, float]:
    """Compare train-mode multi-branch forward vs fused deploy graph."""
    train_net = _unwrap(model)
    if not isinstance(train_net, MobileOneSR):
        return {}

    was_training = train_net.training
    train_net.eval()

    deploy_net = copy.deepcopy(train_net)
    deploy_net.switch_to_deploy()
    deploy_net.eval()

    max_abs: list[float] = []
    mean_abs: list[float] = []
    match_psnr: list[float] = []

    for batch_idx, (lr, _hr) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        lr = lr.to(device, non_blocking=True)
        train_out = torch.clamp(train_net(lr), 0.0, 255.0)
        deploy_out = torch.clamp(deploy_net(lr), 0.0, 255.0)
        diff = (train_out - deploy_out).abs()
        max_abs.append(float(diff.max().item()))
        mean_abs.append(float(diff.mean().item()))
        mse = torch.mean((train_out - deploy_out) ** 2).clamp_min(1e-12)
        match_psnr.append(float((10.0 * torch.log10(255.0 * 255.0 / mse)).item()))

    if was_training:
        train_net.train()

    if not max_abs:
        return {}

    return {
        "deploy/max_abs_diff": max(max_abs),
        "deploy/mean_abs_diff": sum(mean_abs) / len(mean_abs),
        "deploy/psnr_train_vs_deploy": sum(match_psnr) / len(match_psnr),
    }


@contextmanager
def forward_diagnostics(model: nn.Module):
    """Context manager that installs forward hooks for one training run."""
    tracker = ForwardDiagnosticsTracker(model)
    try:
        yield tracker
    finally:
        tracker.close()
