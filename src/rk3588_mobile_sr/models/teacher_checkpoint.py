"""Load teacher checkpoints (BasicSR / plain state_dict)."""

from __future__ import annotations

from pathlib import Path

import torch


def load_teacher_state_dict(weight_path: str | Path) -> dict[str, torch.Tensor]:
    """Load a teacher ``state_dict`` from BasicSR or raw PyTorch checkpoints."""
    raw = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        if "params" in raw:
            state = raw["params"]
        elif "params_ema" in raw:
            state = raw["params_ema"]
        elif "state_dict" in raw:
            state = raw["state_dict"]
        else:
            state = raw
    else:
        raise TypeError(f"Unsupported checkpoint format in {weight_path}")

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint in {weight_path} did not contain a state dict")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        name = key
        if name.startswith("module."):
            name = name.removeprefix("module.")
        normalized[name] = value
    return normalized
