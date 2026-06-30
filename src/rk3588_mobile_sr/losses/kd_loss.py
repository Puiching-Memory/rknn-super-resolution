"""Confidence-weighted knowledge-distillation loss."""

import torch
import torch.nn as nn


class ConfidenceWeightedKDLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 10.0,
        w_min: float = 0.1,
        w_max: float = 0.75,
    ):
        super().__init__()
        self.gamma = gamma
        self.w_min = w_min
        self.w_max = w_max

    def forward(
        self,
        pred: torch.Tensor,
        teacher: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            err = torch.abs(teacher - gt).mean(dim=1, keepdim=True)
            w = torch.exp(-self.gamma * err)
            w = torch.clamp(w, self.w_min, self.w_max)
        diff = torch.abs(pred - teacher)
        return (w * diff).mean()
