"""Charbonnier loss for SR."""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = pred - gt
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
