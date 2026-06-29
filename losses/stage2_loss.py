"""Combined Stage-2 fidelity + distillation loss with component breakdown."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from losses import CharbonnierLoss, ConfidenceWeightedKDLoss, DCTLoss


@dataclass(frozen=True)
class Stage2LossOutput:
    total: torch.Tensor
    charbonnier: torch.Tensor
    dct: torch.Tensor
    kd: torch.Tensor
    lambda_dct: float
    lambda_kd: float

    def log_dict(self) -> dict[str, float]:
        dct_w = self.lambda_dct * self.dct
        kd_w = self.lambda_kd * self.kd
        return {
            "train/loss_charbonnier": float(self.charbonnier.detach()),
            "train/loss_dct": float(self.dct.detach()),
            "train/loss_kd": float(self.kd.detach()),
            "train/loss_dct_weighted": float(dct_w.detach()),
            "train/loss_kd_weighted": float(kd_w.detach()),
            "train/loss_total": float(self.total.detach()),
        }


class Stage2Loss(nn.Module):
    """Charbonnier + weighted DCT + confidence-weighted KD."""

    def __init__(self, *, lambda_dct: float = 0.02, lambda_kd: float = 0.03) -> None:
        super().__init__()
        self.lambda_dct = lambda_dct
        self.lambda_kd = lambda_kd
        self.charbonnier = CharbonnierLoss()
        self.dct = DCTLoss()
        self.kd = ConfidenceWeightedKDLoss()

    def forward(
        self,
        pred: torch.Tensor,
        hr: torch.Tensor,
        teacher: torch.Tensor,
    ) -> Stage2LossOutput:
        l_charb = self.charbonnier(pred, hr)
        l_dct = self.dct(pred, hr)
        l_kd = self.kd(pred, teacher, hr)
        total = l_charb + self.lambda_dct * l_dct + self.lambda_kd * l_kd
        return Stage2LossOutput(
            total=total,
            charbonnier=l_charb,
            dct=l_dct,
            kd=l_kd,
            lambda_dct=self.lambda_dct,
            lambda_kd=self.lambda_kd,
        )
