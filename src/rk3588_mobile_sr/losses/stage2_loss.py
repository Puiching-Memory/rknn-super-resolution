"""Combined Stage-2 fidelity + perceptual + distillation loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .charbonnier import CharbonnierLoss
from .dct_loss import DCTLoss
from .dists_loss import DISTSLoss
from .kd_loss import ConfidenceWeightedKDLoss


@dataclass(frozen=True)
class Stage2LossOutput:
    total: torch.Tensor
    charbonnier: torch.Tensor
    dct: torch.Tensor
    dists: torch.Tensor
    kd: torch.Tensor
    lambda_dct: float
    lambda_dists: float
    lambda_kd: float

    def log_dict(self) -> dict[str, float]:
        dct_w = self.lambda_dct * self.dct
        dists_w = self.lambda_dists * self.dists
        kd_w = self.lambda_kd * self.kd
        return {
            "train/loss_charbonnier": float(self.charbonnier.detach()),
            "train/loss_dct": float(self.dct.detach()),
            "train/loss_dists": float(self.dists.detach()),
            "train/loss_kd": float(self.kd.detach()),
            "train/loss_dct_weighted": float(dct_w.detach()),
            "train/loss_dists_weighted": float(dists_w.detach()),
            "train/loss_kd_weighted": float(kd_w.detach()),
            "train/loss_total": float(self.total.detach()),
        }


class Stage2Loss(nn.Module):
    """Charbonnier_yuv + λ_dct·DCT_rgb + λ_dists·DISTS_rgb + λ_kd·KD."""

    def __init__(
        self,
        *,
        lambda_dct: float = 0.02,
        lambda_dists: float = 0.05,
        lambda_kd: float = 0.03,
    ) -> None:
        super().__init__()
        self.lambda_dct = lambda_dct
        self.lambda_dists = lambda_dists
        self.lambda_kd = lambda_kd
        self.charbonnier = CharbonnierLoss()
        self.dct = DCTLoss()
        self.dists = DISTSLoss()
        self.kd = ConfidenceWeightedKDLoss()

    def forward(
        self,
        pred: torch.Tensor,
        hr: torch.Tensor,
        teacher: torch.Tensor,
        *,
        colorspace: str = "rgb",
    ) -> Stage2LossOutput:
        if colorspace == "yuv":
            from rk3588_mobile_sr.data.yuv_utils import yuv444_to_rgb

            pred_rgb = yuv444_to_rgb(pred)
            hr_rgb = yuv444_to_rgb(hr)
            l_charb = self.charbonnier(pred, hr)
            l_dct = self.dct(pred_rgb, hr_rgb)
            l_dists = self.dists(pred_rgb, hr_rgb)
            l_kd = self.kd(pred_rgb, teacher, hr_rgb)
        else:
            l_charb = self.charbonnier(pred, hr)
            l_dct = self.dct(pred, hr)
            l_dists = self.dists(pred, hr)
            l_kd = self.kd(pred, teacher, hr)
        total = (
            l_charb
            + self.lambda_dct * l_dct
            + self.lambda_dists * l_dists
            + self.lambda_kd * l_kd
        )
        return Stage2LossOutput(
            total=total,
            charbonnier=l_charb,
            dct=l_dct,
            dists=l_dists,
            kd=l_kd,
            lambda_dct=self.lambda_dct,
            lambda_dists=self.lambda_dists,
            lambda_kd=self.lambda_kd,
        )
