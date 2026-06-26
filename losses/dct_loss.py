"""8x8 DCT loss for high-frequency fidelity."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCTLoss(nn.Module):
    """L1 loss on 8x8 DCT coefficients."""

    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
        self.register_buffer("basis", self._build_dct_basis(block_size))

    def _build_dct_basis(self, n: int) -> torch.Tensor:
        basis = torch.zeros(n, n, n, n)
        for i in range(n):
            for j in range(n):
                for u in range(n):
                    for v in range(n):
                        au = 1.0 / (n**0.5) if u == 0 else (2.0 / n) ** 0.5
                        av = 1.0 / (n**0.5) if v == 0 else (2.0 / n) ** 0.5
                        basis[i, j, u, v] = (
                            au
                            * av
                            * torch.cos((2 * i + 1) * u * 3.14159265 / (2 * n))
                            * torch.cos((2 * j + 1) * v * 3.14159265 / (2 * n))
                        )
        return basis

    def _dct2(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = self.block_size
        x = F.unfold(x, kernel_size=n, stride=n)  # b, c*n*n, L
        x = x.view(b, c, n, n, -1).permute(0, 1, 4, 2, 3).contiguous()
        x = x.view(-1, n, n)
        dct = torch.einsum("ijuv,xyuv->ijxy", x, self.basis.to(x.device))
        return dct

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred_dct = self._dct2(pred)
        gt_dct = self._dct2(gt)
        return F.l1_loss(pred_dct, gt_dct)
