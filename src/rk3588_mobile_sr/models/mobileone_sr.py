"""MobileOne-style lightweight SISR model for RK3588."""

import torch
import torch.nn as nn

from .mobileone_block import MobileOneBlock


class MobileOneSR(nn.Module):
    """360p -> 1080p 3x SISR with MobileOne blocks and PixelShuffle upsample."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_channels: int = 32,
        num_blocks: int = 8,
        scale: int = 3,
        num_conv_branches: int = 4,
        inference_mode: bool = False,
        negative_slope: float = 0.1,
    ):
        super().__init__()
        self.scale = scale
        self.num_channels = num_channels
        self.negative_slope = negative_slope

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

        self.body = nn.Sequential(
            *[
                MobileOneBlock(
                    num_channels,
                    num_channels,
                    num_conv_branches=num_conv_branches,
                    inference_mode=inference_mode,
                    negative_slope=negative_slope,
                )
                for _ in range(num_blocks)
            ]
        )

        self.out_conv = nn.Conv2d(
            num_channels, out_channels * scale * scale, kernel_size=3, padding=1
        )
        self.clip = nn.Hardtanh(min_val=0.0, max_val=255.0)
        self.upsample = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.stem(x)
        f = self.body(f0)
        f = f + f0  # global feature skip
        out = self.out_conv(f)
        out = self.clip(out)
        out = self.upsample(out)
        return out

    def switch_to_deploy(self, identity_var_floor: float = 0.0) -> None:
        """Fuse all MobileOne blocks into deploy mode."""
        for m in self.modules():
            if isinstance(m, MobileOneBlock):
                m.reparameterize(identity_var_floor=identity_var_floor)
