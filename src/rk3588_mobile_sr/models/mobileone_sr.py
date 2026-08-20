"""Phase-domain MobileOne super-resolution model optimized for RK3576."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mobileone_block import MobileOneBlock


class MobileOneSR(nn.Module):
    """E-architecture model with a low-resolution NPU core.

    Training uses the ordinary 3-channel LR to 3-channel HR contract. Deployment
    exports only :meth:`forward_core`; CPU-side phase packing is parameter-free.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_channels: int = 32,
        num_blocks: int = 6,
        scale: int = 3,
        phase_factor: int = 2,
        output_kernel_size: int = 3,
        num_conv_branches: int = 4,
        inference_mode: bool = False,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        if phase_factor < 1:
            raise ValueError("phase_factor must be positive")
        if output_kernel_size not in (1, 3):
            raise ValueError("output_kernel_size must be 1 or 3")

        self.scale = scale
        self.phase_factor = phase_factor
        self.core_scale = scale * phase_factor
        self.num_channels = num_channels
        self.negative_slope = negative_slope
        self.input_unshuffle = nn.PixelUnshuffle(phase_factor)

        core_in_channels = in_channels * phase_factor * phase_factor
        self.stem = nn.Sequential(
            nn.Conv2d(core_in_channels, num_channels, kernel_size=3, padding=1, bias=False),
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

        core_out_channels = out_channels * self.core_scale * self.core_scale
        padding = output_kernel_size // 2
        self.out_conv = nn.Conv2d(
            num_channels,
            core_out_channels,
            kernel_size=output_kernel_size,
            padding=padding,
        )
        self.clip = nn.Hardtanh(min_val=0.0, max_val=255.0)
        self.output_shuffle = nn.PixelShuffle(self.core_scale)

    @property
    def core_in_channels(self) -> int:
        return self.stem[0].in_channels

    @property
    def core_out_channels(self) -> int:
        return self.out_conv.out_channels

    def forward_core(self, phases: torch.Tensor) -> torch.Tensor:
        """Run the NPU-resident 12-channel to 108-channel graph."""
        f0 = self.stem(phases)
        f = self.body(f0)
        f = f + f0
        out = self.out_conv(f)
        return self.clip(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phases = self.input_unshuffle(x)
        return self.output_shuffle(self.forward_core(phases))

    def switch_to_deploy(self, identity_var_floor: float = 0.0) -> None:
        """Fuse all MobileOne blocks into deploy mode."""
        for m in self.modules():
            if isinstance(m, MobileOneBlock):
                m.reparameterize(identity_var_floor=identity_var_floor)
