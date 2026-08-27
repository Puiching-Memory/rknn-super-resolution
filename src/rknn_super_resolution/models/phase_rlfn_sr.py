"""RGA-bicubic residual SR with a hardware-friendly local feature network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

type SRInput = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class ResidualLocalBlock(nn.Module):
    """BN-free two-convolution residual block for fidelity-oriented SR."""

    def __init__(self, channels: int, *, negative_slope: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.leaky_relu(
            self.conv1(x),
            negative_slope=self.negative_slope,
            inplace=True,
        )
        return x + self.conv2(residual)


class PhaseRLFNCore(nn.Module):
    """Quantizable residual core; resize, RGA and clipping stay outside it."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_channels: int = 32,
        num_blocks: int = 4,
        scale: int = 3,
        phase_factor: int = 2,
        codec_feature_channels: int = 96,
        codec_project_channels: int = 16,
        codec_upsample_factor: int = 4,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        if phase_factor < 1:
            raise ValueError("phase_factor must be positive")
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if codec_feature_channels < 1 or codec_project_channels < 1:
            raise ValueError("codec feature channel counts must be positive")
        if codec_upsample_factor < 1:
            raise ValueError("codec_upsample_factor must be positive")

        self.scale = scale
        self.phase_factor = phase_factor
        self.core_scale = scale * phase_factor
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.codec_feature_channels = codec_feature_channels
        self.codec_project_channels = codec_project_channels
        self.codec_upsample_factor = codec_upsample_factor
        self.negative_slope = negative_slope

        core_in_channels = in_channels * phase_factor * phase_factor
        self.stem = nn.Conv2d(core_in_channels, num_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                ResidualLocalBlock(num_channels, negative_slope=negative_slope)
                for _ in range(num_blocks)
            ]
        )
        self.feature_fuse = nn.Conv2d(
            num_channels * num_blocks,
            num_channels,
            kernel_size=1,
        )

        codec_expand_channels = (
            codec_project_channels * codec_upsample_factor * codec_upsample_factor
        )
        self.codec_expand = nn.Conv2d(
            codec_feature_channels,
            codec_expand_channels,
            kernel_size=1,
            bias=False,
        )
        self.codec_shuffle = nn.PixelShuffle(codec_upsample_factor)
        self.codec_fuse = nn.Conv2d(
            codec_project_channels,
            num_channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.codec_fuse.weight)

        core_out_channels = out_channels * self.core_scale * self.core_scale
        self.residual_head = nn.Conv2d(num_channels, core_out_channels, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @property
    def core_in_channels(self) -> int:
        return self.stem.in_channels

    @property
    def core_out_channels(self) -> int:
        return self.residual_head.out_channels

    def _codec_context(
        self,
        codec_feature: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        context = self.codec_shuffle(self.codec_expand(codec_feature))
        return context[..., :target_h, :target_w]

    def forward(
        self,
        phases: torch.Tensor,
        codec_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict signed phase residuals with optional codec context."""
        shallow = F.leaky_relu(
            self.stem(phases),
            negative_slope=self.negative_slope,
            inplace=True,
        )
        if codec_feature is not None:
            codec = self._codec_context(
                codec_feature,
                phases.shape[-2],
                phases.shape[-1],
            )
            shallow = shallow + self.codec_fuse(codec)

        feature = shallow
        distilled: list[torch.Tensor] = []
        for block in self.blocks:
            feature = block(feature)
            distilled.append(feature)
        feature = self.feature_fuse(torch.cat(distilled, dim=1)) + shallow
        return self.residual_head(feature)


class PhaseRLFNSR(nn.Module):
    """BN-free phase RLFN with an optional MLVC decoder-feature adapter.

    The image path always works on its own. When the current MLVC decoder feature
    is available, a zero-initialized adapter adds codec-aware temporal context.
    RGA supplies the bicubic base outside the exported NPU core.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_channels: int = 32,
        num_blocks: int = 4,
        scale: int = 3,
        phase_factor: int = 2,
        codec_feature_channels: int = 96,
        codec_project_channels: int = 16,
        codec_upsample_factor: int = 4,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.phase_factor = phase_factor
        self.core_scale = scale * phase_factor
        self._num_channels = num_channels
        self._num_blocks = num_blocks
        self._codec_feature_channels = codec_feature_channels
        self._codec_project_channels = codec_project_channels
        self._codec_upsample_factor = codec_upsample_factor
        self._negative_slope = negative_slope
        self._core_in_channels = in_channels * phase_factor * phase_factor
        self._core_out_channels = out_channels * self.core_scale * self.core_scale
        self.input_unshuffle = nn.PixelUnshuffle(phase_factor)
        self.core = PhaseRLFNCore(
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=num_channels,
            num_blocks=num_blocks,
            scale=scale,
            phase_factor=phase_factor,
            codec_feature_channels=codec_feature_channels,
            codec_project_channels=codec_project_channels,
            codec_upsample_factor=codec_upsample_factor,
            negative_slope=negative_slope,
        )
        self.output_shuffle = nn.PixelShuffle(self.core_scale)
        self.clip = nn.Hardtanh(min_val=0.0, max_val=255.0)

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def codec_feature_channels(self) -> int:
        return self._codec_feature_channels

    @property
    def codec_project_channels(self) -> int:
        return self._codec_project_channels

    @property
    def codec_upsample_factor(self) -> int:
        return self._codec_upsample_factor

    @property
    def negative_slope(self) -> float:
        return self._negative_slope

    @property
    def core_in_channels(self) -> int:
        return self._core_in_channels

    @property
    def core_out_channels(self) -> int:
        return self._core_out_channels

    def forward_core(
        self,
        phases: torch.Tensor,
        codec_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if codec_feature is None:
            return self.core(phases)
        return self.core(phases, codec_feature)

    def bicubic_base(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False),
            0.0,
            255.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        codec_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.bicubic_base(x)
        phases = self.input_unshuffle(x)
        residual = self.output_shuffle(self.forward_core(phases, codec_feature))
        return self.clip(base + residual)

    def switch_to_deploy(self) -> None:
        """The BN-free residual graph is already in deploy form."""


def split_sr_input(model_input: SRInput) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(model_input, torch.Tensor):
        return model_input, None
    if len(model_input) != 2:
        raise ValueError("SR input tuple must contain current frame and codec feature")
    return model_input


def forward_sr(model: nn.Module, model_input: SRInput) -> torch.Tensor:
    """Run SR with an optional MLVC decoder feature."""
    current, codec_feature = split_sr_input(model_input)
    if codec_feature is None:
        return model(current)
    return model(current, codec_feature)
