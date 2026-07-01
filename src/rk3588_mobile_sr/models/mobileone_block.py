"""MobileOne-style re-parameterization block for RK3588 NPU."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileOneBlock(nn.Module):
    """MobileOne block with train-time multi-branch and deploy-time single 3x3."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_conv_branches: int = 4,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_conv_branches = num_conv_branches
        self.inference_mode = inference_mode

        self.kernel_size = 3
        self.padding = 1

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=self.padding,
                bias=True,
            )
        else:
            self.conv_branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=self.kernel_size,
                            stride=stride,
                            padding=self.padding,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_channels),
                    )
                    for _ in range(num_conv_branches)
                ]
            )

            self.conv_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

            self.identity = (
                nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            )

            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return F.relu(self.reparam_conv(x))

        out = 0.0
        for branch in self.conv_branches:
            out += branch(x)
        out += self.conv_1x1(x)
        if self.identity is not None:
            out += self.identity(x)
        return self.relu(out)

    def reparameterize(self, identity_var_floor: float = 0.0) -> None:
        """Fuse multi-branch weights into a single Conv2d."""
        if self.inference_mode:
            return

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        deploy_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        ).to(device=device, dtype=dtype)

        kernel = torch.zeros_like(deploy_conv.weight.data)
        bias = torch.zeros(self.out_channels, device=device, dtype=dtype)

        for branch in self.conv_branches:
            conv, bn = branch[0], branch[1]
            k, b = self._fuse_conv_bn(conv, bn)
            kernel += k
            bias += b

        k1x1, b1x1 = self._fuse_conv_bn(self.conv_1x1[0], self.conv_1x1[1])
        kernel[:, :, 1:2, 1:2] += k1x1
        bias += b1x1

        if self.identity is not None:
            k_id, b_id = self._fuse_bn(self.identity, var_floor=identity_var_floor)
            kernel_eye = torch.zeros(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
                device=device,
                dtype=dtype,
            )
            mid = self.kernel_size // 2
            for i in range(min(self.out_channels, self.in_channels)):
                kernel_eye[i, i, mid, mid] = 1.0
            kernel += k_id.view(-1, 1, 1, 1) * kernel_eye
            bias += b_id

        deploy_conv.weight.data = kernel
        deploy_conv.bias.data = bias

        self.reparam_conv = deploy_conv

        del self.conv_branches
        del self.conv_1x1
        if self.identity is not None:
            del self.identity
        self.inference_mode = True

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
        std = (bn.running_var + bn.eps).sqrt()
        t = bn.weight / std
        t = t.view(-1, 1, 1, 1)
        kernel = conv.weight * t
        if conv.bias is not None:
            bias = bn.bias - bn.running_mean * bn.weight / std + conv.bias
        else:
            bias = bn.bias - bn.running_mean * bn.weight / std
        return kernel, bias

    @staticmethod
    def _fuse_bn(bn: nn.BatchNorm2d, *, var_floor: float = 0.0):
        running_var = bn.running_var
        if var_floor > 0.0:
            running_var = running_var.clamp_min(var_floor)
        std = (running_var + bn.eps).sqrt()
        weight = bn.weight / std
        bias = bn.bias - bn.running_mean * bn.weight / std
        return weight, bias
