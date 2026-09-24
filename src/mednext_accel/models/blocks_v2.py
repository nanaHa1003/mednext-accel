"""Pure-PyTorch building blocks used by MedNeXt v2."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..ops.grn import GlobalResponseNorm3d
from .blocks import EvalModeGELU


class MedNeXtV2Block(nn.Module):
    """A MedNeXt v2 block with depthwise convolution and GRN expansion."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        residual: bool = True,
        checkpoint_expansion: bool = False,
    ) -> None:
        super().__init__()
        self.use_residual = residual
        self.checkpoint_expansion = checkpoint_expansion
        expanded_channels = in_channels * expansion_ratio

        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
        )
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.expand = nn.Conv3d(in_channels, expanded_channels, kernel_size=1)
        self.activation = EvalModeGELU(approximate_eval=False)
        self.grn = GlobalResponseNorm3d(expanded_channels)
        self.project = nn.Conv3d(expanded_channels, out_channels, kernel_size=1)

    def _expansion_forward(self, x: Tensor) -> Tensor:
        return self.project(self.grn(self.activation(self.expand(x))))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(self.depthwise(x))
        if self.checkpoint_expansion and self.training and torch.is_grad_enabled():
            x = checkpoint(self._expansion_forward, x, use_reentrant=False)
        else:
            x = self._expansion_forward(x)
        return x + residual if self.use_residual else x


class MedNeXtV2DownBlock(MedNeXtV2Block):
    """A stride-two MedNeXt v2 downsampling block."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        residual: bool = True,
        checkpoint_expansion: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            residual=False,
            checkpoint_expansion=checkpoint_expansion,
        )
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
        )
        self.residual = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2)
            if residual
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        output = super().forward(x)
        if self.residual is not None:
            output = output + self.residual(x)
        return output


class MedNeXtV2UpBlock(MedNeXtV2Block):
    """A stride-two transposed-convolution MedNeXt v2 upsampling block."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        residual: bool = True,
        checkpoint_expansion: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            residual=False,
            checkpoint_expansion=checkpoint_expansion,
        )
        self.depthwise = nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
        )
        self.residual = (
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=2)
            if residual
            else None
        )
        self._pad = (0, 1) * 3

    def forward(self, x: Tensor) -> Tensor:
        output = F.pad(super().forward(x), self._pad)
        if self.residual is not None:
            output = output + F.pad(self.residual(x), self._pad)
        return output


__all__ = ["MedNeXtV2Block", "MedNeXtV2DownBlock", "MedNeXtV2UpBlock"]
