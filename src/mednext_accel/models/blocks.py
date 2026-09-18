"""Pure-PyTorch building blocks used by MedNeXt v1."""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ConvType: TypeAlias = type[nn.Conv2d] | type[nn.Conv3d]
ConvTransposeType: TypeAlias = type[nn.ConvTranspose2d] | type[nn.ConvTranspose3d]


def _conv_type(spatial_dims: int) -> ConvType:
    if spatial_dims == 2:
        return nn.Conv2d
    if spatial_dims == 3:
        return nn.Conv3d
    raise ValueError("spatial_dims must be 2 or 3")


def _conv_transpose_type(spatial_dims: int) -> ConvTransposeType:
    if spatial_dims == 2:
        return nn.ConvTranspose2d
    if spatial_dims == 3:
        return nn.ConvTranspose3d
    raise ValueError("spatial_dims must be 2 or 3")


class EvalModeGELU(nn.Module):
    """Use exact GELU in training and optionally tanh GELU in evaluation."""

    def __init__(self, approximate_eval: bool = False) -> None:
        super().__init__()
        self.approximate_eval = approximate_eval

    def forward(self, x: Tensor) -> Tensor:
        approximation = "tanh" if self.approximate_eval and not self.training else "none"
        return F.gelu(x, approximate=approximation)

    def extra_repr(self) -> str:
        return f"approximate_eval={self.approximate_eval}"


class MedNeXtBlock(nn.Module):
    """A residual MedNeXt block with a depthwise spatial convolution."""

    def __init__(
        self,
        *,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        kernel_size: int,
        residual: bool = True,
        checkpoint_expansion: bool = False,
        approximate_gelu_eval: bool = False,
    ) -> None:
        super().__init__()
        conv = _conv_type(spatial_dims)
        self.use_residual = residual
        self.checkpoint_expansion = checkpoint_expansion
        self.depthwise = conv(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.expand = conv(in_channels, in_channels * expansion_ratio, kernel_size=1)
        self.activation = EvalModeGELU(approximate_eval=approximate_gelu_eval)
        self.project = conv(in_channels * expansion_ratio, out_channels, kernel_size=1)

    def _expansion_forward(self, x: Tensor) -> Tensor:
        return self.project(self.activation(self.expand(x)))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(self.depthwise(x))
        if self.checkpoint_expansion and self.training and torch.is_grad_enabled():
            x = checkpoint(self._expansion_forward, x, use_reentrant=False)
        else:
            x = self._expansion_forward(x)
        if self.use_residual:
            x = x + residual
        return x


class MedNeXtDownBlock(MedNeXtBlock):
    """Stride-two MedNeXt downsampling block."""

    def __init__(
        self,
        *,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        kernel_size: int,
        residual: bool = True,
        checkpoint_expansion: bool = False,
        approximate_gelu_eval: bool = False,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            kernel_size=kernel_size,
            residual=False,
            checkpoint_expansion=checkpoint_expansion,
            approximate_gelu_eval=approximate_gelu_eval,
        )
        conv = _conv_type(spatial_dims)
        self.depthwise = conv(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        self.residual = (
            conv(in_channels, out_channels, kernel_size=1, stride=2) if residual else None
        )

    def forward(self, x: Tensor) -> Tensor:
        output = super().forward(x)
        if self.residual is not None:
            output = output + self.residual(x)
        return output


class MedNeXtUpBlock(MedNeXtBlock):
    """Stride-two transposed-convolution MedNeXt upsampling block."""

    def __init__(
        self,
        *,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        kernel_size: int,
        residual: bool = True,
        checkpoint_expansion: bool = False,
        approximate_gelu_eval: bool = False,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            kernel_size=kernel_size,
            residual=False,
            checkpoint_expansion=checkpoint_expansion,
            approximate_gelu_eval=approximate_gelu_eval,
        )
        conv_transpose = _conv_transpose_type(spatial_dims)
        self.depthwise = conv_transpose(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        self.residual = (
            conv_transpose(in_channels, out_channels, kernel_size=1, stride=2) if residual else None
        )
        self._pad = (1, 0) * spatial_dims

    def forward(self, x: Tensor) -> Tensor:
        output = F.pad(super().forward(x), self._pad)
        if self.residual is not None:
            output = output + F.pad(self.residual(x), self._pad)
        return output


class OutputHead(nn.Module):
    """A one-by-one transposed convolution matching official checkpoints."""

    def __init__(self, *, spatial_dims: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        conv_transpose = _conv_transpose_type(spatial_dims)
        self.conv = conv_transpose(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
