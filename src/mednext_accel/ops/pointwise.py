"""Opt-in GEMM execution for dense, stride-one 1x1x1 convolutions."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PointwiseShape = tuple[int, int, int, int, int]


class GemmPointwise3d(nn.Module):
    """Execute an eligible Conv3d as a matrix multiplication."""

    def __init__(
        self,
        conv: nn.Conv3d,
        selected_shapes: Iterable[PointwiseShape] | None = None,
    ) -> None:
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.selected_shapes = (
            None if selected_shapes is None else frozenset(selected_shapes)
        )
        self.train(conv.training)

    def forward(self, x: Tensor) -> Tensor:
        unbatched = x.ndim == 4
        if unbatched:
            x = x.unsqueeze(0)
        spatial = x.shape[2:]
        shape_key = (self.in_channels, self.out_channels, *spatial)
        if self.selected_shapes is not None and shape_key not in self.selected_shapes:
            output = F.conv3d(x, self.weight, self.bias)
            return output.squeeze(0) if unbatched else output

        flattened = x.flatten(2)
        weight = self.weight.flatten(1)
        if x.shape[0] == 1:
            if self.bias is None:
                output = torch.mm(weight, flattened[0])
            else:
                output = torch.addmm(self.bias[:, None], weight, flattened[0])
            output = output.unsqueeze(0)
        else:
            output = torch.matmul(weight, flattened)
            if self.bias is not None:
                output = output + self.bias.to(output.dtype)[None, :, None]
        output = output.reshape(x.shape[0], self.out_channels, *spatial)
        return output.squeeze(0) if unbatched else output


def _eligible(conv: nn.Module) -> bool:
    return (
        type(conv) is nn.Conv3d
        and conv.kernel_size == (1, 1, 1)
        and conv.stride == (1, 1, 1)
        and conv.padding == (0, 0, 0)
        and conv.dilation == (1, 1, 1)
        and conv.groups == 1
        and conv.padding_mode == "zeros"
    )


def replace_pointwise_convs(
    model: nn.Module,
    selected_shapes: Iterable[PointwiseShape] | None = None,
) -> int:
    """Replace eligible descendants in place and return the replacement count."""

    count = 0
    for name, child in list(model.named_children()):
        if _eligible(child):
            setattr(model, name, GemmPointwise3d(child, selected_shapes))
            count += 1
        else:
            count += replace_pointwise_convs(child, selected_shapes)
    return count
