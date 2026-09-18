"""Opt-in GEMM prototype for dense, stride-one 1x1x1 Conv3d layers.

Preserves Parameter objects and state-dict keys. Apply before torch.compile,
DDP wrapping, or optimizer construction. Spatial/transposed convolutions stay
on their original implementation. No custom CUDA or autograd is used.
"""
import torch
from torch import nn


class GemmPointwise3d(nn.Module):
    def __init__(self, conv, selected_shapes=None):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.selected_shapes = (None if selected_shapes is None
                                else frozenset(selected_shapes))
        self.train(conv.training)

    def forward(self, x):
        unbatched = x.ndim == 4
        if unbatched:
            x = x.unsqueeze(0)
        spatial = x.shape[2:]
        shape_key = (self.in_channels, self.out_channels, *spatial)
        if self.selected_shapes is not None and shape_key not in self.selected_shapes:
            y = torch.nn.functional.conv3d(
                x, self.weight, self.bias, stride=1, padding=0, groups=1)
            return y.squeeze(0) if unbatched else y
        flat = x.flatten(2)
        weight = self.weight.flatten(1)
        if x.shape[0] == 1:
            if self.bias is None:
                y = torch.mm(weight, flat[0])
            else:
                y = torch.addmm(self.bias[:, None], weight, flat[0])
            y = y.unsqueeze(0)
        else:
            y = torch.matmul(weight, flat)
            if self.bias is not None:
                y = y + self.bias.to(y.dtype)[None, :, None]
        y = y.reshape(x.shape[0], self.out_channels, *spatial)
        return y.squeeze(0) if unbatched else y


def replace_pointwise_convs(model, selected_shapes=None):
    """Replace eligible children in place; optionally select GEMM by shape."""
    count = 0
    for name, child in list(model.named_children()):
        if (type(child) is nn.Conv3d and child.kernel_size == (1, 1, 1)
                and child.stride == (1, 1, 1) and child.padding == (0, 0, 0)
                and child.dilation == (1, 1, 1) and child.groups == 1
                and child.padding_mode == 'zeros'):
            setattr(model, name, GemmPointwise3d(child, selected_shapes))
            count += 1
        else:
            count += replace_pointwise_convs(child, selected_shapes)
    return count
