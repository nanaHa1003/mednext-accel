"""Reference Global Response Normalization for NCDHW tensors."""

from __future__ import annotations

import math

import torch
from torch import nn


def global_response_norm3d_reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply the MedNeXt v2 channel-sum GRN formula using PyTorch operations."""

    if x.ndim != 5:
        raise ValueError("GRN expects an NCDHW tensor")
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    norm = torch.linalg.vector_norm(work, dim=(2, 3, 4), keepdim=True)
    ratio = norm / (norm.sum(dim=1, keepdim=True) + eps)
    output = work + gamma.to(work.dtype) * work * ratio + beta.to(work.dtype)
    return output.to(x.dtype)


class GlobalResponseNorm3d(nn.Module):
    """Reference GRN layer with zero-initialized affine parameters."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return global_response_norm3d_reference(x, self.gamma, self.beta, self.eps)
