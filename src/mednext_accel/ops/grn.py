"""Reference Global Response Normalization for NCDHW tensors."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import nn

from ..optimization.descriptors import OperatorDescriptor
from ..optimization.policy_resolver import PolicyResolver
from .adaptive import ModelOptimizationContext, _context_from_tensor


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


class AdaptiveGlobalResponseNorm3d(nn.Module):
    """Policy-selected training GRN with native inference and fallback execution."""

    def __init__(
        self,
        module: GlobalResponseNorm3d,
        resolver: PolicyResolver,
        model_context: ModelOptimizationContext,
        role: str | None = None,
    ) -> None:
        super().__init__()
        self.gamma = module.gamma
        self.beta = module.beta
        self.eps = module.eps
        self.resolver = resolver
        self.model_context = model_context
        self.descriptor = OperatorDescriptor.normalization(
            family="global_response_norm3d", channels=self.gamma.shape[1], role=role
        )
        self.train(module.training)

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        return global_response_norm3d_reference(x, self.gamma, self.beta, self.eps)

    def _tensor_eligible(self, x: torch.Tensor) -> bool:
        return x.ndim == 5 and x.is_contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not torch.is_grad_enabled() or x.ndim != 5:
            return self._reference(x)
        context = _context_from_tensor(x, self.model_context, training=True)
        # Unlike convolution, GRN does not autocast its input before dispatch.
        context = replace(context, dtype=str(x.dtype).removeprefix("torch."))
        decision = self.resolver.resolve(self.descriptor, context, "training")
        if decision.implementation != "triton_fused_grn" or not self._tensor_eligible(x):
            return self._reference(x)
        try:
            from ._triton.grn import fused_global_response_norm3d
        except (ImportError, AttributeError):
            return self._reference(x)
        return fused_global_response_norm3d(x, self.gamma, self.beta, self.eps)


def install_adaptive_grn(
    model: nn.Module,
    *,
    resolver: PolicyResolver,
    model_context: ModelOptimizationContext,
    _prefix: str = "",
) -> int:
    """Replace reference GRN children while retaining their Parameter objects."""

    count = 0
    for name, child in list(model.named_children()):
        role = f"{_prefix}.{name}" if _prefix else name
        if type(child) is GlobalResponseNorm3d:
            setattr(model, name, AdaptiveGlobalResponseNorm3d(child, resolver, model_context, role))
            count += 1
        else:
            count += install_adaptive_grn(
                child, resolver=resolver, model_context=model_context, _prefix=role
            )
    return count
