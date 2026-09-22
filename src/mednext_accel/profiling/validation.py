"""Component numerical validation, independent from policy acceptance."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def validate_components(
    actual: Mapping[str, Tensor], expected: Mapping[str, Tensor]
) -> dict[str, object]:
    """Compare named tensors independently using finite values and relative L2.

    The 2% relative-L2 threshold tolerates different BF16 reduction orders near
    zero without allowing a bad component to hide behind a larger tensor.
    Returned metrics contain only JSON primitives and retain no tensors.
    """
    import torch

    if not actual or actual.keys() != expected.keys():
        raise ValueError("validation requires the same nonempty component names")
    metrics = {}
    reason = None
    with torch.no_grad():
        for name in sorted(expected):
            got, want = actual[name].detach().float(), expected[name].detach().float()
            finite = bool(torch.isfinite(got).all() and torch.isfinite(want).all())
            relative = maximum = None
            component_reason = None
            if got.shape != want.shape:
                component_reason = f"{name}: shape mismatch"
            elif not finite:
                component_reason = f"{name}: non-finite values"
            else:
                difference = got - want
                relative = (difference.norm() / want.norm().clamp_min(1e-12)).item()
                maximum = difference.abs().max().item() if difference.numel() else 0.0
                if not isfinite(relative) or not isfinite(maximum):
                    component_reason = f"{name}: non-finite error metrics"
                    relative = relative if isfinite(relative) else None
                    maximum = maximum if isfinite(maximum) else None
                elif relative >= 0.02:
                    component_reason = f"{name}: relative L2 error must be below 0.02"
            metrics[name] = {"finite": finite, "relative_l2": relative, "max_absolute": maximum}
            if reason is None:
                reason = component_reason
    return {
        "valid": reason is None,
        "validator": "component-relative-l2-v1",
        "validation_metrics": metrics,
        "rejection_reason": reason,
    }
