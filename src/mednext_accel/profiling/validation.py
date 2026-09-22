"""Component numerical validation, independent from policy acceptance."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

# Error scratch stays bounded even when one activation occupies several GiB.
_CHUNK_ELEMENTS = 1 << 20


def _chunk_metrics(actual: Tensor, expected: Tensor):
    import torch

    got, want = actual.float(), expected.float()
    difference = got - want
    finite = torch.isfinite(got).all() & torch.isfinite(want).all()
    # FP64 accumulation avoids overflow and keeps the relative-L2 criterion
    # independent of chunk boundaries. Tensor differences retain FP32 semantics.
    error_norm = torch.linalg.vector_norm(difference, dtype=torch.float64)
    reference_norm = torch.linalg.vector_norm(want, dtype=torch.float64)
    maximum = difference.abs().max()
    return finite, error_norm.square(), reference_norm.square(), maximum


def _component_metrics(name: str, actual: Tensor, expected: Tensor):
    import torch

    reason = None
    relative = maximum = None
    if actual.shape != expected.shape:
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        reason = f"{name}: shape mismatch"
    else:
        got, want = actual.detach().reshape(-1), expected.detach().reshape(-1)
        finite = torch.ones((), dtype=torch.bool, device=got.device)
        error_squared = torch.zeros((), dtype=torch.float64, device=got.device)
        reference_squared = torch.zeros_like(error_squared)
        max_error = torch.zeros((), dtype=torch.float32, device=got.device)
        for start in range(0, got.numel(), _CHUNK_ELEMENTS):
            chunk_finite, error, reference, peak = _chunk_metrics(
                got[start : start + _CHUNK_ELEMENTS], want[start : start + _CHUNK_ELEMENTS]
            )
            finite &= chunk_finite
            error_squared += error
            reference_squared += reference
            max_error = torch.maximum(max_error, peak)
        finite = bool(finite)
        if not finite:
            reason = f"{name}: non-finite values"
        else:
            relative = (error_squared.sqrt() / reference_squared.sqrt().clamp_min(1e-12)).item()
            maximum = max_error.item()
            if not isfinite(relative) or not isfinite(maximum):
                reason = f"{name}: non-finite error metrics"
                relative = relative if isfinite(relative) else None
                maximum = maximum if isfinite(maximum) else None
            elif relative >= 0.02:
                reason = f"{name}: relative L2 error must be below 0.02"
    return {"finite": finite, "relative_l2": relative, "max_absolute": maximum}, reason


def validate_components(
    actual: Mapping[str, Tensor], expected: Mapping[str, Tensor]
) -> dict[str, object]:
    """Compare components with bounded error scratch and return only scalars.

    The 2% relative-L2 threshold tolerates different BF16 reduction orders near
    zero without allowing a bad component to hide behind a larger tensor.
    Component and chunk frames release their temporary tensors before the next
    comparison. Callers should supply one component at a time for large tensors.
    """
    import torch

    if not actual or actual.keys() != expected.keys():
        raise ValueError("validation requires the same nonempty component names")
    metrics = {}
    reason = None
    with torch.no_grad():
        for name in sorted(expected):
            metrics[name], component_reason = _component_metrics(name, actual[name], expected[name])
            if reason is None:
                reason = component_reason
    return {
        "valid": reason is None,
        "validator": "component-relative-l2-v1",
        "validation_metrics": metrics,
        "rejection_reason": reason,
    }
