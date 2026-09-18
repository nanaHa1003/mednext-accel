"""Static optimization policy definitions."""

from __future__ import annotations

import torch

from .report import BackendSelections

_BLACKWELL_REGULAR = (
    (32, 128),
    (64, 64),
    (128, 32),
    (256, 16),
    (512, 8),
)


def conservative_selections(
    *,
    device_type: str,
    capability: tuple[int, int] | None,
    dtype: torch.dtype,
    input_shape: tuple[int, ...],
) -> BackendSelections:
    """Return only choices validated end-to-end for this execution context."""

    supported = (
        device_type == "cuda"
        and capability == (12, 0)
        and dtype is torch.bfloat16
        and len(input_shape) == 5
        and input_shape[0] >= 1
        and input_shape[2:] == (128, 128, 128)
    )
    if not supported:
        return BackendSelections()
    return BackendSelections(
        depthwise_regular=_BLACKWELL_REGULAR,
        depthwise_transpose=((64, 64),),
    )
