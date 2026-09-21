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

_BLACKWELL_BATCHED_POINTWISE_COMMON = (
    (1, 32, 128, 128, 128),
    (32, 3, 128, 128, 128),
    (32, 96, 64, 64, 64),
    (64, 32, 64, 64, 64),
    (64, 128, 64, 64, 64),
    (64, 256, 32, 32, 32),
    (96, 64, 64, 64, 64),
    (128, 256, 32, 32, 32),
    (128, 512, 32, 32, 32),
    (192, 64, 64, 64, 64),
    (256, 128, 32, 32, 32),
    (256, 512, 16, 16, 16),
    (384, 64, 63, 63, 63),
    (512, 128, 32, 32, 32),
    (512, 256, 16, 16, 16),
)
_BLACKWELL_BATCHED_POINTWISE_EXTRA = {
    2: (
        (32, 64, 128, 128, 128),
        (64, 32, 128, 128, 128),
        (64, 192, 64, 64, 64),
        (128, 32, 127, 127, 127),
        (128, 64, 32, 32, 32),
        (128, 384, 63, 63, 63),
        (256, 1024, 16, 16, 16),
        (512, 2048, 8, 8, 8),
        (512, 2048, 15, 15, 15),
        (1024, 256, 16, 16, 16),
        (1024, 512, 8, 8, 8),
        (2048, 512, 8, 8, 8),
    ),
    4: (
        (32, 64, 128, 128, 128),
        (64, 32, 128, 128, 128),
        (64, 192, 64, 64, 64),
        (128, 32, 127, 127, 127),
        (256, 1024, 16, 16, 16),
        (256, 1024, 31, 31, 31),
        (512, 2048, 8, 8, 8),
        (1024, 128, 31, 31, 31),
        (1024, 256, 16, 16, 16),
        (2048, 512, 8, 8, 8),
    ),
    6: (
        (128, 384, 63, 63, 63),
        (256, 1024, 31, 31, 31),
    ),
}


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
    batch_size = input_shape[0]
    pointwise = ()
    if batch_size in _BLACKWELL_BATCHED_POINTWISE_EXTRA:
        pointwise = (
            *_BLACKWELL_BATCHED_POINTWISE_COMMON,
            *_BLACKWELL_BATCHED_POINTWISE_EXTRA[batch_size],
        )
    return BackendSelections(
        pointwise_gemm=pointwise,
        depthwise_regular=_BLACKWELL_REGULAR,
        depthwise_transpose=((64, 64),),
    )
