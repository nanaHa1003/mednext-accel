"""Lazy dispatch for optional Triton depthwise training operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn


def _backend() -> Any:
    try:
        from ._triton import depthwise
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "depthwise acceleration requires the 'accelerated' extra and a "
            "PyTorch release that provides torch.library.triton_op"
        ) from error
    return depthwise


def replace_depthwise_convs(
    model: nn.Module,
    *,
    include_stride2_input_grad: bool = False,
    policy: Mapping[str, object] | None = None,
) -> int:
    """Replace eligible descendants while preserving parameters and key paths."""

    backend = _backend()
    return backend.replace_highres_depthwise_convs(
        model,
        include_stride2_input_grad=include_stride2_input_grad,
        policy=policy,
    )
