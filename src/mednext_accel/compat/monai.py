"""Architectures that reproduce MONAI MedNeXt checkpoint shapes."""

from __future__ import annotations

from typing import Literal

from ..checkpointing import CheckpointConfig
from ..models.mednext_v1 import DeepSupervisionOutput, MedNeXtV1
from ..models.config import get_mednext_v1_config


def _factory(
    variant: str,
    *,
    in_channels: int,
    out_channels: int,
    spatial_dims: Literal[2, 3] = 3,
    kernel_size: int = 3,
    base_channels: int = 32,
    deep_supervision: bool = False,
    checkpointing: CheckpointConfig | None = None,
    deep_supervision_output: DeepSupervisionOutput = "tuple",
    approximate_gelu_eval: bool = False,
) -> MedNeXtV1:
    config = get_mednext_v1_config(
        variant,
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
        compatibility="monai",
    )
    return MedNeXtV1(
        config,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
    )


def monai_mednext_small(**kwargs: object) -> MedNeXtV1:
    return _factory("small", **kwargs)  # type: ignore[arg-type]


def monai_mednext_base(**kwargs: object) -> MedNeXtV1:
    return _factory("base", **kwargs)  # type: ignore[arg-type]


def monai_mednext_medium(**kwargs: object) -> MedNeXtV1:
    return _factory("medium", **kwargs)  # type: ignore[arg-type]


def monai_mednext_large(**kwargs: object) -> MedNeXtV1:
    return _factory("large", **kwargs)  # type: ignore[arg-type]


__all__ = [
    "monai_mednext_base",
    "monai_mednext_large",
    "monai_mednext_medium",
    "monai_mednext_small",
]
