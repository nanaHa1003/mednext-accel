"""Architectures that reproduce MONAI MedNeXt checkpoint shapes."""

from __future__ import annotations

from typing import Literal

from ..checkpointing import CheckpointConfig
from ..models.config import get_mednext_v1_config
from ..models.mednext_v1 import (
    DeepSupervisionOutput,
    MedNeXtV1,
    OptimizationSource,
    _configure_optimization,
)


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
    optimization: OptimizationSource = "auto",
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
    model = MedNeXtV1(
        config,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
    )
    return _configure_optimization(model, optimization)


def _named_factory(
    variant: str,
    *,
    in_channels: int,
    out_channels: int,
    spatial_dims: Literal[2, 3],
    kernel_size: int,
    base_channels: int,
    deep_supervision: bool,
    checkpointing: CheckpointConfig | None,
    deep_supervision_output: DeepSupervisionOutput,
    approximate_gelu_eval: bool,
    optimization: OptimizationSource,
) -> MedNeXtV1:
    return _factory(
        variant,
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
        optimization=optimization,
    )


def monai_mednext_small(
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
    optimization: OptimizationSource = "auto",
) -> MedNeXtV1:
    return _named_factory(
        "small",
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
        optimization=optimization,
    )


def monai_mednext_base(
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
    optimization: OptimizationSource = "auto",
) -> MedNeXtV1:
    return _named_factory(
        "base",
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
        optimization=optimization,
    )


def monai_mednext_medium(
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
    optimization: OptimizationSource = "auto",
) -> MedNeXtV1:
    return _named_factory(
        "medium",
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
        optimization=optimization,
    )


def monai_mednext_large(
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
    optimization: OptimizationSource = "auto",
) -> MedNeXtV1:
    return _named_factory(
        "large",
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
        optimization=optimization,
    )


__all__ = [
    "monai_mednext_base",
    "monai_mednext_large",
    "monai_mednext_medium",
    "monai_mednext_small",
]
