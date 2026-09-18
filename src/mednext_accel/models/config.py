"""Immutable architecture configuration for MedNeXt v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MedNeXtVariant = Literal["small", "base", "medium", "large"]
CompatibilityFamily = Literal["official", "monai"]

_BLOCK_COUNTS: dict[MedNeXtVariant, tuple[int, ...]] = {
    "small": (2, 2, 2, 2, 2, 2, 2, 2, 2),
    "base": (2, 2, 2, 2, 2, 2, 2, 2, 2),
    "medium": (3, 4, 4, 4, 4, 4, 4, 4, 3),
    "large": (3, 4, 8, 8, 8, 8, 8, 4, 3),
}

_EXPANSION_RATIOS: dict[MedNeXtVariant, tuple[int, ...]] = {
    "small": (2, 2, 2, 2, 2, 2, 2, 2, 2),
    "base": (2, 3, 4, 4, 4, 4, 4, 3, 2),
    "medium": (2, 3, 4, 4, 4, 4, 4, 3, 2),
    "large": (3, 4, 8, 8, 8, 8, 8, 4, 3),
}


@dataclass(frozen=True, slots=True)
class MedNeXtV1Config:
    """Complete, serializable description of a MedNeXt v1 architecture."""

    variant: MedNeXtVariant
    spatial_dims: Literal[2, 3]
    in_channels: int
    out_channels: int
    kernel_size: int
    base_channels: int
    block_counts: tuple[int, ...]
    expansion_ratios: tuple[int, ...]
    downsample_expansion_ratios: tuple[int, ...]
    compatibility: CompatibilityFamily
    residual_blocks: bool = True
    residual_resampling: bool = True
    deep_supervision: bool = False

    def __post_init__(self) -> None:
        if self.spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if len(self.block_counts) != 9 or len(self.expansion_ratios) != 9:
            raise ValueError("MedNeXt v1 requires nine block stages")
        if len(self.downsample_expansion_ratios) != 4:
            raise ValueError("MedNeXt v1 requires four downsample expansion ratios")


def get_mednext_v1_config(
    variant: str,
    *,
    in_channels: int,
    out_channels: int,
    spatial_dims: Literal[2, 3] = 3,
    kernel_size: int = 3,
    base_channels: int = 32,
    deep_supervision: bool = False,
    compatibility: CompatibilityFamily = "official",
) -> MedNeXtV1Config:
    """Return a validated configuration for a published MedNeXt v1 variant."""

    normalized = variant.lower()
    if normalized not in _BLOCK_COUNTS:
        choices = ", ".join(_BLOCK_COUNTS)
        raise ValueError(f"unknown MedNeXt v1 variant {variant!r}; expected one of {choices}")
    if compatibility not in ("official", "monai"):
        raise ValueError("compatibility must be 'official' or 'monai'")

    typed_variant = normalized  # narrowed by the membership check above
    expansion_ratios = _EXPANSION_RATIOS[typed_variant]  # type: ignore[index]
    if compatibility == "official":
        downsample_expansion_ratios = expansion_ratios[1:5]
    else:
        downsample_expansion_ratios = expansion_ratios[:4]

    return MedNeXtV1Config(
        variant=typed_variant,  # type: ignore[arg-type]
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        base_channels=base_channels,
        block_counts=_BLOCK_COUNTS[typed_variant],  # type: ignore[index]
        expansion_ratios=expansion_ratios,
        downsample_expansion_ratios=downsample_expansion_ratios,
        compatibility=compatibility,
        deep_supervision=deep_supervision,
    )

