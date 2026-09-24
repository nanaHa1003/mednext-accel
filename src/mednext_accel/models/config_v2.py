"""Immutable architecture configuration for MedNeXt v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MedNeXtV2Variant = Literal["base", "wide"]

_BLOCK_COUNTS = (3, 4, 8, 8, 8, 8, 8, 4, 3)
_EXPANSION_RATIOS = (3, 4, 8, 8, 8, 8, 8, 4, 3)
_DOWNSAMPLE_EXPANSION_RATIOS = (4, 8, 8, 8)
_UPSAMPLE_EXPANSION_RATIOS = (8, 8, 4, 3)
_BASE_CHANNELS: dict[MedNeXtV2Variant, int] = {"base": 32, "wide": 64}


def _validate_positive_integer_entries(name: str, values: tuple[object, ...]) -> None:
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"{name} entries must be positive integers")


@dataclass(frozen=True, slots=True)
class MedNeXtV2Config:
    """Complete, YAML-friendly description of a MedNeXt v2 architecture."""

    variant: MedNeXtV2Variant
    in_channels: int
    out_channels: int
    base_channels: int
    kernel_size: int
    block_counts: tuple[int, ...]
    expansion_ratios: tuple[int, ...]
    downsample_expansion_ratios: tuple[int, ...]
    upsample_expansion_ratios: tuple[int, ...]
    deep_supervision: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_counts", tuple(self.block_counts))
        object.__setattr__(self, "expansion_ratios", tuple(self.expansion_ratios))
        object.__setattr__(
            self,
            "downsample_expansion_ratios",
            tuple(self.downsample_expansion_ratios),
        )
        object.__setattr__(
            self,
            "upsample_expansion_ratios",
            tuple(self.upsample_expansion_ratios),
        )

        if self.variant not in ("base", "wide"):
            raise ValueError("variant must be 'base' or 'wide'")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if self.kernel_size != 3:
            raise ValueError("MedNeXt v2 kernel_size must be 3")
        if len(self.block_counts) != 9 or len(self.expansion_ratios) != 9:
            raise ValueError("MedNeXt v2 requires nine block stages")
        if len(self.downsample_expansion_ratios) != 4 or len(self.upsample_expansion_ratios) != 4:
            raise ValueError("MedNeXt v2 requires four downsample and upsample expansion ratios")
        for name in (
            "block_counts",
            "expansion_ratios",
            "downsample_expansion_ratios",
            "upsample_expansion_ratios",
        ):
            _validate_positive_integer_entries(name, getattr(self, name))


def get_mednext_v2_config(
    variant: str,
    *,
    in_channels: int,
    out_channels: int,
    deep_supervision: bool = False,
) -> MedNeXtV2Config:
    """Return the published MedNeXt v2 Base or Wide configuration."""

    normalized = variant.lower()
    if normalized not in _BASE_CHANNELS:
        raise ValueError(f"unknown MedNeXt v2 variant {variant!r}; expected 'base' or 'wide'")

    typed_variant = normalized  # narrowed by the membership check above
    return MedNeXtV2Config(
        variant=typed_variant,  # type: ignore[arg-type]
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=_BASE_CHANNELS[typed_variant],  # type: ignore[index]
        kernel_size=3,
        block_counts=_BLOCK_COUNTS,
        expansion_ratios=_EXPANSION_RATIOS,
        downsample_expansion_ratios=_DOWNSAMPLE_EXPANSION_RATIOS,
        upsample_expansion_ratios=_UPSAMPLE_EXPANSION_RATIOS,
        deep_supervision=deep_supervision,
    )
