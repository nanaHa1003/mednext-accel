"""Stable, serializable descriptions used by optimization profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod
from typing import Literal

ExecutionPhase = Literal["training", "inference", "export"]


def _spatial_tuple(value: tuple[int, ...], name: str, *, positive: bool = True) -> tuple[int, ...]:
    result = tuple(value)
    if len(result) not in (2, 3) or any(not isinstance(item, int) for item in result):
        raise ValueError(f"{name} must contain two or three integers")
    lower = 1 if positive else 0
    if any(item < lower for item in result):
        raise ValueError(f"{name} values must be >= {lower}")
    return result


@dataclass(frozen=True, slots=True)
class OperatorDescriptor:
    """Shape-independent identity for one optimizable operator."""

    family: str
    direction: str
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...] | None = None
    stride: tuple[int, ...] | None = None
    padding: tuple[int, ...] | None = None
    dilation: tuple[int, ...] | None = None
    groups: int | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        if not self.family or not self.direction:
            raise ValueError("family and direction must be non-empty")
        for name in ("in_channels", "out_channels"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        geometry = ("kernel_size", "stride", "padding", "dilation", "groups")
        present = tuple(getattr(self, name) is not None for name in geometry)
        if any(present) and not all(present):
            raise ValueError("convolution geometry fields must be supplied together")
        if not any(present):
            return
        if self.groups is None or self.groups <= 0:
            raise ValueError("groups must be positive")
        object.__setattr__(self, "kernel_size", _spatial_tuple(self.kernel_size, "kernel_size"))
        object.__setattr__(self, "stride", _spatial_tuple(self.stride, "stride"))
        object.__setattr__(
            self, "padding", _spatial_tuple(self.padding, "padding", positive=False)
        )
        object.__setattr__(self, "dilation", _spatial_tuple(self.dilation, "dilation"))

    @classmethod
    def normalization(
        cls, *, family: str, channels: int, role: str | None = None
    ) -> OperatorDescriptor:
        return cls(family, "regular", channels, channels, role=role)

    @classmethod
    def convolution(
        cls,
        *,
        family: str,
        direction: str,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        groups: int,
        role: str | None = None,
    ) -> OperatorDescriptor:
        return cls(
            family,
            direction,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            role,
        )

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.family,
            self.direction,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
        )

    def to_primitive(self) -> dict[str, object]:
        value = asdict(self)
        for name in ("kernel_size", "stride", "padding", "dilation"):
            if value[name] is not None:
                value[name] = list(value[name])
        return {name: item for name, item in value.items() if item is not None}


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Runtime facts used for deterministic implementation resolution."""

    phase: ExecutionPhase
    device_type: str
    sm: tuple[int, int] | None
    total_vram_bytes: int
    dtype: str
    batch_size: int
    spatial_shape: tuple[int, ...]
    model_family: str
    variant: str
    checkpointing: str
    export: bool = False
    allow_approximate: bool = False

    def __post_init__(self) -> None:
        if self.phase not in ("training", "inference", "export"):
            raise ValueError(f"unknown phase {self.phase!r}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.total_vram_bytes < 0:
            raise ValueError("total_vram_bytes must be nonnegative")
        object.__setattr__(
            self, "spatial_shape", _spatial_tuple(self.spatial_shape, "spatial_shape")
        )
        if self.sm is not None:
            sm = tuple(self.sm)
            if len(sm) != 2 or any(item < 0 for item in sm):
                raise ValueError("sm must contain two nonnegative integers")
            object.__setattr__(self, "sm", sm)

    @property
    def spatial_volume(self) -> int:
        return prod(self.spatial_shape)

    @property
    def total_vram_gib(self) -> float:
        return self.total_vram_bytes / 2**30

    def to_primitive(self) -> dict[str, object]:
        value = asdict(self)
        value["spatial_shape"] = list(self.spatial_shape)
        value["sm"] = None if self.sm is None else list(self.sm)
        return value
