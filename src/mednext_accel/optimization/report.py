"""Immutable profile-resolution reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias

from .descriptors import ExecutionContext, OperatorDescriptor

JsonValue: TypeAlias = (
    str | int | float | bool | None | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
)


@dataclass(frozen=True, slots=True)
class Decision:
    descriptor: OperatorDescriptor
    phase: str
    implementation: str
    parameters: Mapping[str, JsonValue]
    profile: str
    rule: str
    confidence: Literal["measured", "interpolated", "extrapolated", "default", "guard"]
    warning: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    profile: str
    context: ExecutionContext
    decisions: tuple[Decision, ...]
    warnings: tuple[str, ...] = ()
