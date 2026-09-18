"""Immutable optimization result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OptimizationPolicy = Literal["torch", "conservative", "autotune"]
CompileMode = Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]


@dataclass(frozen=True, slots=True)
class BackendSelections:
    """Shapes assigned to each optional execution backend."""

    pointwise_gemm: tuple[tuple[int, ...], ...] = ()
    depthwise_regular: tuple[tuple[int, int], ...] = ()
    depthwise_transpose: tuple[tuple[int, int], ...] = ()
    depthwise_downsample: tuple[tuple[int, int], ...] = ()

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.pointwise_gemm,
                self.depthwise_regular,
                self.depthwise_transpose,
                self.depthwise_downsample,
            )
        )


@dataclass(frozen=True, slots=True)
class KernelMeasurement:
    """One native-versus-candidate timing decision in milliseconds."""

    kind: str
    shape: tuple[int, ...]
    native_ms: float
    candidate_ms: float
    selected: bool

    @property
    def speedup(self) -> float:
        return self.native_ms / self.candidate_ms


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    """Result of applying an execution policy to a model."""

    policy: OptimizationPolicy
    compile_mode: CompileMode
    input_shape: tuple[int, ...]
    dtype: str
    device: str
    selections: BackendSelections
    measurements: tuple[KernelMeasurement, ...] = ()
    replacements: int = 0
    cache_hit: bool = False
    cache_key: str | None = None
    notes: tuple[str, ...] = ()
