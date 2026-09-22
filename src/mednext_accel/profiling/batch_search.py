"""Deterministic VRAM-aware batch-size exploration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class BatchSearch:
    memory_fraction: float = 0.90
    maximum: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.memory_fraction <= 1:
            raise ValueError("memory_fraction must be in (0, 1]")
        if self.maximum is not None and self.maximum < 1:
            raise ValueError("maximum must be positive")


@dataclass(frozen=True, slots=True)
class ProbeResult:
    batch: int
    feasible: bool
    peak_bytes: int
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BatchSearchResult:
    probes: tuple[ProbeResult, ...]
    maximum_feasible: int

    @property
    def by_batch(self) -> Mapping[int, ProbeResult]:
        return MappingProxyType({item.batch: item for item in self.probes})

    @property
    def probed_batches(self) -> tuple[int, ...]:
        return tuple(item.batch for item in self.probes)


def search_batches(
    config: BatchSearch,
    *,
    total_vram_bytes: int,
    probe: Callable[[int], ProbeResult],
) -> BatchSearchResult:
    """Locate the largest feasible batch at or below an optional upper bound."""

    budget = int(total_vram_bytes * config.memory_fraction)
    results: dict[int, ProbeResult] = {}

    def run(batch: int) -> ProbeResult:
        if batch in results:
            return results[batch]
        result = probe(batch)
        if result.batch != batch:
            raise ValueError("probe returned a result for a different batch")
        if result.feasible and result.peak_bytes > budget:
            result = replace(result, feasible=False, reason="memory_fraction")
        results[batch] = result
        return result

    if config.maximum is not None:
        upper = config.maximum
        if run(upper).feasible:
            return BatchSearchResult((results[upper],), upper)
        if upper == 1 or not run(1).feasible:
            ordered = tuple(results.values())
            return BatchSearchResult(ordered, 0)
        lower = 1
    else:
        if not run(1).feasible:
            return BatchSearchResult((results[1],), 0)
        lower = 1
        upper = 2
        while run(upper).feasible:
            lower = upper
            upper *= 2

    while upper - lower > 1:
        candidate = (lower + upper) // 2
        if run(candidate).feasible:
            lower = candidate
        else:
            upper = candidate

    ordered = tuple(results.values())
    return BatchSearchResult(ordered, lower)
