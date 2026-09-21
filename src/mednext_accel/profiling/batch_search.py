"""Deterministic VRAM-aware batch-size exploration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True, slots=True)
class BatchSearch:
    strategy: Literal["auto", "explicit"] = "auto"
    memory_fraction: float = 0.90
    dense_until: int = 8
    maximum: int | None = None

    def __post_init__(self) -> None:
        if self.strategy not in ("auto", "explicit"):
            raise ValueError("batch search strategy must be 'auto' or 'explicit'")
        if not 0 < self.memory_fraction <= 1:
            raise ValueError("memory_fraction must be in (0, 1]")
        if self.dense_until < 1 or (self.maximum is not None and self.maximum < 1):
            raise ValueError("batch limits must be positive")


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
    def probed_batches(self) -> tuple[int, ...]:
        return tuple(item.batch for item in self.probes)


def search_batches(
    config: BatchSearch,
    *,
    total_vram_bytes: int,
    probe: Callable[[int], ProbeResult],
    transition_batches: tuple[int, ...] = (),
) -> BatchSearchResult:
    """Probe densely at small batches, then locate the VRAM boundary."""

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

    if not run(1).feasible:
        return BatchSearchResult((results[1],), 0)

    explicit_limit = config.maximum
    dense_limit = (
        config.dense_until if explicit_limit is None else min(config.dense_until, explicit_limit)
    )
    lower = 1
    upper: int | None = None
    for batch in range(2, dense_limit + 1):
        if run(batch).feasible:
            lower = batch
        else:
            upper = batch
            break

    if explicit_limit is not None:
        if upper is None:
            for batch in range(dense_limit + 1, explicit_limit + 1):
                if run(batch).feasible:
                    lower = batch
                else:
                    upper = batch
                    break
    elif upper is None:
        candidate = max(2, 2 ** (max(lower, 1).bit_length()))
        while run(candidate).feasible:
            lower = candidate
            candidate *= 2
        upper = candidate

    if upper is not None:
        while upper - lower > 1:
            candidate = (lower + upper) // 2
            if run(candidate).feasible:
                lower = candidate
            else:
                upper = candidate

    for transition in transition_batches:
        for batch in (transition - 1, transition, transition + 1):
            if batch >= 1 and (explicit_limit is None or batch <= explicit_limit):
                run(batch)

    ordered = tuple(results[batch] for batch in sorted(results))
    maximum = max((item.batch for item in ordered if item.feasible), default=0)
    return BatchSearchResult(ordered, maximum)
