"""Whole-model validation records and policy checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WholeModelResult:
    mode: str
    step_ms: float
    peak_bytes: int
    valid: bool
    message: str | None = None


@dataclass(frozen=True, slots=True)
class WholeModelValidation:
    reference: WholeModelResult
    candidate: WholeModelResult
    requires_memory_override: bool
    accepted: bool


def validate_profile_candidate(
    run: Callable[[str], WholeModelResult], *, memory_tolerance: float = 0.10
) -> WholeModelValidation:
    reference = run("reference")
    candidate = run("candidate")
    regression = candidate.peak_bytes > reference.peak_bytes * (1 + memory_tolerance)
    accepted = reference.valid and candidate.valid and candidate.step_ms < reference.step_ms
    return WholeModelValidation(reference, candidate, regression, accepted)
