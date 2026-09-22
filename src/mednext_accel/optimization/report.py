"""Immutable reports of effective policy layers and guarded decisions."""

from dataclasses import dataclass

from .descriptors import ExecutionContext
from .policy_resolver import PolicyDecision


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    policies: tuple[str, ...]
    context: ExecutionContext
    decisions: tuple[PolicyDecision, ...]
    warnings: tuple[str, ...] = ()
