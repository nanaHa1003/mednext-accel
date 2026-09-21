"""Campaign-wide planning for deduplicated profiling execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .campaign import Campaign, Workload
from .matrix import (
    KernelCase,
    KernelCaseKey,
    KernelGroup,
    build_workload_cases,
    deduplicate_cases,
    group_cases,
)


@dataclass(frozen=True, slots=True)
class WorkloadShapes:
    pointwise: tuple[tuple[int, int, tuple[int, int, int]], ...]
    depthwise: tuple[tuple[str, int, int, tuple[int, int, int]], ...]


@dataclass(frozen=True, slots=True)
class BatchSearchResult:
    batches: tuple[int, ...]
    reference_steps: Mapping[int, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class WorkloadRun:
    workload: Workload
    shapes: WorkloadShapes
    batches: tuple[int, ...]
    reference_steps: Mapping[int, Mapping[str, object]]
    case_keys: tuple[KernelCaseKey, ...]


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    workloads: tuple[WorkloadRun, ...]
    kernel_cases: tuple[KernelCase, ...]
    kernel_groups: tuple[KernelGroup, ...]
    requested_case_count: int


def deduplicate_workloads(workloads: tuple[Workload, ...]) -> tuple[Workload, ...]:
    """Return exact workload values once, preserving their first-seen order."""

    return tuple(dict.fromkeys(workloads))


def build_execution_plan(
    campaign: Campaign,
    *,
    discover: Callable[[Workload], WorkloadShapes],
    search: Callable[[Workload], BatchSearchResult],
) -> ExecutionPlan:
    """Discover, search, and globally deduplicate a campaign's kernel cases."""

    workloads = deduplicate_workloads(campaign.workloads)
    discovered = tuple((workload, discover(workload)) for workload in workloads)
    searched = tuple((workload, shapes, search(workload)) for workload, shapes in discovered)
    workload_cases = tuple(
        build_workload_cases(
            workload,
            result.batches,
            pointwise_shapes=shapes.pointwise,
            depthwise_shapes=shapes.depthwise,
        )
        for workload, shapes, result in searched
    )
    kernel_cases, references = deduplicate_cases(workload_cases)
    runs = tuple(
        WorkloadRun(
            workload=workload,
            shapes=shapes,
            batches=result.batches,
            reference_steps=result.reference_steps,
            case_keys=case_keys,
        )
        for (workload, shapes, result), case_keys in zip(searched, references, strict=True)
    )
    return ExecutionPlan(
        workloads=runs,
        kernel_cases=kernel_cases,
        kernel_groups=group_cases(kernel_cases),
        requested_case_count=sum(len(cases) for cases in workload_cases),
    )
