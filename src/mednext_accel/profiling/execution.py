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
from .progress import ProgressEvent, ProgressReporter


@dataclass(frozen=True, slots=True)
class WorkloadShapes:
    pointwise: tuple[tuple[int, int, tuple[int, int, int]], ...]
    depthwise: tuple[tuple[str, int, int, tuple[int, int, int]], ...]


@dataclass(frozen=True, slots=True)
class WorkloadBatchSelection:
    batches: tuple[int, ...]
    reference_steps: Mapping[int, Mapping[str, object]]

    def __post_init__(self) -> None:
        if len(self.batches) > 1:
            raise ValueError("workload batch selection must contain at most one batch")
        if set(self.reference_steps) != set(self.batches):
            raise ValueError("reference steps must match the selected batch")


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


def validate_campaign_dtypes(campaign: Campaign) -> None:
    """Reject dtypes unsupported by the BF16-only kernel and model probes."""
    unsupported = sorted(
        {
            dtype
            for workload in campaign.workloads
            for dtype in workload.dtypes
            if dtype != "bfloat16"
        }
    )
    if unsupported:
        raise ValueError(
            f"unsupported profiling dtype(s): {', '.join(unsupported)}; supported dtype: bfloat16"
        )


def build_execution_plan(
    campaign: Campaign,
    *,
    discover: Callable[[Workload], WorkloadShapes],
    search: Callable[[Workload], WorkloadBatchSelection],
    progress: ProgressReporter | None = None,
    sm: tuple[int, int] | None = None,
) -> ExecutionPlan:
    """Discover, search, and globally deduplicate a campaign's kernel cases."""

    validate_campaign_dtypes(campaign)
    workloads = deduplicate_workloads(campaign.workloads)
    discovered = tuple((workload, discover(workload)) for workload in workloads)
    if progress is not None:
        static_cases, _ = deduplicate_cases(
            tuple(
                build_workload_cases(
                    workload,
                    (1,),
                    pointwise_shapes=shapes.pointwise,
                    depthwise_shapes=shapes.depthwise,
                    sm=sm,
                )
                for workload, shapes in discovered
            )
        )
        pointwise_count = sum(case.key.family == "pointwise_conv3d" for case in static_cases)
        categories = tuple(group.category for group in group_cases(static_cases))
        progress.emit(
            ProgressEvent(
                "status",
                "static-plan",
                message=(
                    f"static shape plan: {len(workloads)} workloads, "
                    f"{pointwise_count} unique pointwise shapes, "
                    f"{len(static_cases) - pointwise_count} unique depthwise phase/shape pairs, "
                    f"{len(categories)} possible kernel groups per feasible batch "
                    f"({', '.join(categories) or 'none'})"
                ),
            )
        )
    searched = tuple((workload, shapes, search(workload)) for workload, shapes in discovered)
    workload_cases = tuple(
        build_workload_cases(
            workload,
            result.batches,
            pointwise_shapes=shapes.pointwise,
            depthwise_shapes=shapes.depthwise,
            sm=sm,
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
