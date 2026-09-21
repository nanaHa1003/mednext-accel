from dataclasses import replace

import pytest

from mednext_accel.profiling.batch_search import BatchSearch
from mednext_accel.profiling.campaign import Campaign, Workload
from mednext_accel.profiling.execution import (
    BatchSearchResult,
    WorkloadShapes,
    build_execution_plan,
    deduplicate_workloads,
)


def reference_step() -> dict[str, object]:
    return {"status": "ok", "step_ms": 10.0, "peak_bytes": 1_000}


def test_duplicate_workloads_search_once() -> None:
    workload = Workload("mednext_v1", "base", (128, 128, 128))
    campaign = Campaign("mednext-v1", (workload, workload), BatchSearch(maximum=2))
    shapes = WorkloadShapes(
        pointwise=((32, 64, (128, 128, 128)),),
        depthwise=(),
    )
    calls = []
    plan = build_execution_plan(
        campaign,
        discover=lambda workload: shapes,
        search=lambda workload: (
            calls.append(workload)
            or BatchSearchResult((1, 2), {1: reference_step(), 2: reference_step()})
        ),
    )
    assert len(calls) == 1
    assert len(plan.workloads) == 1


def test_checkpoint_workloads_search_separately_but_share_kernel_cases() -> None:
    none = Workload("mednext_v1", "base", (128, 128, 128), checkpointing="none")
    expansion = replace(none, checkpointing="all-expansion")
    campaign = Campaign("mednext-v1", (none, expansion), BatchSearch(maximum=2))
    shapes = WorkloadShapes(
        pointwise=((32, 64, (128, 128, 128)),),
        depthwise=(),
    )
    searches = []
    plan = build_execution_plan(
        campaign,
        discover=lambda workload: shapes,
        search=lambda workload: (
            searches.append(workload)
            or BatchSearchResult((1, 2), {1: reference_step(), 2: reference_step()})
        ),
    )
    assert searches == [none, expansion]
    assert len(plan.workloads) == 2
    assert len(plan.kernel_cases) == len(plan.workloads[0].case_keys)
    assert plan.workloads[0].case_keys == plan.workloads[1].case_keys
    assert plan.requested_case_count == sum(len(run.case_keys) for run in plan.workloads)
    assert plan.requested_case_count == 2 * len(plan.kernel_cases)
    assert plan.workloads[0].shapes is shapes
    assert plan.workloads[0].reference_steps[1] == reference_step()


def test_planning_discovers_every_shape_before_batch_search() -> None:
    first = Workload("mednext_v1", "small", (64, 64, 64))
    second = replace(first, checkpointing="whole-block")
    campaign = Campaign("mednext-v1", (first, first, second), BatchSearch(maximum=1))
    shapes = WorkloadShapes(pointwise=(), depthwise=())
    events = []

    plan = build_execution_plan(
        campaign,
        discover=lambda workload: events.append(("discover", workload)) or shapes,
        search=lambda workload: events.append(("search", workload)) or BatchSearchResult((), {}),
    )

    assert deduplicate_workloads(campaign.workloads) == (first, second)
    assert events == [
        ("discover", first),
        ("discover", second),
        ("search", first),
        ("search", second),
    ]
    assert plan.kernel_cases == ()
    assert plan.kernel_groups == ()
    assert plan.requested_case_count == 0


@pytest.mark.parametrize("dtypes", [("float32",), ("bfloat16", "float32")])
def test_plan_rejects_unsupported_dtypes_before_any_discovery_or_search(dtypes):
    supported = Workload("mednext_v1", "base", (128, 128, 128))
    campaign = Campaign(
        "mednext-v1", (supported, replace(supported, dtypes=dtypes)), BatchSearch(maximum=1)
    )
    callbacks = []

    with pytest.raises(ValueError, match=r"unsupported.*dtype.*float32"):
        build_execution_plan(
            campaign,
            discover=lambda workload: callbacks.append("discover") or WorkloadShapes((), ()),
            search=lambda workload: callbacks.append("search") or BatchSearchResult((), {}),
        )

    assert callbacks == []
