import gc
from dataclasses import replace

import pytest

from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor
from mednext_accel.optimization.policy_resolver import PolicyResolver
from mednext_accel.profiling import runner
from mednext_accel.profiling.batch_search import BatchSearch
from mednext_accel.profiling.batch_search import BatchSearchResult as BoundarySearchResult
from mednext_accel.profiling.campaign import Campaign, Workload
from mednext_accel.profiling.execution import WorkloadBatchSelection, WorkloadShapes
from mednext_accel.profiling.grouped import case_payload, run_group_with_bisection
from mednext_accel.profiling.matrix import KernelCase, KernelCaseKey
from mednext_accel.profiling.progress import ProgressEvent
from mednext_accel.profiling.synthesize import synthesize_profile


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 1

    @staticmethod
    def get_device_properties(index: int):
        assert index == 1
        return type("Properties", (), {"total_memory": 48 * 1024**3})()

    @staticmethod
    def get_device_capability() -> tuple[int, int]:
        return (8, 9)


class _Torch:
    cuda = _Cuda()


class _Recorder:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass


def _kernel_case(
    family: str,
    *,
    direction: str = "regular",
    phase: str = "training",
    out_channels: int = 16,
) -> KernelCase:
    return KernelCase(
        KernelCaseKey(
            family=family,
            direction=direction,
            phase=phase,
            batch=1,
            spatial_shape=(16, 16, 16),
            in_channels=8,
            out_channels=out_channels,
            kernel_size=1 if family == "pointwise_conv3d" else 3,
            dtype="bfloat16",
            implementation=(
                "pointwise_gemm_per_sample"
                if family == "pointwise_conv3d"
                else "triton_depthwise_dx"
            ),
            parameters=(),
        )
    )


def test_kernel_group_child_runs_cases_in_order_and_cleans_up_each(monkeypatch) -> None:
    cleanups = []

    class ChildCuda:
        OutOfMemoryError = RuntimeError

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            cleanups.append("cuda")

    class ChildTorch:
        cuda = ChildCuda()

    pointwise = _kernel_case("pointwise_conv3d")
    depthwise = _kernel_case(
        "depthwise_conv3d", direction="regular", phase="backward_input", out_channels=8
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", ChildTorch())
    monkeypatch.setattr(gc, "collect", lambda: cleanups.append("gc"))
    monkeypatch.setattr(
        runner,
        "_pointwise_probe",
        lambda payload: {"status": "ok", "probe": "pointwise"},
    )
    monkeypatch.setattr(
        runner,
        "_depthwise_probe",
        lambda payload: {"status": "ok", "probe": "depthwise"},
    )

    result = runner._child(
        {
            "kind": "kernel_group",
            "cases": [case_payload(pointwise), case_payload(depthwise)],
        }
    )

    assert result == {
        "status": "ok",
        "results": [
            {"case_id": pointwise.identifier, "status": "ok", "probe": "pointwise"},
            {"case_id": depthwise.identifier, "status": "ok", "probe": "depthwise"},
        ],
    }
    assert cleanups == ["gc", "cuda", "gc", "cuda"]


def test_kernel_group_child_converts_case_exceptions_and_continues(monkeypatch) -> None:
    class OutOfMemoryError(Exception):
        pass

    class ChildCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            pass

    ChildCuda.OutOfMemoryError = OutOfMemoryError

    class ChildTorch:
        cuda = ChildCuda()

    cases = (_kernel_case("pointwise_conv3d"), _kernel_case("pointwise_conv3d", out_channels=32))

    def probe(payload):
        if payload["case_id"] == cases[0].identifier:
            raise OutOfMemoryError
        raise ValueError("broken case")

    monkeypatch.setitem(__import__("sys").modules, "torch", ChildTorch())
    monkeypatch.setattr(runner, "_pointwise_probe", probe)

    result = runner._child(
        {"kind": "kernel_group", "cases": [case_payload(case) for case in cases]}
    )

    assert result == {
        "status": "ok",
        "results": [
            {
                "case_id": cases[0].identifier,
                "status": "oom",
                "message": "CUDA out of memory",
                "failure_stage": "dispatch",
            },
            {
                "case_id": cases[1].identifier,
                "status": "error",
                "message": "ValueError: broken case",
                "failure_stage": "dispatch",
            },
        ],
    }


def _campaign_contexts():
    workload = Workload("mednext_v1", "base", (128, 128, 128))
    return Campaign(
        "mednext-v1",
        (workload, replace(workload, checkpointing="all-expansion")),
        BatchSearch(maximum=5),
    )


def _prepare_campaign(monkeypatch, *, batches=(2,), depthwise=()):
    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch())
    monkeypatch.setattr(
        runner,
        "_batches",
        lambda *args, **kwargs: WorkloadBatchSelection(
            batches,
            {batch: {"status": "ok", "step_ms": 10.0, "peak_bytes": 100} for batch in batches},
        ),
    )
    monkeypatch.setattr(
        runner,
        "discover_workload_shapes",
        lambda workload: WorkloadShapes(((32, 64, (128, 128, 128)),), depthwise),
    )


def _winning_result():
    return {
        "status": "ok",
        "valid": True,
        "reference_ms": 10.0,
        "candidate_ms": 8.0,
        "reference_peak_bytes": 100,
        "candidate_peak_bytes": 100,
    }


def _successful_group(payload):
    assert payload["kind"] == "kernel_group", "legacy one-child-per-case path used"
    return {
        "status": "ok",
        "results": [{"case_id": case["case_id"], **_winning_result()} for case in payload["cases"]],
    }


def test_batch_search_advances_after_every_completed_probe(monkeypatch) -> None:
    def search(config, *, total_vram_bytes, probe):
        assert total_vram_bytes == 48 * 1024**3
        probes = (probe(1), probe(3))
        return BoundarySearchResult(probes, maximum_feasible=1)

    outcomes = iter(
        (
            {"status": "ok", "step_ms": 10.0, "peak_bytes": 1024**3},
            {"status": "oom", "message": "out of memory"},
        )
    )
    monkeypatch.setattr(runner, "search_batches", search)
    monkeypatch.setattr(runner, "_invoke", lambda payload: next(outcomes))
    reporter = _Recorder()

    selected = runner._batches(
        _campaign_contexts(),
        _campaign_contexts().workloads[0],
        48 * 1024**3,
        progress=reporter,
    )

    assert selected.batches == (1,)
    advances = [event for event in reporter.events if event.kind == "advance"]
    assert [(event.completed, event.total) for event in advances] == [(1, None), (2, None)]
    assert advances[0].message == "batch=1 feasible peak=1.0 GiB"
    assert advances[1].message == "batch=3 oom peak=0.0 GiB"


@pytest.mark.parametrize("maximum_feasible", [13, 0])
def test_batch_search_keeps_only_selected_maximum_and_reports_outcome(
    monkeypatch, maximum_feasible
):
    batches_to_probe = (16, 1, 8, 12, 14, 13) if maximum_feasible else (16, 1)
    model_results = {
        batch: (
            {"status": "ok", "step_ms": float(batch), "peak_bytes": batch * 100}
            if maximum_feasible and batch <= 13
            else {"status": "oom", "message": "CUDA out of memory"}
        )
        for batch in batches_to_probe
    }

    def search(config, *, total_vram_bytes, probe):
        return BoundarySearchResult(
            tuple(probe(batch) for batch in batches_to_probe), maximum_feasible
        )

    monkeypatch.setattr(runner, "search_batches", search)
    monkeypatch.setattr(runner, "_invoke", lambda payload: model_results[payload["batch"]])
    reporter = _Recorder()
    campaign = _campaign_contexts()
    selected = runner._batches(campaign, campaign.workloads[0], 48 * 1024**3, progress=reporter)

    assert selected.batches == ((13,) if maximum_feasible else ())
    assert selected.reference_steps == ({13: model_results[13]} if maximum_feasible else {})
    assert tuple(item.batch for item in selected.evidence.attempts) == batches_to_probe
    outcome = "maximum feasible batch: 13" if maximum_feasible else "no feasible batch"
    assert reporter.events[-1] == ProgressEvent("status", "batch-search", message=outcome)
    assert sum(event.message == outcome for event in reporter.events) == 1
    advances = [event for event in reporter.events if event.kind == "advance"]
    assert len(advances) == len(batches_to_probe)
    assert all(event.total is None for event in advances)


def test_campaign_shares_groups_and_materializes_each_checkpoint_context(monkeypatch, capsys):
    _prepare_campaign(
        monkeypatch,
        depthwise=(
            ("regular", 32, 3, (128, 128, 128)),
            ("downsample", 32, 3, (128, 128, 128)),
            ("transpose", 32, 3, (64, 64, 64)),
        ),
    )
    groups = []
    validations = {"none": [], "all-expansion": []}

    def execute_group(group, invoke, on_attempt=None, *, seed=0):
        groups.append(group)
        return run_group_with_bisection(group, invoke, on_attempt=on_attempt, seed=seed)

    def invoke(payload):
        if payload["kind"] == "model":
            context = payload["workload"]["checkpointing"]
            validations[context].append(payload["batch"])
            profile = payload["optimization"]
            assert {r["when"]["checkpointing"] for r in profile["rules"]} == {context}
            assert len(profile["rules"]) == 5
            return {"status": "ok", "step_ms": 8.0, "peak_bytes": 100}
        return _successful_group(payload)

    monkeypatch.setattr(runner, "run_group_with_bisection", execute_group, raising=False)
    monkeypatch.setattr(runner, "_invoke", invoke)
    reporter = _Recorder()
    # Exercise the compatible public entry point as well as the richer result below.
    measurements = runner.run_campaign(_campaign_contexts(), progress=reporter)

    assert len(groups) == 5
    assert {group.category for group in groups} == {
        "pointwise",
        "regular-dx",
        "regular-dw",
        "downsample-dx",
        "transpose-dw",
    }
    assert validations == {"none": [2], "all-expansion": [2]}
    assert isinstance(measurements, tuple)
    assert len(measurements) == 10
    assert {item.checkpointing for item in measurements} == {"none", "all-expansion"}
    assert all(item.kernel_valid for item in measurements)
    assert any(event.stage == "batch-search" for event in reporter.events)
    assert capsys.readouterr().err == ""

    groups.clear()
    run = runner.execute_campaign(_campaign_contexts(), progress=_Recorder())
    assert run.measurements == measurements
    assert run.statistics.requested_case_count == 10
    assert run.statistics.kernel_case_count == 5
    assert run.statistics.kernel_group_count == 5
    assert run.statistics.whole_model_validation_count == 2

    events = reporter.events
    kernel_plan = next(
        event for event in events if event.kind == "status" and event.stage == "kernel-groups"
    )
    assert kernel_plan.message == (
        "kernel plan: 5 planned groups, 5 unique experiments, 10 requested references"
    )
    group_advances = [
        event for event in events if event.kind == "advance" and event.stage == "kernel-groups"
    ]
    assert [event.completed for event in group_advances] == list(range(1, 6))
    assert [event.secondary_completed for event in group_advances] == list(range(1, 6))
    assert all(event.secondary_total == 5 for event in group_advances)
    assert (
        ProgressEvent(
            "status",
            "validation",
            message="validation plan: 2 whole-model validations",
        )
        in events
    )
    validation_starts = [
        event for event in events if event.kind == "stage_start" and event.stage == "validation"
    ]
    assert validation_starts == [ProgressEvent("stage_start", "validation", total=2)]
    validation_advances = [
        event for event in events if event.kind == "advance" and event.stage == "validation"
    ]
    assert [event.completed for event in validation_advances] == [1, 2]
    assert all(event.total == 2 for event in validation_advances)


@pytest.mark.parametrize("rejection", ["regression", "error"])
def test_selected_batch_rejection_preserves_kernel_evidence_and_records_policy_outcome(
    monkeypatch, rejection
):
    _prepare_campaign(monkeypatch, batches=(5,))
    validations = {"none": [], "all-expansion": []}

    def invoke(payload):
        if payload["kind"] != "model":
            return _successful_group(payload)
        context = payload["workload"]["checkpointing"]
        validations[context].append(payload["batch"])
        if context == "none":
            if rejection == "error":
                return {"status": "error"}
            return {"status": "ok", "step_ms": 12.0, "peak_bytes": 100}
        return {"status": "ok", "step_ms": 8.0, "peak_bytes": 100}

    monkeypatch.setattr(runner, "_invoke", invoke)
    run = runner.execute_campaign(_campaign_contexts())

    assert validations == {"none": [5], "all-expansion": [5]}
    assert [m.kernel_valid for m in run.measurements if m.checkpointing == "none"] == [True]
    assert [m.objective_winner for m in run.measurements] == [True, True]
    assert [item.policy_accepted for item in run.model_comparisons] == [False, True]
    assert run.model_comparisons[0].candidate.status == ("error" if rejection == "error" else "ok")
    assert run.model_comparisons[0].reference.step_ms == 10.0
    assert [item["name"] for item in run.model_comparisons[0].effective_policy] == [
        "campaign-candidate",
        "sm89",
        "shared-nvidia",
    ]
    assert [m.kernel_valid for m in run.measurements if m.checkpointing == "all-expansion"] == [
        True
    ]
    assert run.statistics.whole_model_validation_count == 2


def test_variants_validate_independently_against_their_own_reference(monkeypatch):
    _prepare_campaign(monkeypatch, batches=(3,))
    base = _campaign_contexts().workloads[0]
    campaign = replace(
        _campaign_contexts(),
        workloads=(
            base,
            replace(base, variant="large"),
            replace(base, checkpointing="all-expansion"),
        ),
    )
    validations = []
    monkeypatch.setattr(
        runner,
        "_batches",
        lambda campaign, workload, total_vram, progress=None: WorkloadBatchSelection(
            (3,),
            {
                batch: {
                    "status": "ok",
                    "step_ms": 10.0 if workload.variant == "base" else 6.0,
                    "peak_bytes": 100,
                }
                for batch in (3,)
            },
        ),
    )

    def invoke(payload):
        if payload["kind"] != "model":
            return _successful_group(payload)
        validations.append(
            (
                payload["workload"]["variant"],
                payload["workload"]["checkpointing"],
                payload["batch"],
            )
        )
        assert payload["optimization"]["version"] == 2
        assert len(payload["optimization"]["rules"]) == 1
        return {"status": "ok", "step_ms": 8.0, "peak_bytes": 100}

    monkeypatch.setattr(runner, "_invoke", invoke)
    run = runner.execute_campaign(campaign)

    assert validations == [
        ("base", "none", 3),
        ("large", "none", 3),
        ("base", "all-expansion", 3),
    ]
    assert run.statistics.kernel_case_count == 1
    assert run.statistics.requested_case_count == 3

    profile = synthesize_profile(
        run.policy_measurements, name="final", sm=(8, 9), objective="balanced"
    )
    resolver = PolicyResolver(external=profile)
    descriptor = OperatorDescriptor(
        "pointwise_conv3d", "regular", 32, 64, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1), 1
    )
    for batch in (3,):
        for variant in ("base", "large"):
            for checkpointing in ("none", "all-expansion"):
                context = ExecutionContext(
                    "training",
                    "cuda",
                    (8, 9),
                    48 * 1024**3,
                    "bfloat16",
                    batch,
                    (128, 128, 128),
                    "mednext-v1",
                    variant,
                    checkpointing,
                )
                decision = resolver.resolve(descriptor, context, "training")
                assert decision.implementation == (
                    "reference" if checkpointing == "none" else "pointwise_gemm_per_sample"
                )
    assert [m.kernel_valid for m in run.measurements] == [True, True, True]
    assert [item.policy_accepted for item in run.model_comparisons] == [True, False, True]


def test_statistics_count_physical_bisection_attempts_and_keep_failed_evidence(monkeypatch):
    _prepare_campaign(monkeypatch, batches=(1,))
    monkeypatch.setattr(
        runner,
        "discover_workload_shapes",
        lambda workload: WorkloadShapes(
            ((32, 64, (128, 128, 128)), (32, 128, (128, 128, 128))), ()
        ),
    )
    attempts = []

    def invoke(payload):
        if payload["kind"] == "model":
            return {"status": "ok", "step_ms": 8.0, "peak_bytes": 100}
        attempts.append(payload)
        if len(payload["cases"]) == 2:
            return {"status": "timeout"}
        if payload["cases"][0]["out_channels"] == 128:
            return {"status": "error"}
        return _successful_group(payload)

    monkeypatch.setattr(runner, "_invoke", invoke)
    reporter = _Recorder()
    run = runner.execute_campaign(_campaign_contexts(), progress=reporter)

    assert len(attempts) == run.statistics.kernel_group_count == 3
    assert run.statistics.requested_case_count == 4
    assert run.statistics.kernel_case_count == 2
    assert run.statistics.whole_model_validation_count == 2
    assert len(run.measurements) == 4
    for item in run.measurements:
        assert item.kernel_valid is (True if item.out_channels == 64 else None)
        if item.out_channels == 128:
            assert item.reference_ms == item.candidate_ms == 0.0

    group_advances = [
        event
        for event in reporter.events
        if event.kind == "advance" and event.stage == "kernel-groups"
    ]
    assert [
        (event.completed, event.total, event.secondary_completed, event.secondary_total)
        for event in group_advances
    ] == [
        (0, 3, 0, 2),
        (1, 3, 0, 2),
        (2, 3, 1, 2),
        (3, 3, 2, 2),
    ]


def test_no_feasible_batches_launch_no_kernel_or_validation_children(monkeypatch):
    _prepare_campaign(monkeypatch, batches=())
    monkeypatch.setattr(runner, "_invoke", lambda payload: pytest.fail("unexpected child"))

    run = runner.execute_campaign(_campaign_contexts())

    assert run.measurements == ()
    assert run.statistics.requested_case_count == 0
    assert run.statistics.kernel_case_count == 0
    assert run.statistics.kernel_group_count == 0
    assert run.statistics.whole_model_validation_count == 0


def test_static_shape_plan_is_emitted_after_discovery_before_adaptive_search(monkeypatch, capsys):
    _prepare_campaign(monkeypatch, batches=())
    timeline = []

    class Recorder(_Recorder):
        def emit(self, event):
            super().emit(event)
            if event.stage == "static-plan":
                timeline.append("static-plan")

    reporter = Recorder()

    def discover(workload):
        timeline.append("discover")
        return WorkloadShapes(
            (
                (32, 64, (128, 128, 128)),
                (32, 128 if workload.checkpointing == "none" else 256, (128, 128, 128)),
            ),
            (
                ("regular", 32, 3, (128, 128, 128)),
                ("regular", 64, 3, (64, 64, 64)),
                ("downsample", 64, 3, (64, 64, 64)),
                ("transpose", 128, 3, (32, 32, 32)),
            ),
        )

    def search(*args, **kwargs):
        timeline.append("search")
        return WorkloadBatchSelection((), {})

    monkeypatch.setattr(runner, "discover_workload_shapes", discover)
    monkeypatch.setattr(runner, "_batches", search)

    runner.execute_campaign(_campaign_contexts(), progress=reporter)

    assert timeline == ["discover", "discover", "static-plan", "search", "search"]
    static = [event for event in reporter.events if event.stage == "static-plan"]
    assert len(static) == 1
    event = static[0]
    assert event.kind == "status" and event.total is None and event.completed is None
    for fact in (
        "static shape plan",
        "2 workloads",
        "3 unique pointwise shapes",
        "6 unique depthwise phase/shape pairs",
        "5 possible kernel groups per feasible batch",
        "pointwise",
        "regular-dx",
        "regular-dw",
        "downsample-dx",
        "transpose-dw",
    ):
        assert fact in event.message
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("dtypes", [("float32",), ("bfloat16", "float32")])
@pytest.mark.parametrize("entrypoint", [runner.execute_campaign, runner.run_campaign])
def test_runner_rejects_unsupported_dtypes_before_gpu_or_planning(monkeypatch, dtypes, entrypoint):
    supported = _campaign_contexts().workloads[0]
    campaign = replace(
        _campaign_contexts(), workloads=(supported, replace(supported, dtypes=dtypes))
    )
    callbacks = []

    class Cuda(_Cuda):
        @staticmethod
        def is_available():
            callbacks.append("cuda")
            return True

    class Torch:
        cuda = Cuda()

    monkeypatch.setitem(__import__("sys").modules, "torch", Torch())
    monkeypatch.setattr(
        runner,
        "discover_workload_shapes",
        lambda workload: callbacks.append("discover") or WorkloadShapes((), ()),
    )
    monkeypatch.setattr(
        runner,
        "_batches",
        lambda *args, **kwargs: callbacks.append("search") or ((), {}),
    )
    monkeypatch.setattr(runner, "_invoke", lambda payload: callbacks.append("invoke") or {})

    with pytest.raises(ValueError, match=r"unsupported.*dtype.*float32"):
        entrypoint(campaign)

    assert callbacks == []


def test_campaign_profiles_and_validates_only_each_contexts_selected_maximum(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch())
    monkeypatch.setattr(
        runner,
        "discover_workload_shapes",
        lambda workload: WorkloadShapes(((32, 64, (128, 128, 128)),), ()),
    )
    references = {"none": [], "all-expansion": []}
    validations = {"none": [], "all-expansion": []}
    kernel_batches = []

    def invoke(payload):
        if payload["kind"] == "kernel_group":
            kernel_batches.extend(case["batch"] for case in payload["cases"])
            return _successful_group(payload)
        context = payload["workload"]["checkpointing"]
        batch = payload["batch"]
        if "optimization" in payload:
            validations[context].append(batch)
            assert {r["when"]["batch"] for r in payload["optimization"]["rules"]} == {batch}
            return {"status": "ok", "step_ms": 8.0, "peak_bytes": 100}
        references[context].append(batch)
        if context == "none" and batch > 3:
            return {"status": "oom", "message": "CUDA out of memory"}
        return {"status": "ok", "step_ms": 10.0, "peak_bytes": 100}

    monkeypatch.setattr(runner, "_invoke", invoke)
    run = runner.execute_campaign(_campaign_contexts())

    assert references == {"none": [5, 1, 3, 4], "all-expansion": [5]}
    assert [[probe.batch for probe in item.attempts] for item in run.batch_searches] == [
        [5, 1, 3, 4],
        [5],
    ]
    assert [item.selected_maximum for item in run.batch_searches] == [3, 5]
    assert run.batch_searches[0].by_batch[5].result.status == "oom"
    assert all(item.reference.seed == item.candidate.seed == 0 for item in run.model_comparisons)
    assert kernel_batches == [3, 5]
    assert validations == {"none": [3], "all-expansion": [5]}
    assert [(m.checkpointing, m.batch) for m in run.measurements] == [
        ("none", 3),
        ("all-expansion", 5),
    ]
    assert run.statistics.whole_model_validation_count == 2
    profile = synthesize_profile(
        run.policy_measurements, name="final", sm=(8, 9), objective="balanced"
    )
    assert {(rule.when["batch"].minimum, rule.when["batch"].maximum) for rule in profile.rules} == {
        (3, 3),
        (5, 5),
    }


@pytest.mark.parametrize("metric", ["step_ms", "peak_bytes"])
@pytest.mark.parametrize("value", [None, 0, float("nan"), float("inf")])
def test_batch_probe_with_invalid_metrics_is_recorded_and_cannot_be_selected(
    monkeypatch, metric, value
):
    data = {"status": "ok", "step_ms": 10.0, "peak_bytes": 100, metric: value}
    monkeypatch.setattr(runner, "_invoke", lambda payload: data)
    campaign = replace(_campaign_contexts(), batch_search=BatchSearch(maximum=1))
    selected = runner._batches(campaign, campaign.workloads[0], 48 * 1024**3, progress=_Recorder())
    assert selected.batches == ()
    assert selected.evidence.attempts[0].feasible is False
    assert f"invalid {metric}" in selected.evidence.attempts[0].reason
