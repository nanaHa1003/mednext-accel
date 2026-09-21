import gc
from dataclasses import replace

import pytest

from mednext_accel.profiling import runner
from mednext_accel.profiling.batch_search import BatchSearch
from mednext_accel.profiling.campaign import Campaign, Workload
from mednext_accel.profiling.execution import WorkloadShapes
from mednext_accel.profiling.grouped import case_payload, run_group_with_bisection
from mednext_accel.profiling.matrix import KernelCase, KernelCaseKey
from mednext_accel.profiling.progress import ProgressEvent


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
            },
            {
                "case_id": cases[1].identifier,
                "status": "error",
                "message": "ValueError: broken case",
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


def _prepare_campaign(monkeypatch, *, batches=(1, 2), depthwise=()):
    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch())
    monkeypatch.setattr(
        runner,
        "_batches",
        lambda *args, **kwargs: (
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

    def execute_group(group, invoke, on_attempt=None):
        groups.append(group)
        return run_group_with_bisection(group, invoke, on_attempt=on_attempt)

    def invoke(payload):
        if payload["kind"] == "model":
            context = payload["workload"]["checkpointing"]
            validations[context].append(payload["batch"])
            profile = payload["optimization"]
            assert {m["checkpointing"] for m in profile["measurements"]} == {context}
            assert len(profile["measurements"]) == 10
            return {"status": "ok", "step_ms": 8.0, "peak_bytes": 100}
        return _successful_group(payload)

    monkeypatch.setattr(runner, "run_group_with_bisection", execute_group, raising=False)
    monkeypatch.setattr(runner, "_invoke", invoke)
    reporter = _Recorder()
    # Exercise the compatible public entry point as well as the richer result below.
    measurements = runner.run_campaign(_campaign_contexts(), progress=reporter)

    assert len(groups) == 10
    assert {group.category for group in groups} == {
        "pointwise",
        "regular-dx",
        "regular-dw",
        "downsample-dx",
        "transpose-dw",
    }
    assert validations == {"none": [1, 2], "all-expansion": [1, 2]}
    assert isinstance(measurements, tuple)
    assert len(measurements) == 20
    assert {item.checkpointing for item in measurements} == {"none", "all-expansion"}
    assert all(item.valid for item in measurements)
    assert any(event.stage == "batch-search" for event in reporter.events)
    assert capsys.readouterr().err == ""

    groups.clear()
    run = runner.execute_campaign(_campaign_contexts(), progress=_Recorder())
    assert run.measurements == measurements
    assert run.statistics.requested_case_count == 20
    assert run.statistics.kernel_case_count == 10
    assert run.statistics.kernel_group_count == 10
    assert run.statistics.whole_model_validation_count == 4


@pytest.mark.parametrize("rejection", ["regression", "error", "missing-reference"])
def test_boundary_rejection_invalidates_entire_segment_only_in_its_context(monkeypatch, rejection):
    _prepare_campaign(monkeypatch, batches=(1, 2, 3, 4, 5))
    validations = {"none": [], "all-expansion": []}
    if rejection == "missing-reference":

        def batches(campaign, workload, total_vram, progress=None):
            return (1, 2, 3, 4, 5), {
                batch: {"status": "ok", "step_ms": 10.0, "peak_bytes": 100}
                for batch in (1, 2, 3, 4, 5)
                if not (workload.checkpointing == "none" and batch == 5)
            }

        monkeypatch.setattr(runner, "_batches", batches)

    def invoke(payload):
        if payload["kind"] != "model":
            return _successful_group(payload)
        context = payload["workload"]["checkpointing"]
        validations[context].append(payload["batch"])
        if context == "none" and payload["batch"] == 5:
            if rejection == "error":
                return {"status": "error"}
            if rejection == "regression":
                return {"status": "ok", "step_ms": 12.0, "peak_bytes": 100}
        return {"status": "ok", "step_ms": 8.0, "peak_bytes": 100}

    monkeypatch.setattr(runner, "_invoke", invoke)
    run = runner.execute_campaign(_campaign_contexts())

    assert validations == {"none": [1, 5], "all-expansion": [1, 5]}
    assert [m.valid for m in run.measurements if m.checkpointing == "none"] == [False] * 5
    assert [m.valid for m in run.measurements if m.checkpointing == "all-expansion"] == [True] * 5
    assert run.statistics.whole_model_validation_count == 4


def test_segment_changes_and_sampling_gaps_each_get_their_own_boundaries(monkeypatch):
    _prepare_campaign(monkeypatch, batches=(1, 2, 3, 4, 6))
    validations = []

    def invoke(payload):
        if payload["kind"] == "model":
            validations.append(payload["batch"])
            return {"status": "ok", "step_ms": 8.0, "peak_bytes": 100}
        result = _successful_group(payload)
        if payload["cases"][0]["batch"] == 3:
            result["results"][0]["candidate_ms"] = 12.0
        return result

    monkeypatch.setattr(runner, "_invoke", invoke)
    campaign = replace(_campaign_contexts(), workloads=_campaign_contexts().workloads[:1])
    run = runner.execute_campaign(campaign)

    assert validations == [1, 2, 3, 4, 6]
    assert len(run.measurements) == 5
    assert run.statistics.whole_model_validation_count == 5


def test_variants_validate_independently_against_their_own_reference(monkeypatch):
    _prepare_campaign(monkeypatch, batches=(1, 2, 3))
    base = _campaign_contexts().workloads[0]
    campaign = replace(_campaign_contexts(), workloads=(base, replace(base, variant="large")))
    validations = []
    monkeypatch.setattr(
        runner,
        "_batches",
        lambda campaign, workload, total_vram, progress=None: (
            (1, 2, 3),
            {
                batch: {
                    "status": "ok",
                    "step_ms": 10.0 if workload.variant == "base" else 6.0,
                    "peak_bytes": 100,
                }
                for batch in (1, 2, 3)
            },
        ),
    )

    def invoke(payload):
        if payload["kind"] != "model":
            return _successful_group(payload)
        validations.append((payload["workload"]["variant"], payload["batch"]))
        assert all(m["valid"] for m in payload["optimization"]["measurements"])
        assert len(payload["optimization"]["measurements"]) == 3
        return {"status": "ok", "step_ms": 8.0, "peak_bytes": 100}

    monkeypatch.setattr(runner, "_invoke", invoke)
    run = runner.execute_campaign(campaign)

    assert validations == [("base", 1), ("base", 3), ("large", 1), ("large", 3)]
    assert [m.valid for m in run.measurements] == [True, True, True, False, False, False]
    assert run.statistics.kernel_case_count == 3
    assert run.statistics.requested_case_count == 6


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
    run = runner.execute_campaign(_campaign_contexts())

    assert len(attempts) == run.statistics.kernel_group_count == 3
    assert run.statistics.requested_case_count == 4
    assert run.statistics.kernel_case_count == 2
    assert run.statistics.whole_model_validation_count == 2
    assert len(run.measurements) == 4
    for item in run.measurements:
        assert item.valid is (item.out_channels == 64)
        if item.out_channels == 128:
            assert item.reference_ms == item.candidate_ms == 0.0


def test_no_feasible_batches_launch_no_kernel_or_validation_children(monkeypatch):
    _prepare_campaign(monkeypatch, batches=())
    monkeypatch.setattr(runner, "_invoke", lambda payload: pytest.fail("unexpected child"))

    run = runner.execute_campaign(_campaign_contexts())

    assert run.measurements == ()
    assert run.statistics.requested_case_count == 0
    assert run.statistics.kernel_case_count == 0
    assert run.statistics.kernel_group_count == 0
    assert run.statistics.whole_model_validation_count == 0


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
