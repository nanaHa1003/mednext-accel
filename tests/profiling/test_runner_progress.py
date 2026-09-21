import gc

from mednext_accel.profiling import runner
from mednext_accel.profiling.batch_search import BatchSearch
from mednext_accel.profiling.campaign import Campaign, Workload
from mednext_accel.profiling.grouped import case_payload
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


def test_runner_reports_workload_and_counted_operator_validation(monkeypatch, capsys) -> None:
    workload = Workload("mednext_v1", "base", (128, 128, 128))
    campaign = Campaign("mednext-v1", (workload,), BatchSearch(maximum=1))
    reporter = _Recorder()
    monkeypatch.setattr(
        runner,
        "_batches",
        lambda *args, **kwargs: ((1,), {1: {"status": "ok", "step_ms": 10.0, "peak_bytes": 100}}),
    )
    monkeypatch.setattr(runner, "_pointwise_shapes", lambda workload: ())
    monkeypatch.setattr(runner, "_depthwise_shapes", lambda workload: ())
    monkeypatch.setattr(
        runner,
        "_invoke",
        lambda payload: {"status": "ok", "step_ms": 8.0, "peak_bytes": 100},
    )
    monkeypatch.setattr(runner, "_whole_model_accepts", lambda *args: False)
    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch())

    assert runner.run_campaign(campaign, progress=reporter) == ()

    assert [(event.kind, event.stage) for event in reporter.events] == [
        ("workload", "workload"),
        ("stage_start", "batch-search"),
        ("stage_start", "operators"),
        ("item_start", "operators"),
        ("advance", "operators"),
    ]
    assert reporter.events[2].total == 1
    assert "whole-model" in reporter.events[3].message
    assert reporter.events[4].completed == 1
    assert "whole-model" in reporter.events[4].message
    assert "rejected" in reporter.events[4].message
    assert capsys.readouterr().err == ""
