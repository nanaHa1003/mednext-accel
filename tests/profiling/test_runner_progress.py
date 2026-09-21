from mednext_accel.profiling import runner
from mednext_accel.profiling.batch_search import BatchSearch
from mednext_accel.profiling.campaign import Campaign, Workload
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


def test_runner_reports_workload_and_counted_operator_validation(
    monkeypatch, capsys
) -> None:
    workload = Workload("mednext_v1", "base", (128, 128, 128))
    campaign = Campaign("mednext-v1", (workload,), BatchSearch(maximum=1))
    reporter = _Recorder()
    monkeypatch.setattr(runner, "_batches", lambda *args, **kwargs: (
        (1,), {1: {"status": "ok", "step_ms": 10.0, "peak_bytes": 100}}
    ))
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
