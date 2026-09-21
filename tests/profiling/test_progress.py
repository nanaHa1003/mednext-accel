from io import StringIO

from mednext_accel.profiling.progress import PlainProgressReporter, ProgressEvent


def test_plain_progress_reports_completion_elapsed_and_eta() -> None:
    times = iter((0.0, 10.0, 20.0))
    output = StringIO()
    reporter = PlainProgressReporter(output, clock=lambda: next(times))

    reporter.emit(ProgressEvent("stage_start", "operators", total=4))
    reporter.emit(
        ProgressEvent(
            "advance",
            "operators",
            completed=1,
            total=4,
            message="depthwise_conv3d/backward_weight",
        )
    )
    reporter.emit(
        ProgressEvent(
            "advance",
            "operators",
            completed=2,
            total=4,
            message="pointwise_conv3d/training",
        )
    )

    text = output.getvalue()
    assert "1/4" in text
    assert "25.0%" in text
    assert "elapsed 00:00:10" in text
    assert "ETA 00:00:30" in text
    assert "2/4" in text
    assert "ETA 00:00:20" in text


def test_plain_progress_shows_batch_search_without_inventing_eta() -> None:
    times = iter((0.0, 12.0))
    output = StringIO()
    reporter = PlainProgressReporter(output, clock=lambda: next(times))

    reporter.emit(ProgressEvent("stage_start", "batch-search"))
    reporter.emit(
        ProgressEvent(
            "status",
            "batch-search",
            message="batch=3 feasible peak=31.4 GiB",
        )
    )

    assert "estimating workload" in output.getvalue()
    assert "elapsed 00:00:12" in output.getvalue()
    assert "ETA" not in output.getvalue()
