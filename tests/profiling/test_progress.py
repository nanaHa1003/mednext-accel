from io import StringIO

from mednext_accel.profiling.progress import (
    PlainProgressReporter,
    ProgressEvent,
    RichProgressReporter,
)


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


def test_plain_unknown_total_reports_count_and_average_without_eta() -> None:
    times = iter((0.0, 20.0, 50.0))
    output = StringIO()
    reporter = PlainProgressReporter(output, clock=lambda: next(times))

    reporter.emit(ProgressEvent("stage_start", "batch-search"))
    reporter.emit(ProgressEvent("advance", "batch-search", completed=1))
    reporter.emit(ProgressEvent("advance", "batch-search", completed=2))

    text = output.getvalue()
    assert "1/?" in text
    assert "2/?" in text
    assert "25.0 s/probe" in text
    assert "%" not in text
    assert "ETA" not in text


def test_plain_group_progress_reports_physical_and_logical_counts() -> None:
    times = iter((0.0, 70.0))
    output = StringIO()
    reporter = PlainProgressReporter(output, clock=lambda: next(times))

    reporter.emit(ProgressEvent("stage_start", "kernel-groups", total=20))
    reporter.emit(
        ProgressEvent(
            "advance",
            "kernel-groups",
            completed=7,
            total=20,
            secondary_completed=71,
            secondary_total=192,
        )
    )

    text = output.getvalue()
    assert "groups 7/20" in text
    assert "experiments 71/192" in text
    assert "10.0 s/group" in text
    assert "ETA 00:02:10" in text


def test_rich_progress_exposes_unknown_and_secondary_counts_without_unknown_eta() -> None:
    batch_output = StringIO()
    reporter = RichProgressReporter(batch_output)

    reporter.emit(ProgressEvent("stage_start", "batch-search"))
    reporter.emit(
        ProgressEvent(
            "advance", "batch-search", completed=2, message="batch=3 feasible peak=1.0 GiB"
        )
    )
    reporter.close()

    batch_text = batch_output.getvalue()
    assert "probes 2/?" in batch_text
    assert "batch=3 feasible peak=1.0 GiB" in batch_text
    assert "%" not in batch_text
    assert "ETA" not in batch_text

    group_output = StringIO()
    reporter = RichProgressReporter(group_output)
    reporter.emit(ProgressEvent("stage_start", "kernel-groups", total=4))
    reporter.emit(
        ProgressEvent(
            "advance",
            "kernel-groups",
            completed=2,
            total=4,
            secondary_completed=7,
            secondary_total=9,
        )
    )
    reporter.close()

    group_text = group_output.getvalue()
    assert "groups 2/4" in group_text
    assert "experiments 7/9" in group_text
    assert "s/group" in group_text
    assert "ETA" in group_text
