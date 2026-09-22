import sys

import pytest

from mednext_accel.profiling import runner
from mednext_accel.profiling.batch_search import BatchSearch
from mednext_accel.profiling.benchmark import run_json_subprocess
from mednext_accel.profiling.campaign import Campaign, Workload
from mednext_accel.profiling.evidence import ModelProbeEvidence
from mednext_accel.profiling.grouped import run_group_with_bisection
from mednext_accel.profiling.matrix import KernelCase, KernelCaseKey, KernelGroup
from mednext_accel.profiling.synthesize import measurement_from_result


def case():
    return KernelCase(
        KernelCaseKey(
            "pointwise_conv3d",
            "regular",
            "training",
            1,
            (16, 16, 16),
            32,
            64,
            1,
            "bfloat16",
            "pointwise_gemm_per_sample",
            (),
        )
    )


def test_silent_nonzero_subprocess_is_preserved_by_batch_and_kernel_profiling(monkeypatch):
    def child(*args, **kwargs):
        return run_json_subprocess([sys.executable, "-c", "raise SystemExit(7)"], timeout=5)

    monkeypatch.setattr(runner, "run_json_subprocess", child)
    workload = Workload("mednext_v1", "small", (32, 32, 32))
    campaign = Campaign("mednext-v1", (workload,), BatchSearch(maximum=1))
    selected = runner._batches(campaign, workload, 1024)
    assert selected.batches == ()
    probe = selected.evidence.attempts[0]
    assert not probe.feasible
    assert probe.result.status == "infrastructure_error"
    assert probe.result.message == "probe exited with status 7 without output"
    target = case()
    raw = run_group_with_bisection(KernelGroup("pointwise", 1, (target,)), runner._invoke)
    measured = measurement_from_result(target, raw[target.identifier], checkpointing="none")
    assert measured.probe_status == "infrastructure_error"
    assert measured.probe_message == probe.result.message
    assert measured.kernel_valid is None and measured.objective_winner is False


@pytest.mark.parametrize("message", [None, "", "  \n\t"])
@pytest.mark.parametrize("status", ["oom", "error", "timeout", "infrastructure_error"])
def test_empty_failure_diagnostics_are_normalized_at_both_result_boundaries(status, message):
    raw = {"status": status, "message": message}
    model = ModelProbeEvidence.from_result(raw, seed=0)
    measured = measurement_from_result(case(), raw, checkpointing="none")
    expected = f"probe reported {status} without a diagnostic"
    assert model.message == measured.probe_message == expected
    assert model.status == measured.probe_status == status
    assert measured.objective_winner is False


@pytest.mark.parametrize("message", [None, "", " \n"])
def test_structured_child_failure_always_has_a_diagnostic(message):
    import json

    payload = json.dumps({"status": "error", "message": message})
    result = run_json_subprocess([sys.executable, "-c", f"print({payload!r})"], timeout=5)
    assert result.status == "error"
    assert result.message == "probe reported error without a diagnostic"


def test_launch_error_with_no_text_is_recorded(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError()

    monkeypatch.setattr("subprocess.run", fail)
    result = run_json_subprocess(["unavailable"], timeout=5)
    assert result.status == "infrastructure_error"
    assert result.message == "OSError: could not start probe"


def test_loaded_evidence_keeps_strict_nonempty_message_invariant():
    with pytest.raises(ValueError, match="message"):
        ModelProbeEvidence("error", 0, message="")


@pytest.mark.parametrize(
    "stdout", ["", " \n", "not-json", "[]", '{"status": 1}', '{"status": "error", "message": 1}']
)
def test_every_malformed_subprocess_response_has_a_diagnostic(monkeypatch, stdout):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )
    result = run_json_subprocess(["probe"], timeout=5)
    assert result.status == "infrastructure_error"
    assert result.message and result.message.strip()
    raw = {"status": result.status, "message": result.message}
    assert ModelProbeEvidence.from_result(raw, seed=0).message == result.message
    assert (
        measurement_from_result(case(), raw, checkpointing="none").probe_message == result.message
    )


def test_timeout_failure_is_preserved_at_both_result_boundaries(monkeypatch):
    import subprocess

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("probe", 5)

    monkeypatch.setattr(subprocess, "run", timeout)
    result = run_json_subprocess(["probe"], timeout=5)
    assert result.status == "timeout"
    assert result.message == "probe exceeded 5 seconds"
    raw = {"status": result.status, "message": result.message}
    assert ModelProbeEvidence.from_result(raw, seed=0).message == result.message
    assert (
        measurement_from_result(case(), raw, checkpointing="none").probe_message == result.message
    )
