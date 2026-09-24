import json
from dataclasses import FrozenInstanceError

import pytest

from mednext_accel.profiling.campaign import campaign_to_primitive, load_campaign


def test_evidence_roundtrip_freezes_nested_environment_and_campaign():
    from mednext_accel.profiling import evidence

    environment = {"gpu": {"sm": [8, 6]}, "software": {"torch": "2.12"}}
    campaign = campaign_to_primitive(load_campaign({"seed": 31}))
    record = evidence.ProfilingEvidence(
        environment=evidence.EnvironmentEvidence(environment),
        campaign=evidence.CampaignEvidence(campaign),
    )
    environment["gpu"]["sm"][0] = 12
    document = record.to_primitive()
    assert document["kind"] == "mednext-accel-evidence"
    assert document["version"] == 1
    assert document["environment"]["gpu"]["sm"] == [8, 6]
    assert document["campaign"]["phase"] == "training"
    assert document["campaign"]["seed"] == 31
    assert evidence.ProfilingEvidence.from_primitive(json.loads(json.dumps(document))) == record
    with pytest.raises(TypeError):
        record.environment.data["gpu"]["sm"][0] = 12
    with pytest.raises(FrozenInstanceError):
        record.batch_searches = ()


@pytest.mark.parametrize(
    ("objective", "ms", "peak", "performance", "memory", "accepted"),
    [
        ("balanced", 99.0, 1150, True, True, True),
        ("balanced", 100.0, 1000, False, True, False),
        ("balanced", 90.0, 1151, True, False, False),
        ("throughput", 99.0, 1250, True, True, True),
        ("throughput", 99.0, 1251, True, False, False),
        ("memory", 110.0, 999, True, True, True),
        ("memory", 110.01, 999, False, True, False),
        ("memory", 90.0, 1000, True, False, False),
    ],
)
def test_model_objective_gates_are_independent(objective, ms, peak, performance, memory, accepted):
    from mednext_accel.profiling.whole_model import compare_model_results

    comparison = compare_model_results(
        objective,
        {"status": "ok", "step_ms": 100.0, "peak_bytes": 1000, "loss": 2.0},
        {"status": "ok", "step_ms": ms, "peak_bytes": peak, "loss": 2000.0},
        workload={"variant": "base"},
        batch=12,
        seed=31,
        effective_policy=({"name": "provisional", "sha256": "a" * 64},),
    )
    assert comparison.performance_accepted is performance
    assert comparison.memory_accepted is memory
    assert comparison.policy_accepted is accepted
    assert comparison.numerical_equivalence == "not-measured"
    assert comparison.reference.seed == comparison.candidate.seed == 31
    assert comparison.reference.diagnostics["loss"] == 2.0
    assert comparison.candidate.diagnostics["loss"] == 2000.0
    assert comparison.reason == (
        "accepted" if accepted else "performance rejected" if not performance else "memory rejected"
    )


@pytest.mark.parametrize("field", ["step_ms", "peak_bytes"])
@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_invalid_model_metrics_cannot_pass(field, value, side):
    from mednext_accel.profiling.whole_model import compare_model_results

    reference = {"status": "ok", "step_ms": 100.0, "peak_bytes": 1000}
    candidate = {"status": "ok", "step_ms": 90.0, "peak_bytes": 900}
    (reference if side == "reference" else candidate)[field] = value
    result = compare_model_results("balanced", reference, candidate, workload={}, batch=1, seed=0)
    assert not result.policy_accepted
    assert f"{side}: invalid {field}" in result.reason
    # Strict JSON persists missing/nonfinite values as null, with rejection reason retained.
    json.dumps(result.to_primitive(), allow_nan=False)


def test_failed_model_probe_preserves_failure_and_missing_metrics():
    from mednext_accel.profiling.whole_model import compare_model_results

    result = compare_model_results(
        "balanced",
        {"status": "ok", "step_ms": 10.0, "peak_bytes": 100},
        {"status": "oom", "message": "allocation failed", "failure_stage": "timing"},
        workload={},
        batch=1,
        seed=0,
    )
    assert result.candidate.status == "oom"
    assert result.candidate.step_ms is None
    assert result.candidate.message == "allocation failed"
    assert result.candidate.failure_stage == "timing"
    assert result.reason == "candidate: status oom"
    assert not result.performance_accepted and not result.memory_accepted


def test_kernel_evidence_copies_metrics_and_retains_numerics_on_timing_failure():
    from mednext_accel.profiling.matrix import KernelCase, KernelCaseKey
    from mednext_accel.profiling.synthesize import measurement_from_result

    case = KernelCase(
        KernelCaseKey(
            "pointwise_conv3d",
            "regular",
            "training",
            12,
            (16, 16, 16),
            32,
            64,
            1,
            "bfloat16",
            "pointwise_gemm_per_sample",
            (),
        )
    )
    metrics = {"dW": {"relative_l2": 0.001, "finite": True}}
    result = {
        "status": "oom",
        "valid": True,
        "validation_metrics": metrics,
        "failure_stage": "candidate_timing",
        "reference_ms": 10.0,
        "reference_peak_bytes": 100,
        "seed": 99,
    }
    record = measurement_from_result(case, result, checkpointing="none", objective="balanced")
    metrics["dW"]["relative_l2"] = 3.0
    assert record.kernel_valid is True
    assert record.validation_metrics["dW"]["relative_l2"] == 0.001
    assert record.reference_ms == 10.0
    assert record.failure_stage == "candidate_timing"
    assert record.objective_winner is False
    assert record.seed == 99
    assert "whole_model_valid" not in record.to_primitive()
    with pytest.raises(TypeError):
        record.validation_metrics["dW"]["relative_l2"] = 2.0


@pytest.mark.cuda
def test_model_probe_repeats_initialization_input_and_scalar_diagnostic(monkeypatch):
    import torch

    from mednext_accel.profiling import runner

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(
        runner, "_factory", lambda variant: lambda **kwargs: torch.nn.Conv3d(1, 2, 1)
    )
    monkeypatch.setattr(torch.nn.Module, "compile", lambda *args, **kwargs: None)
    payload = {
        "workload": {
            "variant": "small",
            "in_channels": 1,
            "out_channels": 2,
            "checkpointing": "none",
            "spatial": [8, 8, 8],
        },
        "batch": 1,
        "seed": 61,
    }
    first = runner._model_probe(payload)
    torch.manual_seed(291)
    second = runner._model_probe({**payload, "optimization": {"not_used_by_test_factory": True}})
    assert first["seed"] == second["seed"] == 61
    assert first["diagnostics"] == second["diagnostics"]
    assert first["diagnostics"]["loss"] > 0


def test_kernel_record_has_one_numerical_verdict():
    from mednext_accel.profiling.synthesize import Measurement

    assert "kernel_valid" in Measurement.__dataclass_fields__
    assert "valid" not in Measurement.__dataclass_fields__


def test_full_evidence_roundtrip_preserves_failed_probes_and_model_rejection():
    from mednext_accel.profiling.evidence import (
        BatchProbeEvidence,
        BatchSearchEvidence,
        CampaignEvidence,
        EnvironmentEvidence,
        ModelProbeEvidence,
        ProfilingEvidence,
    )
    from mednext_accel.profiling.whole_model import compare_model_results

    workload = {"model_family": "mednext_v1", "variant": "base", "spatial": [128] * 3}
    comparison = compare_model_results(
        "balanced",
        {"status": "ok", "step_ms": 10, "peak_bytes": 100},
        {"status": "ok", "step_ms": float("nan"), "peak_bytes": 110},
        workload=workload,
        batch=12,
        seed=0,
    )
    record = ProfilingEvidence(
        EnvironmentEvidence({"gpu": {"sm": [8, 6]}}),
        CampaignEvidence(campaign_to_primitive(load_campaign(None))),
        (
            BatchSearchEvidence(
                workload,
                100,
                (
                    BatchProbeEvidence(
                        16,
                        ModelProbeEvidence("oom", 0, message="OOM"),
                        False,
                        False,
                        "probe: status oom",
                    ),
                    BatchProbeEvidence(12, ModelProbeEvidence("ok", 0, 10.0, 90), True, True),
                ),
                12,
            ),
        ),
        model_comparisons=(comparison,),
    )
    encoded = json.dumps(record.to_primitive(), allow_nan=False)
    decoded = ProfilingEvidence.from_primitive(json.loads(encoded))
    assert decoded == record
    assert list(decoded.batch_searches[0].by_batch) == [16, 12]
    assert decoded.model_comparisons[0].policy_accepted is False


def test_grn_evidence_roundtrip_preserves_nullable_kernel_and_diagnostics(grn_case, grn_result):
    from mednext_accel.profiling.evidence import (
        CampaignEvidence,
        EnvironmentEvidence,
        ProfilingEvidence,
    )
    from mednext_accel.profiling.synthesize import measurement_from_result

    record = measurement_from_result(grn_case, grn_result, checkpointing="none")
    source = ProfilingEvidence(
        EnvironmentEvidence({}),
        CampaignEvidence(
            campaign_to_primitive(load_campaign({"workloads": [{"family": "mednext_v2"}]}))
        ),
        kernel_measurements=(record,),
    )
    data = json.loads(json.dumps(source.to_primitive()))
    assert data["kernel_measurements"][0]["kernel_size"] is None
    assert data["kernel_measurements"][0]["dgamma_ms"] == {"reference": 0.4, "candidate": 0.2}
    assert ProfilingEvidence.from_primitive(data) == source
    with pytest.raises(TypeError):
        record.dx_ms["candidate"] = 10


@pytest.mark.parametrize("field", ["forward_ms", "dx_ms", "dgamma_ms", "dbeta_ms"])
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        {"candidate": 0.2},
        {"reference": 1, "candidate": 0},
        {"reference": 1, "candidate": float("nan")},
    ],
)
def test_successful_grn_requires_complete_positive_component_diagnostics(
    grn_case, grn_result, field, invalid
):
    from mednext_accel.profiling.synthesize import measurement_from_result

    grn_result[field] = invalid
    with pytest.raises(ValueError, match=field):
        measurement_from_result(grn_case, grn_result, checkpointing="none")


def test_partial_grn_failure_retains_completed_diagnostics(grn_case):
    from mednext_accel.profiling.synthesize import measurement_from_result

    record = measurement_from_result(
        grn_case,
        {
            "status": "oom",
            "forward_ms": {"reference": 0.4},
            "benchmark_kind": "integrated_operator",
            "gradient_mask": (True, True, True),
        },
        checkpointing="none",
    )
    assert record.kernel_size is None
    assert record.forward_ms == {"reference": 0.4}
    assert record.objective_winner is False
