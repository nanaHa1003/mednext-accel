"""Untrusted JSON cannot manufacture numerical or policy acceptance evidence."""

from copy import deepcopy
from dataclasses import replace

import pytest

from mednext_accel.profiling.campaign import campaign_to_primitive, load_campaign
from mednext_accel.profiling.evidence import (
    BatchProbeEvidence,
    BatchSearchEvidence,
    CampaignEvidence,
    EnvironmentEvidence,
    ModelProbeEvidence,
    ProfilingEvidence,
)
from mednext_accel.profiling.matrix import KernelCase, KernelCaseKey
from mednext_accel.profiling.synthesize import measurement_from_result
from mednext_accel.profiling.whole_model import compare_model_results


def kernel_result():
    case = KernelCase(
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
    return case, {
        "status": "ok",
        "valid": True,
        "reference_ms": 10.0,
        "candidate_ms": 8.0,
        "reference_peak_bytes": 100,
        "candidate_peak_bytes": 90,
    }


def document():
    case, raw = kernel_result()
    reference = {"status": "ok", "step_ms": 10.0, "peak_bytes": 100}
    candidate = {"status": "ok", "step_ms": 8.0, "peak_bytes": 90}
    return ProfilingEvidence(
        EnvironmentEvidence({"gpu": {"sm": [8, 6]}}),
        CampaignEvidence(campaign_to_primitive(load_campaign(None))),
        (
            BatchSearchEvidence(
                {},
                100,
                (
                    BatchProbeEvidence(
                        2, ModelProbeEvidence("oom", 0), False, False, "probe: status oom"
                    ),
                    BatchProbeEvidence(
                        1, ModelProbeEvidence.from_result(reference, seed=0), True, True
                    ),
                ),
                1,
            ),
        ),
        (measurement_from_result(case, raw, checkpointing="none"),),
        (compare_model_results("balanced", reference, candidate, workload={}, batch=1, seed=0),),
    ).to_primitive()


@pytest.mark.parametrize("value", ["false", "true", 1, 0, [], {}, None])
def test_non_boolean_kernel_child_verdict_is_rejected(value):
    case, raw = kernel_result()
    raw["valid"] = value
    with pytest.raises(ValueError, match="valid.*bool"):
        measurement_from_result(case, raw, checkpointing="none")


@pytest.mark.parametrize("field", ["kernel_valid", "objective_winner"])
@pytest.mark.parametrize("value", ["false", 1, 0])
def test_kernel_evidence_verdicts_cannot_be_coerced(field, value):
    data = document()
    data["kernel_measurements"][0][field] = value
    with pytest.raises(ValueError, match=field):
        ProfilingEvidence.from_primitive(data)


@pytest.mark.parametrize(
    "change",
    [
        {"numerical_equivalence": "passed"},
        {"policy_accepted": False},
        {"performance_accepted": False, "policy_accepted": False},
        {"memory_accepted": False, "policy_accepted": False},
        {"performance_accepted": 1},
        {"memory_accepted": "true"},
        {"policy_accepted": 1},
        {"reason": "memory rejected"},
        {"objective": "fastest"},
        {"protocol": "unmeasured"},
    ],
)
def test_comparison_fixed_semantics_and_derived_gates_are_enforced(change):
    data = document()
    data["model_comparisons"][0].update(change)
    with pytest.raises(ValueError):
        ProfilingEvidence.from_primitive(data)


@pytest.mark.parametrize(
    "side,field,value",
    [
        ("candidate", "step_ms", 20),
        ("candidate", "peak_bytes", 200),
        ("reference", "status", "oom"),
        ("reference", "seed", 2),
    ],
)
def test_comparison_gates_must_agree_with_recorded_metrics(side, field, value):
    data = document()
    data["model_comparisons"][0][side][field] = value
    with pytest.raises(ValueError):
        ProfilingEvidence.from_primitive(data)


@pytest.mark.parametrize(
    "change",
    [
        {"selected_maximum": 2},
        {"selected_maximum": 0},
        {"selected_maximum": True},
        {"budget_bytes": 99},
    ],
)
def test_batch_selection_and_budget_must_agree_with_attempts(change):
    data = document()
    data["batch_searches"][0].update(change)
    with pytest.raises(ValueError):
        ProfilingEvidence.from_primitive(data)


def test_duplicate_batch_attempts_are_rejected_even_if_summary_agrees():
    data = document()
    data["batch_searches"][0]["attempts"].append(deepcopy(data["batch_searches"][0]["attempts"][1]))
    with pytest.raises(ValueError, match="duplicate"):
        ProfilingEvidence.from_primitive(data)


@pytest.mark.parametrize("value", [False, 1])
def test_indexed_summary_cannot_disagree_with_ordered_attempts(value):
    data = document()
    data["batch_searches"][0]["by_batch"]["1"]["feasible"] = value
    with pytest.raises(ValueError, match="by_batch"):
        ProfilingEvidence.from_primitive(data)


@pytest.mark.parametrize(
    "field,value",
    [("feasible", 1), ("within_budget", "false"), ("feasible", False), ("reason", "unexplained")],
)
def test_batch_attempt_flags_and_reason_are_consistent(field, value):
    data = document()
    attempt = data["batch_searches"][0]["attempts"][1]
    attempt[field] = value
    data["batch_searches"][0]["by_batch"]["1"] = deepcopy(attempt)
    with pytest.raises(ValueError):
        ProfilingEvidence.from_primitive(data)


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("batch_searches", 0),
        ("batch_searches", 0, "attempts", 0),
        ("batch_searches", 0, "attempts", 0, "result"),
        ("model_comparisons", 0),
        ("model_comparisons", 0, "reference"),
        ("kernel_measurements", 0),
    ],
)
def test_unknown_record_fields_are_rejected(path):
    data = document()
    target = data
    for key in path:
        target = target[key]
    target["unrecognized"] = 1
    with pytest.raises(ValueError, match="unknown"):
        ProfilingEvidence.from_primitive(data)


@pytest.mark.parametrize("value", [True, "1", 1.0])
def test_evidence_version_requires_exact_integer(value):
    data = document()
    data["version"] = value
    with pytest.raises(ValueError, match="version"):
        ProfilingEvidence.from_primitive(data)


@pytest.mark.parametrize(
    "field,value",
    [("seed", "0"), ("seed", True), ("status", 1), ("diagnostics", []), ("message", 2)],
)
def test_model_child_metadata_is_not_coerced(field, value):
    with pytest.raises(ValueError, match=field):
        ModelProbeEvidence.from_result({"status": "ok", field: value}, seed=0)


def test_direct_record_construction_cannot_bypass_verdict_invariants():
    case, raw = kernel_result()
    measurement = measurement_from_result(case, raw, checkpointing="none")
    with pytest.raises(ValueError, match="kernel_valid"):
        replace(measurement, kernel_valid="false")
    with pytest.raises(ValueError, match="objective_winner"):
        replace(measurement, objective_winner=False)
    with pytest.raises(ValueError, match="numerical_equivalence"):
        replace(
            compare_model_results(
                "balanced", {"status": "error"}, {"status": "error"}, workload={}, batch=1, seed=0
            ),
            numerical_equivalence="passed",
        )


def test_valid_complete_document_has_stable_roundtrip():
    data = document()
    assert ProfilingEvidence.from_primitive(data).to_primitive() == data


@pytest.mark.parametrize(
    "path",
    [
        ("environment",),
        ("environment", "gpu"),
        ("campaign",),
        ("campaign", "workloads", 0),
        ("batch_searches", 0, "workload"),
        ("model_comparisons", 0, "workload"),
        ("kernel_measurements", 0, "workload"),
    ],
)
def test_unknown_metadata_fields_are_rejected(path):
    data = document()
    target = data
    for key in path:
        target = target[key]
    target["unrecognized"] = 1
    with pytest.raises(ValueError, match="unknown"):
        ProfilingEvidence.from_primitive(data)


@pytest.mark.parametrize(
    "key,value", [("phase", "inference"), ("seed", "0"), ("objective", "memory")]
)
def test_campaign_identity_cannot_contradict_recorded_comparison(key, value):
    data = document()
    data["campaign"][key] = value
    with pytest.raises(ValueError):
        ProfilingEvidence.from_primitive(data)


@pytest.mark.parametrize(
    "key,value",
    [
        ("finite", "true"),
        ("relative_l2", -0.1),
        ("relative_l2", 0.05),
        ("max_absolute", float("nan")),
        ("surprise", 0),
    ],
)
def test_kernel_component_metrics_are_typed_and_consistent(key, value):
    data = document()
    metrics = {"finite": True, "relative_l2": 0.001, "max_absolute": 0.01, key: value}
    data["kernel_measurements"][0]["validation_metrics"] = {"dW": metrics}
    with pytest.raises(ValueError):
        ProfilingEvidence.from_primitive(data)


@pytest.mark.parametrize("key,value", [("name", 1), ("sha256", "bad"), ("extra", 0)])
def test_effective_policy_identity_rejects_malformed_layers(key, value):
    data = document()
    data["model_comparisons"][0]["effective_policy"] = [
        {"name": "overlay", "sha256": "a" * 64, key: value}
    ]
    with pytest.raises(ValueError):
        ProfilingEvidence.from_primitive(data)


@pytest.mark.parametrize(
    "objective,ms,peak", [("balanced", 9.9, 115), ("throughput", 9.9, 125), ("memory", 11.0, 99)]
)
def test_objective_boundaries_roundtrip_without_reinterpreting_gates(objective, ms, peak):
    data = document()
    data["campaign"]["objective"] = objective
    data["model_comparisons"][0] = compare_model_results(
        objective,
        {"status": "ok", "step_ms": 10, "peak_bytes": 100},
        {"status": "ok", "step_ms": ms, "peak_bytes": peak},
        workload={},
        batch=1,
        seed=0,
    ).to_primitive()
    assert ProfilingEvidence.from_primitive(data).to_primitive() == data


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_batch_search_probe_feasibility_requires_a_boolean(value):
    from mednext_accel.profiling.batch_search import ProbeResult

    with pytest.raises(ValueError, match="feasible.*bool"):
        ProbeResult(1, value, 100)


@pytest.mark.parametrize(
    "metrics",
    [
        {"finite": True, "relative_l2": None},
        {"finite": True, "relative_l2": 0.001, "max_absolute": float("inf")},
        {"relative_l2": 0.001},
    ],
)
def test_child_cannot_claim_numerical_success_with_invalid_component_diagnostics(metrics):
    case, raw = kernel_result()
    raw["validation_metrics"] = {"dW": metrics}
    with pytest.raises(ValueError, match="kernel_valid"):
        measurement_from_result(case, raw, checkpointing="none")
