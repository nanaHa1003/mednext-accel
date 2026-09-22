from dataclasses import replace

import pytest

from mednext_accel.profiling.synthesize import (
    Measurement,
    candidate_wins,
    synthesize_profile,
)


def measured(batch: int, reference_ms: float, candidate_ms: float) -> Measurement:
    return Measurement(
        family="pointwise_conv3d",
        direction="regular",
        phase="training",
        implementation="pointwise_gemm_per_sample",
        batch=batch,
        spatial_shape=(128, 128, 128),
        in_channels=32,
        out_channels=64,
        dtype="bfloat16",
        checkpointing="all-expansion",
        reference_ms=reference_ms,
        candidate_ms=candidate_ms,
        reference_peak_bytes=100,
        candidate_peak_bytes=100,
        kernel_valid=True,
    )


def test_adjacent_measured_batches_remain_exact_local_rules() -> None:
    profile = synthesize_profile(
        [measured(2, 4.0, 3.0), measured(3, 6.0, 4.5), measured(4, 8.0, 6.2)],
        name="test",
        sm=(12, 0),
        objective="balanced",
    )
    assert len(profile.rules) == 3
    assert [(rule.when["batch"].minimum, rule.when["batch"].maximum) for rule in profile.rules] == [
        (2, 2),
        (3, 3),
        (4, 4),
    ]
    assert profile.rules[0].use["training"].implementation == "pointwise_gemm_per_sample"


def test_invalid_candidate_and_memory_objective_choose_safely() -> None:
    invalid = measured(2, 4.0, 1.0)
    invalid = Measurement(**{**invalid.to_primitive(), "kernel_valid": False})
    memory = measured(3, 4.0, 4.1)
    memory = Measurement(
        **{
            **memory.to_primitive(),
            "reference_peak_bytes": 200,
            "candidate_peak_bytes": 100,
        }
    )
    profile = synthesize_profile([invalid, memory], name="test", sm=(12, 0), objective="memory")
    assert [r.use["training"].implementation for r in profile.rules] == [
        "reference",
        "pointwise_gemm_per_sample",
    ]
    assert profile.rules[1].when["batch"].minimum == 3


def test_reported_winner_uses_the_same_balanced_threshold_as_synthesis() -> None:
    assert not candidate_wins(measured(2, 100.0, 99.0), "balanced")
    assert candidate_wins(measured(2, 100.0, 96.0), "balanced")


@pytest.mark.parametrize(
    "selection",
    [{}, {"implementation": "reference"}, {"parameters": (("tile", 32),)}],
)
def test_invalid_rule_match_blocks_other_selections_for_the_same_match(selection) -> None:
    candidate = measured(2, 4.0, 1.0)
    rejected = replace(candidate, kernel_valid=False, **selection)

    profile = synthesize_profile(
        [candidate, rejected], name="test", sm=(12, 0), objective="balanced"
    )

    assert len(profile.rules) == 1
    assert profile.rules[0].use["training"].implementation == "reference"
    assert candidate.kernel_valid is True


@pytest.mark.parametrize(
    "different_match",
    [
        {"family": "depthwise_conv3d"},
        {"direction": "downsample"},
        {"phase": "inference"},
        {"batch": 3},
        {"spatial_shape": (64, 64, 64)},
        {"in_channels": 16},
        {"out_channels": 128},
        {"dtype": "float32"},
        {"checkpointing": "none"},
    ],
)
def test_invalid_rule_match_preserves_distinguishable_candidate(different_match) -> None:
    candidate = measured(2, 4.0, 1.0)
    rejected = replace(candidate, kernel_valid=False, **different_match)

    profile = synthesize_profile(
        [candidate, rejected], name="test", sm=(12, 0), objective="balanced"
    )

    assert len(profile.rules) == 2
    assert (
        sum(
            rule.use.get("training", rule.use.get("inference")).implementation
            == "pointwise_gemm_per_sample"
            for rule in profile.rules
        )
        == 1
    )


def test_raw_result_materialization_restores_context_and_uses_planned_identity():
    from mednext_accel.profiling import synthesize
    from mednext_accel.profiling.matrix import KernelCase, KernelCaseKey

    case = KernelCase(
        KernelCaseKey(
            family="depthwise_conv3d",
            direction="regular",
            phase="backward_weight",
            batch=2,
            spatial_shape=(128, 128, 128),
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            dtype="bfloat16",
            implementation="triton_split_dw",
            parameters=(("dw_splits", 128), ("dw_block", 512)),
        )
    )
    raw = {
        "status": "ok",
        "valid": True,
        "reference_ms": 10,
        "candidate_ms": 8,
        "reference_peak_bytes": 100,
        "candidate_peak_bytes": 90,
        "parameters": [],
        "implementation": "untrusted",
        "checkpointing": "none",
    }
    item = synthesize.measurement_from_result(case, raw, checkpointing="all-expansion")

    assert item.checkpointing == "all-expansion"
    assert item.parameters == (("dw_splits", 128), ("dw_block", 512))
    assert item.implementation == "triton_split_dw"
    assert item.kernel_valid and item.reference_ms == 10.0 and item.candidate_ms == 8.0
    assert item.reference_peak_bytes == 100 and item.candidate_peak_bytes == 90
    assert raw["checkpointing"] == "none"

    for failed in ({}, {**raw, "status": "error"}):
        item = synthesize.measurement_from_result(case, failed, checkpointing="none")
        assert not candidate_wins(item, "balanced")
        assert item.reference_ms == failed.get("reference_ms", 0.0)
        assert item.candidate_peak_bytes == failed.get("candidate_peak_bytes", 0)


def test_single_selected_batch_produces_only_an_exact_batch_rule() -> None:
    profile = synthesize_profile(
        [measured(13, 10.0, 8.0)], name="selected", sm=(12, 0), objective="balanced"
    )

    assert len(profile.rules) == 1
    assert profile.rules[0].when["batch"].minimum == profile.rules[0].when["batch"].maximum == 13


def test_measurement_serializes_probe_and_numerical_diagnostics():
    from mednext_accel.optimization.policy import parse_policy, policy_to_primitive
    from mednext_accel.profiling.matrix import KernelCase, KernelCaseKey
    from mednext_accel.profiling.synthesize import measurement_from_result

    case = KernelCase(
        KernelCaseKey(
            family="pointwise_conv3d",
            direction="regular",
            phase="training",
            batch=2,
            spatial_shape=(16, 16, 16),
            in_channels=8,
            out_channels=16,
            kernel_size=1,
            dtype="bfloat16",
            implementation="pointwise_gemm_per_sample",
            parameters=(),
        )
    )
    metrics = {"dW": {"finite": True, "relative_l2": 0.03, "max_absolute": 0.5}}
    raw = {
        "status": "ok",
        "message": "completed",
        "valid": False,
        "validator": "component-relative-l2-v1",
        "validation_metrics": metrics,
        "rejection_reason": "dW: relative L2 error must be below 0.02",
    }
    item = measurement_from_result(case, raw, checkpointing="none")
    data = item.to_primitive()
    assert data["probe_status"] == "ok"
    assert data["probe_message"] == "completed"
    assert data["kernel_valid"] is False
    assert "whole_model_valid" not in data
    assert data["validator"] == "component-relative-l2-v1"
    assert data["validation_metrics"] == metrics
    assert data["rejection_reason"] == raw["rejection_reason"]
    profile = synthesize_profile([item], name="diagnostic", sm=(8, 9), objective="balanced")
    roundtrip = parse_policy(policy_to_primitive(profile))
    assert roundtrip.rules[0].use["training"].implementation == "reference"
    assert item.to_primitive()["kernel_valid"] is False

    failure = measurement_from_result(
        case, {"status": "oom", "message": "CUDA out of memory"}, checkpointing="none"
    ).to_primitive()
    assert failure["probe_status"] == "oom"
    assert failure["probe_message"] == "CUDA out of memory"
    assert failure["kernel_valid"] is None
    assert failure["kernel_valid"] is None


def test_legacy_direct_measurement_defaults_preserve_synthesis():
    item = measured(1, 4.0, 3.0)
    assert item.kernel_valid is True
    assert not hasattr(item, "whole_model_valid")
    assert item.validator is None
    assert item.validation_metrics == {}
    assert candidate_wins(item, "balanced")


def test_conflicting_policy_contexts_do_not_rewrite_kernel_evidence():
    candidate = measured(2, 4.0, 1.0)
    rejected = replace(candidate, kernel_valid=False, rejection_reason="numerical failure")
    before = [item.to_primitive() for item in (candidate, rejected)]
    profile = synthesize_profile(
        [candidate, rejected], name="conflicting", sm=(12, 0), objective="balanced"
    )
    assert len(profile.rules) == 1
    assert profile.rules[0].use["training"].implementation == "reference"
    assert before == [item.to_primitive() for item in (candidate, rejected)]
