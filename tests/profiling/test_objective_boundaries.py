"""Exact inclusive ratios and strict inequalities for recorded numeric values."""

import math
from dataclasses import replace

import pytest

from mednext_accel.profiling.evidence import ModelProbeEvidence, model_acceptance
from mednext_accel.profiling.synthesize import Measurement, candidate_wins


def kernel(*, reference_ms=10.0, candidate_ms=8.0, reference_peak=100, candidate_peak=90):
    return Measurement(
        "pointwise_conv3d",
        "regular",
        "training",
        "pointwise_gemm_per_sample",
        1,
        (16, 16, 16),
        32,
        64,
        "bfloat16",
        "none",
        reference_ms,
        candidate_ms,
        reference_peak,
        candidate_peak,
        True,
        benchmark_kind="integrated_operator",
        memory_measured=True,
    )


@pytest.mark.parametrize(
    "objective,numerator,denominator", [("balanced", 23, 20), ("throughput", 5, 4)]
)
@pytest.mark.parametrize("reference_peak", [20, 100, 1000, 2**60 * 20])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_model_memory_ratio_boundaries_are_exact(
    objective, numerator, denominator, reference_peak, delta
):
    boundary = reference_peak * numerator // denominator
    reference = ModelProbeEvidence("ok", 0, 10.0, reference_peak)
    candidate = ModelProbeEvidence("ok", 0, 8.0, boundary + delta)
    performance, memory, _ = model_acceptance(objective, reference, candidate, 0)
    assert performance
    assert memory is (delta <= 0)


@pytest.mark.parametrize("reference_peak", [20, 100, 1000, 2**60 * 20])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_kernel_balanced_memory_ratio_boundary_is_exact(reference_peak, delta):
    boundary = reference_peak * 23 // 20
    assert candidate_wins(
        kernel(reference_peak=reference_peak, candidate_peak=boundary + delta), "balanced"
    ) is (delta <= 0)


@pytest.mark.parametrize(
    "reference_time,boundary", [(1.0, 1.1), (3.0, 3.3), (10.0, 11.0), (100.0, 110.0)]
)
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_model_and_kernel_memory_timing_limit_is_inclusive_without_rounding_slack(
    reference_time, boundary, direction
):
    candidate_time = (
        boundary
        if direction == 0
        else math.nextafter(boundary, math.inf if direction > 0 else -math.inf)
    )
    reference = ModelProbeEvidence("ok", 0, reference_time, 100)
    candidate = ModelProbeEvidence("ok", 0, candidate_time, 90)
    performance, memory, _ = model_acceptance("memory", reference, candidate, 0)
    assert memory
    assert performance is (direction <= 0)
    assert candidate_wins(
        kernel(reference_ms=reference_time, candidate_ms=candidate_time), "memory"
    ) is (direction <= 0)


@pytest.mark.parametrize(
    "reference_time,boundary", [(1.0, 0.97), (3.0, 2.91), (10.0, 9.7), (100.0, 97.0)]
)
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_kernel_balanced_speed_limit_is_inclusive_without_rounding_slack(
    reference_time, boundary, direction
):
    candidate_time = (
        boundary
        if direction == 0
        else math.nextafter(boundary, math.inf if direction > 0 else -math.inf)
    )
    assert candidate_wins(
        kernel(reference_ms=reference_time, candidate_ms=candidate_time), "balanced"
    ) is (direction <= 0)


@pytest.mark.parametrize("direction", [-1, 0, 1])
@pytest.mark.parametrize("objective", ["balanced", "throughput"])
def test_model_time_improvement_remains_strict(objective, direction):
    candidate_time = (
        10.0 if direction == 0 else math.nextafter(10.0, math.inf if direction > 0 else -math.inf)
    )
    performance, memory, _ = model_acceptance(
        objective,
        ModelProbeEvidence("ok", 0, 10.0, 100),
        ModelProbeEvidence("ok", 0, candidate_time, 90),
        0,
    )
    assert memory
    assert performance is (direction < 0)
    if objective == "throughput":
        assert candidate_wins(kernel(candidate_ms=candidate_time), objective) is (direction < 0)


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_memory_improvement_remains_strict_for_model_and_kernel(delta):
    performance, memory, _ = model_acceptance(
        "memory",
        ModelProbeEvidence("ok", 0, 10.0, 100),
        ModelProbeEvidence("ok", 0, 8.0, 100 + delta),
        0,
    )
    assert performance
    assert memory is (delta < 0)
    assert candidate_wins(replace(kernel(), candidate_peak_bytes=100 + delta), "memory") is (
        delta < 0
    )
