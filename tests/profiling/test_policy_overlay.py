"""Local observations override exact contexts and leave other policy inference intact."""

from dataclasses import replace

import pytest

from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor
from mednext_accel.optimization.policies import PolicyRegistry
from mednext_accel.optimization.policy import policy_to_primitive
from mednext_accel.profiling.synthesize import Measurement, synthesize_profile


def measurement(**updates):
    base = Measurement(
        "pointwise_conv3d",
        "regular",
        "training",
        "pointwise_gemm_per_sample",
        10,
        (128, 128, 128),
        1,
        32,
        "bfloat16",
        "none",
        10.0,
        8.0,
        100,
        100,
        True,
        workload={"model_family": "mednext_v1", "variant": "base"},
    )
    return replace(base, **updates)


def resolver(items):
    policy = synthesize_profile(items, name="local", sm=(8, 9), objective="balanced")
    return PolicyRegistry(external=policy_to_primitive(policy)).resolver(), policy


def decision(resolve, *, batch=10, variant="base", channels=(1, 32)):
    op = OperatorDescriptor(
        "pointwise_conv3d", "regular", *channels, (1,) * 3, (1,) * 3, (0,) * 3, (1,) * 3, 1
    )
    ctx = ExecutionContext(
        "training",
        "cuda",
        (8, 9),
        48 * 1024**3,
        "bfloat16",
        batch,
        (128,) * 3,
        "mednext_v1",
        variant,
        "none",
    )
    return resolve.resolve(op, ctx, "training")


@pytest.mark.parametrize(
    "updates", [{"kernel_valid": False}, {"candidate_ms": 10.0}, {"candidate_peak_bytes": 116}]
)
def test_numerical_failures_and_valid_losers_block_inherited_winner_only_at_exact_match(updates):
    resolve, policy = resolver([measurement(**updates)])
    selected = decision(resolve)
    assert selected.implementation == "reference"
    assert selected.policy == "local"
    assert decision(resolve, batch=11).implementation == "pointwise_gemm_per_sample"
    assert decision(resolve, variant="large").implementation == "pointwise_gemm_per_sample"
    assert len(policy.rules) == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"probe_status": "oom", "kernel_valid": None},
        {"probe_status": "infrastructure", "kernel_valid": None},
        {"probe_status": "error", "kernel_valid": True},
        {"candidate_ms": 0.0},
        {"reference_peak_bytes": None},
    ],
)
def test_oom_infrastructure_and_incomplete_metrics_leave_policy_gaps(updates):
    resolve, policy = resolver([measurement(**updates)])
    assert policy.rules == ()
    assert decision(resolve).implementation == "pointwise_gemm_per_sample"
    assert decision(resolve).policy != "local"


def test_numerical_failure_wins_over_conflicting_candidate_without_changing_evidence():
    positive = measurement()
    negative = measurement(kernel_valid=False)
    before = [item.to_primitive() for item in (positive, negative)]
    resolve, policy = resolver([positive, negative])
    assert decision(resolve).implementation == "reference"
    assert len(policy.rules) == 1
    assert before == [item.to_primitive() for item in (positive, negative)]


def test_generated_rules_match_observed_kernel_geometry_and_model_variant():
    _, policy = resolver([measurement()])
    rule = policy.rules[0]
    assert rule.when["kernel_size"] == (1, 1, 1)
    assert rule.when["stride"] == (1, 1, 1)
    assert rule.when["model_family"] == "mednext_v1"
    assert rule.when["variant"] == "base"
