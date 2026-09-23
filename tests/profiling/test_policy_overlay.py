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
        benchmark_kind="integrated_operator",
        memory_measured=True,
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
    assert decision(resolve, variant="large").implementation == "reference"
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


def test_generated_rules_match_operator_geometry_independently_of_model_variant():
    _, policy = resolver([measurement()])
    rule = policy.rules[0]
    assert rule.when["kernel_size"] == (1, 1, 1)
    assert rule.when["stride"] == (1, 1, 1)
    assert not {"model_family", "variant", "checkpointing"} & rule.when.keys()


@pytest.mark.parametrize("include_winners", [True, False])
@pytest.mark.parametrize("negative", [{"kernel_valid": False}, {"candidate_ms": 12.0}])
def test_suppressed_alternative_preserves_tombstone_for_rejected_inherited_recipe(
    include_winners, negative
):
    rejected = measurement(
        family="depthwise_conv3d",
        phase="backward_weight",
        implementation="triton_split_dw",
        batch=1,
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        parameters=(("dw_block", 512), ("dw_splits", 64)),
        **negative,
    )
    winner = replace(
        rejected,
        kernel_valid=True,
        candidate_ms=7.0,
        parameters=(("dw_block", 256), ("dw_splits", 64)),
    )
    op = OperatorDescriptor(
        "depthwise_conv3d", "regular", 32, 32, (3,) * 3, (1,) * 3, (1,) * 3, (1,) * 3, 32
    )
    ctx = ExecutionContext(
        "training",
        "cuda",
        (8, 9),
        48 * 1024**3,
        "bfloat16",
        1,
        (128,) * 3,
        "mednext_v1",
        "base",
        "none",
    )
    inherited = PolicyRegistry().resolver().resolve(op, ctx, "backward_weight")
    assert inherited.implementation == "triton_split_dw"
    assert inherited.parameters == {"dw_block": 512, "dw_splits": 64}

    policy = synthesize_profile(
        [rejected, winner],
        name="local",
        sm=(8, 9),
        objective="balanced",
        include_winners=include_winners,
    )
    resolve = PolicyRegistry(external=policy_to_primitive(policy)).resolver()
    selected = resolve.resolve(op, ctx, "backward_weight")
    assert selected.policy == "local"
    if include_winners:
        assert selected.implementation == "triton_split_dw"
        assert selected.parameters == {"dw_block": 256, "dw_splits": 64}
    else:
        assert selected.implementation == "reference"
    assert resolve.resolve(op, replace(ctx, batch_size=2), "backward_weight").policy != "local"
