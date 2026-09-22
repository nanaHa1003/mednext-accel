"""Contract for the sanitized, legacy L40S evidence bundled with the package."""

import hashlib
import json
import warnings
from dataclasses import replace

import pytest

from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor
from mednext_accel.optimization.profiles import ProfileRegistry, load_bundled_profile
from mednext_accel.optimization.schema import profile_to_primitive


def context(batch=1, spatial=128, checkpointing="all-expansion", sm=(8, 9)):
    return ExecutionContext(
        "training",
        "cuda",
        sm,
        47667740672,
        "bfloat16",
        batch,
        (spatial,) * 3,
        "mednext_v1",
        "base",
        checkpointing,
    )


def descriptor(family="pointwise_conv3d", direction="regular", channels=1, output=32):
    pointwise = family == "pointwise_conv3d"
    return OperatorDescriptor(
        family,
        direction,
        channels,
        output,
        (1 if pointwise else 3,) * 3,
        (2 if direction in ("downsample", "transpose") else 1,) * 3,
        (0 if pointwise else 1,) * 3,
        (1,) * 3,
        1 if pointwise else channels,
    )


def test_sm89_retains_source_rules_and_sanitized_provenance():
    profile = load_bundled_profile("sm89")
    primitive = profile_to_primitive(profile)
    assert profile.target_sm == (8, 9)
    assert len(profile.rules) == 78
    assert (
        hashlib.sha256(
            json.dumps(primitive["rules"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == "7085bbbed55c32cafbfef3a087925e231687f99b887cdae85162c3e973855793"
    )
    assert primitive["measurements"] == []
    assert primitive["overrides"] == []
    assert all(
        selection.implementation == "reference"
        for phases in profile.defaults.values()
        for selection in phases.values()
    )
    provenance = primitive["profile"]["provenance"]
    assert provenance["source_sha256"] == (
        "c17c79e3a752ab9871cec8ccd2ec4a71767ca3381f28d26cea5e85c005d81279"
    )
    assert provenance["coverage"]["batches"] == [1, 2, 3, 4, 5, 6, 7, 8, 10]
    assert provenance["coverage"]["batch_search"] == "legacy-dense"
    assert provenance["coverage"]["workload"]["variant"] == "base"
    assert provenance["coverage"]["workload"]["identity_source"] == "user-confirmed"
    assert provenance["environment"]["gpu"]["name"] == "NVIDIA L40S"
    assert provenance["source_measurement_count"] == 432
    assert provenance["rule_count"] == 78
    assert provenance["execution"]["whole_model_validation_count"] == 9
    assert "rejected measurements were not relabeled" in provenance["limitations"][0]
    serialized = json.dumps(primitive)
    for sensitive in ("/home/", "hostname", "username", "password", "output_dir"):
        assert sensitive not in serialized


def test_sm89_registry_selects_exact_profile_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolver = ProfileRegistry().resolver_for(sm=(8, 9))
    assert not caught
    assert [p.name for p in resolver.profiles] == ["sm89", "generic-nvidia"]
    assert resolver.warning is None


@pytest.mark.parametrize(
    ("op", "ctx", "phase", "implementation", "parameters", "rule"),
    [
        (descriptor(), context(), "training", "pointwise_gemm_per_sample", {}, "generated-0077"),
        (
            descriptor("depthwise_conv3d", "downsample", 256, 256),
            context(3, 16),
            "backward_input",
            "triton_downsample_dx",
            {"dx_block": 128},
            "generated-0000",
        ),
        (
            descriptor("depthwise_conv3d", "regular", 512, 512),
            context(6, 8),
            "backward_input",
            "triton_depthwise_dx",
            {"dx_block": 128},
            "generated-0008",
        ),
        (
            descriptor("depthwise_conv_transpose3d", "transpose", 512, 512),
            context(8, 8),
            "backward_weight",
            "triton_transpose_split_dw",
            {"dw_block": 512, "dw_splits": 1},
            "generated-0048",
        ),
    ],
)
def test_sm89_measured_decisions(op, ctx, phase, implementation, parameters, rule):
    decision = ProfileRegistry().resolver_for(sm=(8, 9)).resolve(op, ctx, phase)
    assert decision.profile == "sm89"
    assert decision.implementation == implementation
    assert decision.parameters == parameters
    assert decision.rule == rule


@pytest.mark.parametrize(
    ("op", "ctx", "phase"),
    [
        (descriptor(), context(2), "training"),  # Rejected legacy pointwise shape.
        (descriptor(channels=32, output=64), context(), "training"),
        (descriptor(channels=7, output=13), context(), "training"),
        (descriptor(), context(9), "training"),
        (descriptor("depthwise_conv3d", "downsample", 256, 256), context(9, 16), "backward_input"),
        (descriptor(), context(checkpointing="none"), "training"),
        (descriptor(), context(checkpointing="whole-block"), "training"),
        (
            descriptor("depthwise_conv3d", "downsample", 256, 256),
            context(3, 16, "none"),
            "backward_input",
        ),
        (
            descriptor("depthwise_conv3d", "downsample", 256, 256),
            context(3, 16, "whole-block"),
            "backward_input",
        ),
    ],
)
def test_sm89_uncovered_contexts_use_reference(op, ctx, phase):
    decision = ProfileRegistry().resolver_for(sm=(8, 9)).resolve(op, ctx, phase)
    assert decision.profile == "sm89"
    assert decision.implementation == "reference"
    assert decision.confidence == "default"


def test_explicit_sm89_on_another_sm_warns_and_applies():
    profile = profile_to_primitive(load_bundled_profile("sm89"))
    registry = ProfileRegistry(external=profile)
    with pytest.warns(UserWarning, match="applying it as requested"):
        resolver = registry.resolver_for(sm=(12, 0))
    decision = resolver.resolve(descriptor(), replace(context(), sm=(12, 0)), "training")
    assert decision.profile == "sm89"
    assert decision.implementation == "pointwise_gemm_per_sample"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        registry.resolver_for(sm=(12, 0))
    assert not caught
