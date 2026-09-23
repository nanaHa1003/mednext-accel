import warnings
from dataclasses import replace

import pytest

from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor
from mednext_accel.optimization.policy import parse_policy
from mednext_accel.optimization.policy_resolver import PolicyResolver


def descriptor(**changes):
    values = dict(
        family="depthwise_conv3d",
        direction="regular",
        in_channels=128,
        out_channels=128,
        groups=128,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        dilation=(1, 1, 1),
        role="encoder.block",
    )
    values.update(changes)
    return OperatorDescriptor(**values)


def context(**changes):
    values = dict(
        phase="training",
        device_type="cuda",
        sm=(8, 9),
        total_vram_bytes=48 * 2**30,
        dtype="bfloat16",
        batch_size=9,
        spatial_shape=(32, 32, 32),
        model_family="mednext_v1",
        variant="base",
        checkpointing="all-expansion",
    )
    values.update(changes)
    return ExecutionContext(**values)


def rule(
    identifier="dw",
    *,
    implementation="triton_split_dw",
    phase="backward_weight",
    when=None,
    parameters="auto",
    confidence="validated-cross-sm",
):
    selection = {"implementation": implementation}
    if parameters is not None:
        selection["parameters"] = parameters
    return {
        "id": identifier,
        "when": {"family": "depthwise_conv3d", **(when or {})},
        "use": {phase: selection},
        "confidence": confidence,
    }


def policy(name, rules=(), *, sm=None, vendor="nvidia"):
    target = {"vendor": vendor}
    if sm is not None:
        target["sm"] = list(sm)
    return parse_policy(
        {
            "version": 2,
            "kind": "mednext-accel-policy",
            "name": name,
            "target": target,
            "rules": list(rules),
        }
    )


def resolver(bundled=(), **kwargs):
    kwargs.setdefault("triton_available", True)
    return PolicyResolver(bundled, **kwargs)


def test_layers_choose_external_then_exact_sm_then_shared_then_reference():
    shared = policy("shared", [rule()])
    exact = policy("exact", [rule(when={"batch": {"max": 8}})], sm=(8, 9))
    external = policy("external", [rule(when={"batch": 4})])
    subject = resolver([shared, exact], external=external)
    assert (
        subject.resolve(descriptor(), context(batch_size=4), "backward_weight").policy == "external"
    )
    assert subject.resolve(descriptor(), context(batch_size=3), "backward_weight").policy == "exact"
    selected = subject.resolve(descriptor(), context(), "backward_weight")
    assert selected.policy == "shared"
    assert selected.rule == "dw"
    assert selected.confidence == "validated-cross-sm"
    assert selected.parameters == {"dw_splits": 9, "dw_block": 512}
    assert selected.disposition == "custom"
    fallback = subject.resolve(descriptor(), context(), "forward")
    assert fallback.implementation == fallback.policy == "reference"
    assert fallback.disposition == "native"


def test_tombstone_stops_lower_layers_and_missing_phase_falls_through():
    shared = policy("shared", [rule()])
    blocked = policy("blocked", [rule(implementation="reference", parameters=None)], sm=(8, 9))
    selected = resolver([shared, blocked]).resolve(descriptor(), context(), "backward_weight")
    assert (selected.implementation, selected.policy, selected.rule) == (
        "reference",
        "blocked",
        "dw",
    )
    dx_only = policy(
        "dx-only", [rule(implementation="triton_depthwise_dx", phase="backward_input")], sm=(8, 9)
    )
    assert (
        resolver([shared, dx_only]).resolve(descriptor(), context(), "backward_weight").policy
        == "shared"
    )


def test_first_matching_rule_with_requested_phase_wins():
    rules = [
        rule("other-phase", implementation="triton_depthwise_dx", phase="backward_input"),
        rule("first", parameters={"dw_splits": 2}),
        rule("second", parameters={"dw_splits": 3}),
    ]
    result = resolver([policy("ordered", rules)]).resolve(
        descriptor(), context(), "backward_weight"
    )
    assert result.rule == "first"
    assert result.parameters == {"dw_splits": 2, "dw_block": 512}


@pytest.mark.parametrize("sm", [(8, 9), None])
def test_guarded_external_choice_falls_through_to_valid_lower_layer(sm):
    external = policy("external", [rule("guarded", implementation="triton_transpose_split_dw")])
    lower = policy("lower", [rule("valid")], sm=sm)
    selected = resolver([lower], external=external).resolve(
        descriptor(), context(), "backward_weight"
    )
    assert (selected.policy, selected.rule, selected.implementation) == (
        "lower",
        "valid",
        "triton_split_dw",
    )
    assert selected.guard_reason is None


def test_reference_tombstone_after_guard_stops_resolution():
    external = policy("external", [rule(implementation="triton_transpose_split_dw")])
    exact = policy("exact", [rule("stop", implementation="reference", parameters=None)], sm=(8, 9))
    shared = policy("shared", [rule()])
    selected = resolver([shared, exact], external=external).resolve(
        descriptor(), context(), "backward_weight"
    )
    assert (selected.policy, selected.rule, selected.implementation) == (
        "exact",
        "stop",
        "reference",
    )
    assert selected.guard_reason is None


def test_guard_fallthrough_never_searches_later_rules_in_same_policy():
    external = policy(
        "external",
        [
            rule("guarded", implementation="triton_transpose_split_dw"),
            rule("must-not-select"),
        ],
    )
    selected = resolver([policy("shared", [rule("valid")])], external=external).resolve(
        descriptor(), context(), "backward_weight"
    )
    assert (selected.policy, selected.rule) == ("shared", "valid")


def test_all_guarded_layers_preserve_highest_priority_diagnostic():
    external = policy("external", [rule("first-guard", implementation="triton_transpose_split_dw")])
    lower = policy("lower", [rule()], sm=(8, 9))
    selected = resolver([lower], external=external, triton_available=False).resolve(
        descriptor(), context(), "backward_weight"
    )
    assert (selected.policy, selected.rule, selected.implementation, selected.confidence) == (
        "external",
        "first-guard",
        "reference",
        "guard",
    )
    assert selected.guard_reason == "implementation does not support this operator family"


def test_execution_sm_is_not_frozen_at_construction_and_unknown_sm_gets_shared():
    shared = policy("shared", [rule()])
    overlays = [policy("sm89", [rule()], sm=(8, 9)), policy("sm120", [rule()], sm=(12, 0))]
    subject = resolver([shared, *overlays])
    assert subject.resolve(descriptor(), context(), "backward_weight").policy == "sm89"
    large = subject.resolve(
        descriptor(),
        context(sm=(12, 0), batch_size=24, total_vram_bytes=96 * 2**30),
        "backward_weight",
    )
    assert large.policy == "sm120"
    assert large.parameters == {"dw_splits": 48, "dw_block": 1024}
    assert subject.resolve(descriptor(), context(sm=(9, 0)), "backward_weight").policy == "shared"


def test_external_wrong_sm_applies_with_one_warning_and_is_reported_each_time():
    external = policy("external", [rule()], sm=(12, 0))
    subject = resolver(external=external)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = subject.resolve(descriptor(), context(), "backward_weight")
        second = subject.resolve(descriptor(), context(), "backward_weight")
    assert first.policy == "external"
    assert first.parameters == {"dw_splits": 9, "dw_block": 512}
    assert first.warning == second.warning
    assert "applying it as requested" in first.warning
    assert len(caught) == 1


def test_vendor_filter_does_not_apply_non_nvidia_policy():
    subject = resolver([policy("other", [rule()], vendor="amd")])
    assert subject.resolve(descriptor(), context(), "backward_weight").implementation == "reference"


@pytest.mark.parametrize(
    "conditions,matching",
    [
        ({"batch": 9}, True),
        ({"batch": {"max": 8}}, False),
        ({"spatial_volume": 32768}, True),
        ({"work": 37748736}, True),
        ({"work": {"min": 37748737}}, False),
        ({"reduction_work": 294912}, True),
        ({"total_vram_gib": {"min": 48, "max": 48}}, True),
        ({"total_vram_gib": {"min": 49}}, False),
        ({"direction": "regular", "role": "encoder.block"}, True),
        ({"kernel_size": [3, 3, 3], "stride": [1, 1, 1]}, True),
        ({"spatial_shape": [32, 32, 32], "channels": [128, 128]}, True),
        ({"channels": [64, 128]}, False),
        ({"dtype": "float32"}, False),
        ({"model_family": "mednext_v1", "variant": "base", "checkpointing": "all-expansion"}, True),
    ],
)
def test_match_conditions_use_descriptor_and_execution_facts(conditions, matching):
    result = resolver([policy("test", [rule(when=conditions)])]).resolve(
        descriptor(), context(), "backward_weight"
    )
    assert (result.implementation == "triton_split_dw") == matching


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"device_type": "cpu", "sm": None}, "NVIDIA"),
        ({"sm": None}, "NVIDIA"),
        ({"dtype": "int64"}, "dtype"),
        ({"phase": "inference"}, "training"),
        ({"export": True}, "export"),
        ({"spatial_shape": (16, 32, 64)}, "cubic"),
    ],
)
def test_static_context_guards_are_visible_in_decisions(changes, reason):
    result = resolver([policy("test", [rule()])]).resolve(
        descriptor(), context(**changes), "backward_weight"
    )
    assert result.implementation == "reference"
    assert result.confidence == "guard"
    assert result.disposition == "native"
    assert reason in result.guard_reason


@pytest.mark.parametrize(
    "changes",
    [
        {"kernel_size": (3, 5, 3)},
        {"kernel_size": (2, 2, 2)},
        {"stride": (2, 2, 2)},
        {"padding": (0, 0, 0)},
        {"dilation": (2, 2, 2)},
        {"groups": 1},
        {"direction": "downsample"},
    ],
)
def test_geometry_guards_prevent_broad_rules_from_selecting_invalid_kernels(changes):
    result = resolver([policy("test", [rule()])]).resolve(
        descriptor(**changes), context(), "backward_weight"
    )
    assert result.implementation == "reference"
    assert "geometry" in result.guard_reason


def test_absent_triton_guards_depthwise_but_does_not_disable_pointwise():
    subject = resolver([policy("test", [rule()])], triton_available=False)
    result = subject.resolve(descriptor(), context(), "backward_weight")
    assert result.implementation == "reference"
    assert "Triton" in result.guard_reason
    pointwise = rule(
        implementation="pointwise_gemm_per_sample",
        phase="training",
        parameters=None,
        when={"family": "pointwise_conv3d"},
    )
    operator = descriptor(
        family="pointwise_conv3d", kernel_size=(1, 1, 1), padding=(0, 0, 0), groups=1
    )
    result = resolver([policy("pointwise", [pointwise])], triton_available=False).resolve(
        operator, context(), "training"
    )
    assert result.implementation == "pointwise_gemm_per_sample"
    assert "grad_enabled" in result.execution_guards


def test_metadata_guards_enforce_family_phase_and_approximation():
    wrong_family = rule(
        implementation="pointwise_gemm_per_sample", phase="training", parameters=None
    )
    result = resolver([policy("wrong-family", [wrong_family])]).resolve(
        descriptor(), context(), "training"
    )
    assert result.implementation == "reference"
    assert "family" in result.guard_reason
    wrong_phase = rule(phase="backward_input")
    result = resolver([policy("wrong-phase", [wrong_phase])]).resolve(
        descriptor(), context(), "backward_input"
    )
    assert result.implementation == "reference"
    assert "phase" in result.guard_reason
    gelu = rule(
        implementation="gelu_tanh", phase="inference", parameters=None, when={"family": "gelu"}
    )
    subject = resolver([policy("gelu", [gelu])])
    operator = descriptor(family="gelu")
    result = subject.resolve(operator, context(phase="inference"), "inference")
    assert result.implementation == "reference"
    assert "approximate" in result.guard_reason
    allowed = subject.resolve(
        operator, context(phase="inference", allow_approximate=True), "inference"
    )
    assert allowed.implementation == "gelu_tanh"


def test_decisions_are_immutable_and_report_tensor_time_checks():
    subject = resolver([policy("test", [rule()])])
    decisions = subject.resolve_all(
        [descriptor(), replace(descriptor(), role="other")], context(), "backward_weight"
    )
    assert len(decisions) == 2
    assert {"grad_enabled", "rank_5", "contiguous"} <= set(decisions[0].execution_guards)
    with pytest.raises(TypeError):
        decisions[0].parameters["dw_splits"] = 100


def test_compiled_resolution_retains_warning_text_without_consuming_eager_notice(monkeypatch):
    import torch

    external = policy("external", [rule()], sm=(12, 0))
    subject = resolver(external=external)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with monkeypatch.context() as tracing:
            tracing.setattr(torch.compiler, "is_compiling", lambda: True)
            compiled = subject.resolve(descriptor(), context(), "backward_weight")
        assert len(caught) == 0
        assert compiled.implementation == "triton_split_dw"
        eager = subject.resolve(descriptor(), context(), "backward_weight")
        repeated = subject.resolve(descriptor(), context(), "backward_weight")
    assert len(caught) == 1
    assert compiled.warning == eager.warning == repeated.warning == str(caught[0].message)


def test_warning_tracks_device_changes_without_freezing_execution_sm():
    subject = resolver(external=policy("external", [rule()], sm=(8, 9)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        same = subject.resolve(descriptor(), context(), "backward_weight")
        first = subject.resolve(descriptor(), context(sm=(12, 0)), "backward_weight")
        other = subject.resolve(descriptor(), context(sm=(8, 6)), "backward_weight")
        back = subject.resolve(descriptor(), context(sm=(12, 0)), "backward_weight")
    assert same.warning is None
    assert len(caught) == 2
    assert first.warning == back.warning != other.warning
    assert first.parameters != other.parameters
