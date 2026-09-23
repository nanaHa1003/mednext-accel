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
        benchmark_kind="integrated_operator",
        memory_measured=True,
    )


def test_adjacent_measured_batches_form_a_bounded_work_region() -> None:
    profile = synthesize_profile(
        [measured(2, 4.0, 3.0), measured(3, 6.0, 4.5), measured(4, 8.0, 6.2)],
        name="test",
        sm=(12, 0),
        objective="balanced",
    )
    assert len(profile.rules) == 1
    bound = profile.rules[0].when["reduction_work"]
    assert (bound.minimum, bound.maximum) == (4194304, 8388608)
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
    assert profile.rules[1].when["reduction_work"].minimum == 6291456


def test_reported_winner_uses_the_same_balanced_threshold_as_synthesis() -> None:
    assert not candidate_wins(measured(2, 100.0, 99.0), "balanced")
    assert candidate_wins(measured(2, 100.0, 96.0), "balanced")


@pytest.mark.parametrize(
    "selection",
    [{}, {"implementation": "reference"}, {"parameters": (("tile", 32),)}],
)
def test_invalid_candidate_rejects_only_its_own_selection(selection) -> None:
    candidate = measured(2, 4.0, 1.0)
    rejected = replace(candidate, kernel_valid=False, **selection)

    profile = synthesize_profile(
        [candidate, rejected], name="test", sm=(12, 0), objective="balanced"
    )

    assert len(profile.rules) == 1
    expected = "pointwise_gemm_per_sample" if selection else "reference"
    assert profile.rules[0].use["training"].implementation == expected
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


def test_single_selected_batch_produces_only_a_bounded_work_rule() -> None:
    profile = synthesize_profile(
        [measured(13, 10.0, 8.0)], name="selected", sm=(12, 0), objective="balanced"
    )

    assert len(profile.rules) == 1
    work = profile.rules[0].when["reduction_work"]
    assert work.minimum == work.maximum == 27262976


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
        "benchmark_kind": "integrated_operator",
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


def test_integrated_measurement_without_optional_diagnostics_preserves_synthesis():
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


@pytest.mark.parametrize("objective", ["balanced", "throughput", "memory"])
def test_winner_ranking_uses_objective_and_is_independent_of_input_order(objective):
    from itertools import permutations

    fastest = replace(measured(2, 10.0, 7.0), parameters=(("tile", 32),), candidate_peak_bytes=90)
    smallest = replace(measured(2, 10.0, 9.0), parameters=(("tile", 64),), candidate_peak_bytes=60)
    loser = replace(measured(2, 10.0, 12.0), parameters=(("tile", 128),))
    incomplete = replace(measured(2, 10.0, 1.0), parameters=(("tile", 256),), probe_status="oom")
    expected = {"tile": 64 if objective == "memory" else 32}
    for items in permutations((fastest, smallest, loser, incomplete)):
        profile = synthesize_profile(items, name="ranking", sm=(12, 0), objective=objective)
        assert len(profile.rules) == 1
        assert profile.rules[0].use["training"].implementation == "pointwise_gemm_per_sample"
        assert profile.rules[0].use["training"].parameters == expected


@pytest.mark.parametrize("objective", ["balanced", "throughput", "memory"])
def test_exact_ties_use_stable_implementation_and_canonical_parameter_order(objective):
    from itertools import permutations

    first = replace(
        measured(2, 10.0, 7.0), candidate_peak_bytes=90, parameters=(("a", 1), ("z", 2))
    )
    parameter_alternative = replace(first, parameters=(("z", 1), ("a", 2)))
    implementation_alternative = replace(first, implementation="triton_depthwise_dx", parameters=())
    for items in permutations((first, parameter_alternative, implementation_alternative)):
        profile = synthesize_profile(items, name="ties", sm=(12, 0), objective=objective)
        selection = profile.rules[0].use["training"]
        assert selection.implementation == "pointwise_gemm_per_sample"
        assert selection.parameters == {"a": 1, "z": 2}


@pytest.mark.parametrize("objective", ["balanced", "throughput", "memory"])
def test_all_complete_losers_produce_reference(objective):
    items = [
        replace(measured(2, 10.0, 12.0), parameters=(("tile", 32),)),
        replace(measured(2, 10.0, 13.0), parameters=(("tile", 64),)),
    ]
    profile = synthesize_profile(items, name="losers", sm=(12, 0), objective=objective)
    assert len(profile.rules) == 1
    assert profile.rules[0].use["training"].implementation == "reference"


def test_incomplete_only_alternatives_leave_a_gap():
    items = [
        replace(measured(2, 10.0, 1.0), probe_status="oom", kernel_valid=None),
        replace(measured(2, 10.0, 1.0), probe_status="timeout", parameters=(("tile", 32),)),
        replace(measured(2, 10.0, 0.0), parameters=(("tile", 64),)),
    ]
    profile = synthesize_profile(items, name="gaps", sm=(12, 0), objective="balanced")
    assert profile.rules == ()


def test_aggregate_rejection_preserves_conclusive_negative_alternative():
    winner = measured(2, 10.0, 7.0)
    loser = replace(winner, candidate_ms=12.0, parameters=(("tile", 32),))
    profile = synthesize_profile(
        [winner, loser], name="aggregate", sm=(12, 0), objective="balanced", include_winners=False
    )
    assert len(profile.rules) == 1
    assert profile.rules[0].use["training"].implementation == "reference"


def test_aggregate_rejection_without_negative_evidence_leaves_a_gap():
    profile = synthesize_profile(
        [measured(2, 10.0, 7.0)],
        name="aggregate",
        sm=(12, 0),
        objective="balanced",
        include_winners=False,
    )
    assert profile.rules == ()


@pytest.mark.parametrize("objective", ["balanced", "throughput", "memory"])
@pytest.mark.parametrize("include_winners", [True, False])
@pytest.mark.parametrize(
    "updates",
    [
        {"candidate_ms": 1.0, "candidate_peak_bytes": 1},
        {"candidate_ms": 12.0},
        {"kernel_valid": False},
    ],
)
def test_raw_diagnostics_never_create_rules_or_reject_integrated_candidates(
    objective, include_winners, updates
):
    integrated = replace(measured(2, 10.0, 7.0), candidate_peak_bytes=90)
    raw = replace(integrated, benchmark_kind="raw_kernel_diagnostic", **updates)
    assert not candidate_wins(raw, objective)
    options = dict(
        name="diagnostics", sm=(12, 0), objective=objective, include_winners=include_winners
    )
    assert synthesize_profile([raw], **options).rules == ()
    mixed = synthesize_profile([raw, integrated], **options)
    expected = synthesize_profile([integrated], **options)
    assert mixed == expected


def test_legacy_winner_deserializes_as_diagnostic_and_cannot_synthesize_rules():
    from mednext_accel.profiling.campaign import campaign_to_primitive, load_campaign
    from mednext_accel.profiling.evidence import (
        CampaignEvidence,
        EnvironmentEvidence,
        ProfilingEvidence,
    )

    original = replace(measured(2, 10.0, 7.0), objective="balanced", objective_winner=True)
    evidence = ProfilingEvidence(
        EnvironmentEvidence({"gpu": {"sm": [12, 0]}}),
        CampaignEvidence(campaign_to_primitive(load_campaign(None))),
        kernel_measurements=(original,),
    )
    document = evidence.to_primitive()
    legacy = document["kernel_measurements"][0]
    for field in (
        "benchmark_kind",
        "compile_mode",
        "gradient_mask",
        "raw_diagnostic",
        "memory_measured",
    ):
        del legacy[field]
    restored = ProfilingEvidence.from_primitive(document).kernel_measurements[0]
    assert restored.benchmark_kind == "raw_kernel_diagnostic"
    assert restored.memory_measured is None
    assert restored.objective_winner is True  # Historical diagnostic, not a dispatch verdict.
    assert not candidate_wins(restored, "balanced")
    assert (
        synthesize_profile([restored], name="legacy", sm=(12, 0), objective="balanced").rules == ()
    )


@pytest.mark.parametrize("objective", ["balanced", "memory"])
@pytest.mark.parametrize("compile_mode", ["reduce-overhead", "max-autotune"])
@pytest.mark.parametrize("candidate_ms,candidate_peak", [(1.0, 1), (12.0, 1000)])
def test_cudagraph_memory_unavailable_is_neither_a_winner_nor_a_loser(
    objective, compile_mode, candidate_ms, candidate_peak
):
    item = replace(
        measured(2, 10.0, candidate_ms),
        compile_mode=compile_mode,
        candidate_peak_bytes=candidate_peak,
        memory_measured=False,
    )
    assert not candidate_wins(item, objective)
    assert (
        synthesize_profile([item], name="unavailable", sm=(12, 0), objective=objective).rules == ()
    )


@pytest.mark.parametrize(
    "candidate_ms,implementation", [(1.0, "pointwise_gemm_per_sample"), (12.0, "reference")]
)
def test_throughput_uses_latency_when_memory_is_unavailable(candidate_ms, implementation):
    item = replace(
        measured(2, 10.0, candidate_ms),
        compile_mode="reduce-overhead",
        memory_measured=False,
        reference_peak_bytes=None,
        candidate_peak_bytes=None,
    )
    profile = synthesize_profile([item], name="latency", sm=(12, 0), objective="throughput")
    assert len(profile.rules) == 1
    assert profile.rules[0].use["training"].implementation == implementation


def test_unrecorded_memory_retains_legacy_verdict_but_cannot_establish_memory_policy():
    original = replace(measured(2, 10.0, 7.0), objective="balanced", objective_winner=True)
    fields = original.to_primitive()
    fields.pop("memory_measured", None)
    restored = Measurement(**fields)
    assert restored.memory_measured is None
    assert restored.objective_winner is True
    assert not candidate_wins(restored, "balanced")
    assert (
        synthesize_profile([restored], name="legacy", sm=(12, 0), objective="balanced").rules == ()
    )


def test_duplicate_model_and_checkpoint_projections_collapse():
    original = measured(2, 4.0, 3.0)
    projections = [
        replace(
            original,
            checkpointing=checkpointing,
            workload={"model_family": "mednext_v1", "variant": variant},
        )
        for checkpointing in ("none", "whole-block")
        for variant in ("base", "large")
    ]
    profile = synthesize_profile(projections, name="projected", sm=(8, 9), objective="balanced")
    assert len(profile.rules) == 1
    assert not {"model_family", "variant", "checkpointing"} & profile.rules[0].when.keys()
    rejected = replace(projections[0], kernel_valid=False)
    blocked = synthesize_profile(
        [*projections, rejected], name="projected", sm=(8, 9), objective="balanced"
    )
    assert len(blocked.rules) == 1
    assert blocked.rules[0].use["training"].implementation == "reference"


@pytest.mark.parametrize(
    "phase,implementation,metric,unit",
    [
        ("backward_input", "triton_depthwise_dx", "work", 262144),
        ("backward_weight", "triton_split_dw", "reduction_work", 4096),
    ],
)
@pytest.mark.parametrize("barrier", ["negative", "incomplete"])
def test_depthwise_intervals_use_auto_recipes_and_stop_at_observed_barriers(
    phase, implementation, metric, unit, barrier
):
    items = [
        replace(
            measured(batch, 10.0, 7.0),
            family="depthwise_conv3d",
            phase=phase,
            implementation=implementation,
            kernel_size=3,
            spatial_shape=(16,) * 3,
            in_channels=64,
            out_channels=64,
        )
        for batch in range(1, 6)
    ]
    items[2] = (
        replace(items[2], candidate_ms=12.0)
        if barrier == "negative"
        else replace(items[2], kernel_valid=None, probe_status="oom")
    )
    profile = synthesize_profile(items, name="segmented", sm=(8, 9), objective="balanced")
    positives = [rule for rule in profile.rules if rule.use[phase].implementation != "reference"]
    assert len(positives) == 2
    assert [(rule.when[metric].minimum, rule.when[metric].maximum) for rule in positives] == [
        (unit, 2 * unit),
        (4 * unit, 5 * unit),
    ]
    assert all(rule.use[phase].parameters == "auto" for rule in positives)
    assert all(not {"batch", "channels", "spatial_shape"} & rule.when.keys() for rule in positives)
    if barrier == "negative":
        first = profile.rules[0]
        assert first.use[phase].implementation == "reference"
        assert first.when["batch"].minimum == first.when["batch"].maximum == 3
        assert first.when["channels"] == (64, 64)
        assert first.when["spatial_shape"] == (16, 16, 16)
    else:
        assert len(profile.rules) == 2


def pointwise_grid():
    return [
        replace(
            measured(batch, 10.0, 7.0), spatial_shape=(8,) * 3, in_channels=cin, out_channels=cout
        )
        for cin in (32, 64)
        for cout in (64, 128)
        for batch in (2, 4)
    ]


def test_complete_pointwise_grid_forms_one_bounded_channel_and_work_region():
    from mednext_accel.optimization.policy import policy_to_primitive

    profile = synthesize_profile(pointwise_grid(), name="grid", sm=(8, 9), objective="balanced")
    assert len(profile.rules) == 1
    when = policy_to_primitive(profile)["rules"][0]["when"]
    assert when["in_channels"] == {"min": 32, "max": 64}
    assert when["out_channels"] == {"min": 64, "max": 128}
    assert when["reduction_work"] == {"min": 1024, "max": 2048}
    assert "spatial_shape" not in when and "batch" not in when and "channels" not in when
    assert (
        synthesize_profile(
            list(reversed(pointwise_grid())), name="grid", sm=(8, 9), objective="balanced"
        )
        == profile
    )


@pytest.mark.parametrize("missing", ["absent", "oom", "negative"])
def test_pointwise_rectangles_do_not_fill_incomplete_or_losing_corners(missing):
    from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor
    from mednext_accel.optimization.policy_resolver import PolicyResolver

    items = pointwise_grid()
    corner = items.pop()
    if missing != "absent":
        items.append(
            replace(corner, probe_status="oom", kernel_valid=None)
            if missing == "oom"
            else replace(corner, candidate_ms=12.0)
        )
    profile = synthesize_profile(items, name="grid", sm=(8, 9), objective="balanced")
    op = OperatorDescriptor(
        "pointwise_conv3d", "regular", 64, 128, (1,) * 3, (1,) * 3, (0,) * 3, (1,) * 3, 1
    )
    ctx = ExecutionContext(
        "training",
        "cuda",
        (8, 9),
        48 * 2**30,
        "bfloat16",
        4,
        (8,) * 3,
        "mednext_v1",
        "base",
        "none",
    )
    selected = PolicyResolver(external=profile).resolve(op, ctx, "training")
    assert selected.implementation == "reference"
    assert selected.policy == ("grid" if missing == "negative" else "reference")
    if missing == "negative":
        assert profile.rules[0].use["training"].implementation == "reference"


def test_single_depthwise_observation_does_not_create_unbounded_positive_range():
    item = replace(
        measured(2, 10.0, 7.0),
        family="depthwise_conv3d",
        phase="backward_input",
        implementation="triton_depthwise_dx",
        kernel_size=3,
        in_channels=64,
        out_channels=64,
        spatial_shape=(16,) * 3,
    )
    profile = synthesize_profile([item], name="singleton", sm=(8, 9), objective="balanced")
    assert profile.rules[0].when["work"].minimum == profile.rules[0].when["work"].maximum == 524288


def test_conflicting_depthwise_work_projection_keeps_valid_context_exact():
    winner = replace(
        measured(2, 10.0, 7.0),
        family="depthwise_conv3d",
        phase="backward_input",
        implementation="triton_depthwise_dx",
        kernel_size=3,
        in_channels=64,
        out_channels=64,
        spatial_shape=(16,) * 3,
    )
    loser = replace(winner, batch=4, in_channels=32, out_channels=32, candidate_ms=12.0)
    profile = synthesize_profile([winner, loser], name="collision", sm=(8, 9), objective="balanced")
    assert len(profile.rules) == 2
    negative, positive = profile.rules
    assert negative.use["backward_input"].implementation == "reference"
    assert positive.use["backward_input"].implementation == "triton_depthwise_dx"
    assert positive.when["channels"] == (64, 64)
    assert positive.when["batch"].minimum == positive.when["batch"].maximum == 2


def test_pointwise_region_resolves_unseen_channels_and_batch_but_respects_bounds():
    from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor
    from mednext_accel.optimization.policy_resolver import PolicyResolver

    profile = synthesize_profile(pointwise_grid(), name="grid", sm=(8, 9), objective="balanced")
    resolver = PolicyResolver(external=profile)
    op = OperatorDescriptor(
        "pointwise_conv3d", "regular", 48, 96, (1,) * 3, (1,) * 3, (0,) * 3, (1,) * 3, 1
    )
    ctx = ExecutionContext(
        "training",
        "cuda",
        (8, 9),
        48 * 2**30,
        "bfloat16",
        3,
        (8,) * 3,
        "mednext_v1",
        "large",
        "whole-block",
    )
    assert resolver.resolve(op, ctx, "training").implementation == "pointwise_gemm_per_sample"
    for descriptor, context in [
        (replace(op, in_channels=65), ctx),
        (replace(op, out_channels=129), ctx),
        (op, replace(ctx, batch_size=5)),
        (op, replace(ctx, batch_size=1)),
    ]:
        assert resolver.resolve(descriptor, context, "training").implementation == "reference"
