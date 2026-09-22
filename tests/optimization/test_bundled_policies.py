"""Evidence-backed policy boundaries and dynamic integration contracts."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor
from mednext_accel.optimization.policies import PolicyRegistry


def registry(**kwargs):
    return PolicyRegistry(**kwargs).resolver(triton_available=True)


def context(batch=9, size=32, sm=(8, 9), **changes):
    result = ExecutionContext(
        "training",
        "cuda",
        sm,
        96 * 2**30,
        "bfloat16",
        batch,
        (size,) * 3,
        "mednext_v1",
        "base",
        "all-expansion",
    )
    return replace(result, **changes)


def op(channels=128, output=None, direction="regular", pointwise=False):
    output = channels if output is None else output
    return OperatorDescriptor(
        "pointwise_conv3d"
        if pointwise
        else "depthwise_conv_transpose3d"
        if direction == "transpose"
        else "depthwise_conv3d",
        direction,
        channels,
        output,
        (1 if pointwise else 3,) * 3,
        (1 if direction == "regular" else 2,) * 3,
        (0 if pointwise else 1,) * 3,
        (1,) * 3,
        1 if pointwise else channels,
    )


@pytest.mark.parametrize("sm", [(8, 6), (8, 9), (12, 0), (9, 0)])
@pytest.mark.parametrize("batch", [9, 13, 64])
@pytest.mark.parametrize("checkpointing", ["none", "whole-block", "all-expansion", "expansion:0,1"])
def test_inferred_regions_are_not_an_exact_batch_allowlist(sm, batch, checkpointing):
    resolver = registry()
    cases = [
        (op(256, direction="downsample"), 16, "backward_input", "triton_downsample_dx"),
        (op(128), 32, "backward_input", "triton_depthwise_dx"),
        (op(128), 32, "backward_weight", "triton_split_dw"),
        (op(512, direction="transpose"), 8, "backward_weight", "triton_transpose_split_dw"),
        (op(1, 32, pointwise=True), 128, "training", "pointwise_gemm_per_sample"),
    ]
    for descriptor, size, phase, expected in cases:
        assert (
            resolver.resolve(
                descriptor, context(batch, size, sm, checkpointing=checkpointing), phase
            ).implementation
            == expected
        )


@pytest.mark.parametrize(
    ("descriptor", "size", "phase", "below", "above"),
    [
        (op(256, direction="downsample"), 16, "backward_input", None, 1),
        (op(256), 16, "backward_input", 1, 2),
        (op(512), 8, "backward_input", 5, 6),
        (op(512), 8, "backward_weight", 1, 2),
        (op(512, direction="transpose"), 8, "backward_weight", 7, 8),
    ],
)
def test_shared_threshold_boundaries(descriptor, size, phase, below, above):
    resolver = registry()
    if below:
        assert (
            resolver.resolve(descriptor, context(below, size, (9, 0)), phase).implementation
            == "reference"
        )
    assert (
        resolver.resolve(descriptor, context(above, size, (9, 0)), phase).implementation
        != "reference"
    )


@pytest.mark.parametrize(
    ("sm", "size", "channels", "batch", "expected", "source"),
    [
        ((8, 6), 128, 32, 12, "reference", "sm86"),
        ((8, 6), 16, 256, 11, "reference", "reference"),
        ((8, 6), 16, 256, 12, "triton_split_dw", "sm86"),
        ((8, 9), 128, 32, 9, "triton_split_dw", "sm89"),
        ((8, 9), 128, 32, 11, "reference", "reference"),
        ((8, 9), 16, 256, 8, "triton_split_dw", "sm89"),
        ((8, 9), 16, 256, 10, "reference", "sm89"),
        ((9, 0), 128, 32, 64, "reference", "reference"),
        ((9, 0), 16, 256, 64, "reference", "reference"),
        ((12, 0), 128, 32, 64, "triton_split_dw", "sm120"),
    ],
)
@pytest.mark.parametrize("checkpointing", ["none", "whole-block", "all-expansion", "expansion:0,1"])
def test_regular_dw_counterexamples(sm, size, channels, batch, expected, source, checkpointing):
    decision = registry().resolve(
        op(channels), context(batch, size, sm, checkpointing=checkpointing), "backward_weight"
    )
    assert (decision.implementation, decision.policy) == (expected, source)


def test_sm120_pointwise_is_bounded_and_does_not_spill_into_shared():
    resolver = registry()
    descriptor = op(32, 64, pointwise=True)
    for sm in [(8, 6), (8, 9), (9, 0), (12, 0)]:
        for batch in [3, 5, 64]:
            decision = resolver.resolve(descriptor, context(batch, 128, sm), "training")
            assert (decision.implementation != "reference") == (sm == (12, 0) and batch == 3)


@pytest.mark.parametrize(("channels", "output", "size"), [(64, 32, 128), (128, 384, 63)])
def test_sm89_new_pointwise_winners_are_exact_context(channels, output, size):
    resolver = registry()
    decision = resolver.resolve(op(channels, output, pointwise=True), context(10, size), "training")
    assert decision.implementation == "pointwise_gemm_per_sample"
    assert decision.confidence == "measured-exact-context"
    assert (
        resolver.resolve(
            op(channels, output, pointwise=True), context(11, size), "training"
        ).implementation
        == "reference"
    )


def test_runtime_sm_changes_after_factory_construction(monkeypatch):
    import torch

    from mednext_accel import mednext_small

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = mednext_small(in_channels=1, out_channels=3)
    resolver = model.__dict__["_optimization_resolver"]
    for sm, expected in [((12, 0), "sm120"), ((8, 6), "sm86"), ((8, 9), "sm89")]:
        assert resolver.resolve(op(32), context(10, 128, sm), "backward_weight").policy == expected


def test_external_yaml_falls_through_and_reports_policy_fields(tmp_path, monkeypatch):
    import torch

    from mednext_accel import CheckpointConfig, mednext_small

    policy = {
        "version": 2,
        "kind": "mednext-accel-policy",
        "name": "local",
        "target": {"vendor": "nvidia"},
        "rules": [
            {
                "id": "blocked",
                "when": {"role": "stem"},
                "use": {"training": {"implementation": "reference"}},
                "confidence": "measured-exact-context",
            }
        ],
    }
    path = tmp_path / "local.policy.yaml"
    path.write_text(yaml.safe_dump(policy))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = mednext_small(
        in_channels=1, out_channels=3, checkpointing=CheckpointConfig(), optimization=path
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a: type("GPU", (), {"total_memory": 96 * 2**30})(),
    )
    report = model.explain_optimization(
        input_shape=(64, 1, 128, 128, 128), dtype="bfloat16", device="cuda"
    )
    assert report.policies == ("local", "sm120", "shared-nvidia")
    stem = next(d for d in report.decisions if d.descriptor.role == "stem")
    assert stem.policy == "local" and stem.rule == "blocked" and stem.implementation == "reference"
    assert any(
        d.implementation == "triton_split_dw" and d.execution_guards for d in report.decisions
    )


@pytest.mark.parametrize(
    "document", [{"schema_version": 1}, {"version": 1, "kind": "mednext-accel-evidence"}]
)
def test_factory_rejects_legacy_or_evidence_documents(document):
    from mednext_accel import mednext_small

    with pytest.raises(ValueError, match="policy v2|runtime policy"):
        mednext_small(in_channels=1, out_channels=3, optimization=document)


def test_bundled_yaml_has_no_defaults_and_is_compact():
    resolver = registry()
    assert {p.name for p in resolver.bundled} == {"shared-nvidia", "sm86", "sm89", "sm120"}
    directory = Path(__file__).parents[2] / "src/mednext_accel/policies"
    paths = list(directory.glob("*.yaml"))
    assert len(paths) == 4
    assert sum(len(p.read_text().splitlines()) for p in paths) < 650
    assert sum(len(p.read_bytes()) for p in paths) < 30000
    for path in paths:
        document = yaml.safe_load(path.read_text())
        assert "defaults" not in document
        assert document["evidence"]


def test_retained_sm89_dispatch_and_launch_fixture():
    import csv
    import json

    fixture = Path(__file__).parent / "fixtures/legacy_sm89_rules.csv"
    resolver = registry()
    with fixture.open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 78
    for row in rows:
        descriptor = op(
            int(row["in_channels"]),
            int(row["out_channels"]),
            row["direction"],
            row["family"] == "pointwise_conv3d",
        )
        for batch in range(int(row["batch_min"]), int(row["batch_max"]) + 1):
            decision = resolver.resolve(
                descriptor, context(batch, int(row["spatial"])), row["phase"]
            )
            assert decision.implementation == row["implementation"], row
            assert decision.parameters == json.loads(row["parameters"]), row


def test_bundled_sm120_launches_and_only_real_exceptions():
    import csv

    fixture = Path(__file__).parent / "fixtures/legacy_dw_launches.csv"
    with fixture.open() as source:
        rows = [r for r in csv.DictReader(source) if r["profile"] == "sm120"]
    resolver = registry()
    for row in rows:
        batch, size, channels = (int(row[k]) for k in ("batch", "size", "channels"))
        transpose = row["implementation"] == "triton_transpose_split_dw"
        descriptor = op(channels, direction="transpose" if transpose else "regular")
        decision = resolver.resolve(descriptor, context(batch, size, (12, 0)), "backward_weight")
        expected = {"dw_block": int(row["dw_block"]), "dw_splits": int(row["dw_splits"])}
        if not transpose and size == 32 and batch in (4, 6):
            expected["dw_splits"] = 4 if batch == 4 else 8
        assert decision.implementation == row["implementation"]
        assert decision.parameters == expected, row


@pytest.mark.parametrize("checkpointing", ["none", "whole-block"])
def test_sm120_stem_preserves_supported_checkpoint_contexts(checkpointing):
    selected = registry().resolve(
        op(1, 32, pointwise=True),
        context(3, 128, (12, 0), checkpointing=checkpointing),
        "training",
    )
    assert selected.implementation == "pointwise_gemm_per_sample"


@pytest.mark.parametrize(
    "style, stages", [(None, None), ("block", None), ("expansion", None), ("expansion", (0, 1))]
)
def test_factory_checkpoint_choice_preserves_operator_optimization(style, stages):
    from mednext_accel import CheckpointConfig, mednext_small
    from mednext_accel.ops.adaptive import AdaptiveDepthwise3d

    checkpointing = None if style is None else CheckpointConfig(style=style, stages=stages)
    model = mednext_small(in_channels=1, out_channels=3, checkpointing=checkpointing)
    module = next(m for m in model.modules() if isinstance(m, AdaptiveDepthwise3d))
    for sm in [(8, 6), (8, 9), (9, 0), (12, 0)]:
        execution = context(9, 32, sm, checkpointing=module.model_context.checkpointing)
        assert (
            module.resolver.resolve(op(128), execution, "backward_input").implementation
            == "triton_depthwise_dx"
        )
        assert (
            module.resolver.resolve(op(128), execution, "backward_weight").implementation
            == "triton_split_dw"
        )
