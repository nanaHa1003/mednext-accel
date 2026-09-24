from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

import mednext_accel.models as models
from mednext_accel import CheckpointConfig
from mednext_accel.models.blocks_v2 import (
    MedNeXtV2Block,
    MedNeXtV2DownBlock,
    MedNeXtV2UpBlock,
)
from mednext_accel.models.config_v2 import get_mednext_v2_config


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config(*, deep_supervision=False):
    return replace(
        get_mednext_v2_config("base", in_channels=1, out_channels=3),
        base_channels=2,
        block_counts=(1,) * 9,
        expansion_ratios=(2,) * 9,
        downsample_expansion_ratios=(2,) * 4,
        upsample_expansion_ratios=(2,) * 4,
        deep_supervision=deep_supervision,
    )


def test_v2_config_rejects_nonpaper_spatial_kernel():
    with pytest.raises(ValueError, match="kernel_size.*3"):
        replace(small_config(), kernel_size=5)


@pytest.mark.parametrize("variant,count", [("base", 61_969_507), ("wide", 246_536_387)])
def test_published_parameter_counts(variant, count):
    with torch.device("meta"):
        model = getattr(models, f"mednext_v2_{variant}")(
            in_channels=1, out_channels=3, optimization="reference"
        )
    assert sum(parameter.numel() for parameter in model.parameters()) == count


def test_stage_depths_and_ratios():
    with torch.device("meta"):
        model = models.mednext_v2_base(in_channels=1, out_channels=3)
    assert [len(stage) for stage in model.encoder_stages] == [3, 4, 8, 8]
    assert len(model.bottleneck) == 8
    assert [len(stage) for stage in model.decoder_stages] == [8, 8, 4, 3]
    stages = [*model.encoder_stages, model.bottleneck, *model.decoder_stages]
    for stage, ratio in zip(stages, [3, 4, 8, 8, 8, 8, 8, 4, 3], strict=True):
        for block in stage:
            assert type(block) is MedNeXtV2Block
            assert block.expand.out_channels == block.expand.in_channels * ratio
            assert not block.activation.approximate_eval
    for blocks, block_type, ratios in (
        (model.downsamples, MedNeXtV2DownBlock, [4, 8, 8, 8]),
        (model.upsamples, MedNeXtV2UpBlock, [8, 8, 4, 3]),
    ):
        for block, ratio in zip(blocks, ratios, strict=True):
            assert type(block) is block_type
            assert block.expand.out_channels == block.expand.in_channels * ratio


@pytest.mark.parametrize("shape", [(17, 19, 21), (16, 17, 32), (1, 1, 17)])
def test_spatial_shapes_round_trip(shape):
    model = models.MedNeXtV2(small_config()).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, *shape))
    assert output.shape == (1, 3, *shape)


@pytest.mark.parametrize("training", [True, False])
def test_batch_one_rejects_single_value_group_norm(training):
    model = models.MedNeXtV2(small_config()).train(training)
    with pytest.raises(ValueError, match="GroupNorm.*batch"):
        model(torch.randn(1, 1, 16, 16, 16))


def test_batch_two_accepts_single_voxel_bottleneck():
    model = models.MedNeXtV2(small_config())
    output = model(torch.randn(2, 1, 16, 16, 16))
    assert output.shape == (2, 3, 16, 16, 16)


def test_reconcile_spatial_right_crops_and_pads():
    from mednext_accel.models.mednext_v2 import _reconcile_spatial

    sample = torch.arange(24.0).reshape(1, 1, 2, 3, 4).requires_grad_()
    output = _reconcile_spatial(sample, (3, 2, 5))
    expected = torch.zeros(1, 1, 3, 2, 5)
    expected[..., :2, :, :4] = sample.detach()[..., :2, :]
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    expected_gradient = torch.zeros_like(sample)
    expected_gradient[..., :2, :] = 1
    torch.testing.assert_close(sample.grad, expected_gradient)


@pytest.mark.parametrize("output_format", ["tuple", "list", "stacked"])
def test_deep_supervision_outputs_and_eval(output_format):
    model = models.MedNeXtV2(
        small_config(deep_supervision=True), deep_supervision_output=output_format
    ).train()
    # Distinct constant logits verify primary-to-bottleneck head ordering.
    for head, value in zip(
        [model.head, *model.deep_supervision_heads], [0, 4, 3, 2, 1], strict=True
    ):
        torch.nn.init.zeros_(head.conv.weight)
        torch.nn.init.constant_(head.conv.bias, value)
    sample = torch.randn(1, 1, 17, 19, 21)
    output = model(sample)
    if output_format == "stacked":
        assert output.shape == (1, 5, 3, 17, 19, 21)
        outputs = output.unbind(1)
    else:
        assert isinstance(output, tuple if output_format == "tuple" else list)
        assert [tuple(item.shape) for item in output] == [
            (1, 3, 17, 19, 21),
            (1, 3, 9, 10, 11),
            (1, 3, 5, 5, 6),
            (1, 3, 3, 3, 3),
            (1, 3, 2, 2, 2),
        ]
        outputs = output
    for value, item in enumerate(outputs):
        torch.testing.assert_close(item, torch.full_like(item, value))
    with torch.no_grad():
        primary = model.eval()(sample)
    assert isinstance(primary, torch.Tensor)
    assert primary.shape == (1, 3, 17, 19, 21)


def test_stacked_outputs_resize_native_logits():
    model = models.MedNeXtV2(small_config(deep_supervision=True)).train()
    sample = torch.randn(1, 1, 16, 17, 32)
    with torch.no_grad():
        native = model(sample)
        model.deep_supervision_output = "stacked"
        stacked = model(sample)
    expected = torch.stack([F.interpolate(item, size=(16, 17, 32)) for item in native], 1)
    torch.testing.assert_close(stacked, expected)


def test_invalid_deep_supervision_output_is_rejected():
    with pytest.raises(ValueError, match="deep_supervision_output"):
        models.MedNeXtV2(small_config(), deep_supervision_output="tensor")


@pytest.mark.parametrize("style", ["expansion", "block"])
def test_checkpoint_policy_preserves_state_and_gradients(style):
    torch.manual_seed(42)
    reference = models.MedNeXtV2(small_config(deep_supervision=True)).train()
    checkpointed = models.MedNeXtV2(
        small_config(deep_supervision=True),
        checkpointing=CheckpointConfig(style=style, stages=(0, 1)),
    ).train()
    checkpointed.load_state_dict(reference.state_dict(), strict=True)
    reference_input = torch.randn(1, 1, 17, 19, 21, requires_grad=True)
    checkpointed_input = reference_input.detach().clone().requires_grad_(True)
    expected = reference(reference_input)
    actual = checkpointed(checkpointed_input)
    sum(item.square().mean() for item in expected).backward()
    sum(item.square().mean() for item in actual).backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(checkpointed_input.grad, reference_input.grad)
    assert checkpointed.state_dict().keys() == reference.state_dict().keys()
    for name, parameter in reference.named_parameters():
        gradient = checkpointed.get_parameter(name).grad
        assert parameter.grad is not None, name
        assert gradient is not None, name
        torch.testing.assert_close(
            gradient, parameter.grad, msg=lambda msg, name=name: f"{name}: {msg}"
        )


def test_custom_config_honors_independent_resampling_ratios():
    config = replace(
        small_config(),
        downsample_expansion_ratios=(1, 2, 3, 4),
        upsample_expansion_ratios=(4, 3, 2, 1),
    )
    model = models.MedNeXtV2(config)
    assert [b.expand.out_channels // b.expand.in_channels for b in model.downsamples] == [
        1,
        2,
        3,
        4,
    ]
    assert [b.expand.out_channels // b.expand.in_channels for b in model.upsamples] == [4, 3, 2, 1]


@pytest.mark.parametrize("variant", ["base", "wide"])
def test_factory_installs_both_adaptive_families_and_preserves_state(monkeypatch, variant):
    from mednext_accel.models import mednext_v2 as v2
    from mednext_accel.ops import AdaptiveDepthwise3d, AdaptiveGlobalResponseNorm3d
    from mednext_accel.ops.adaptive import AdaptivePointwise3d

    monkeypatch.setattr(v2, "get_mednext_v2_config", lambda *a, **kw: small_config())
    factory = getattr(models, f"mednext_v2_{variant}")
    reference = factory(in_channels=1, out_channels=3, optimization="reference")
    automatic = factory(in_channels=1, out_channels=3, optimization="auto")
    assert isinstance(automatic.stem, AdaptivePointwise3d)
    assert isinstance(automatic.encoder_stages[0][0].depthwise, AdaptiveDepthwise3d)
    assert isinstance(automatic.encoder_stages[0][0].grn, AdaptiveGlobalResponseNorm3d)
    assert automatic.state_dict().keys() == reference.state_dict().keys()
    automatic.load_state_dict(reference.state_dict(), strict=True)
    assert not any(
        isinstance(m, (AdaptivePointwise3d, AdaptiveDepthwise3d, AdaptiveGlobalResponseNorm3d))
        for m in reference.modules()
    )
    parameters = dict(reference.named_parameters())
    assert automatic.stem.model_context.family == "mednext_v2"
    x = torch.randn(2, 1, 16, 17, 18, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = reference(x), automatic(y)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(automatic.get_parameter(name).grad, parameter.grad)
    v2._configure_optimization(reference, "auto")
    assert all(reference.get_parameter(name) is p for name, p in parameters.items())


@pytest.mark.parametrize("source", ["torch", "conservative", "autotune"])
def test_v2_rejects_removed_optimization_sources(monkeypatch, source):
    from mednext_accel.models import mednext_v2 as v2

    monkeypatch.setattr(v2, "get_mednext_v2_config", lambda *a, **kw: small_config())
    with pytest.raises(ValueError, match="was removed"):
        models.mednext_v2_base(in_channels=1, out_channels=3, optimization=source)


def test_optimization_report_uses_each_operator_input_resolution(monkeypatch):
    from mednext_accel.models import mednext_v2 as v2
    from mednext_accel.ops import AdaptiveDepthwise3d, AdaptiveGlobalResponseNorm3d
    from mednext_accel.ops.adaptive import AdaptivePointwise3d

    model = v2._configure_optimization(
        models.MedNeXtV2(small_config(deep_supervision=True)), "auto"
    )
    actual_shapes = {}

    def record_input(role):
        def hook(module, args):
            actual_shapes[role] = tuple(args[0].shape[2:])

        return hook

    handles = [
        m.register_forward_pre_hook(record_input(role))
        for role, m in model.named_modules()
        if isinstance(m, (AdaptiveDepthwise3d, AdaptivePointwise3d, AdaptiveGlobalResponseNorm3d))
    ]
    with torch.no_grad():
        model(torch.randn(1, 1, 17, 19, 21))
    for handle in handles:
        handle.remove()
    resolved_shapes = {}
    resolver = model._optimization_resolver
    original = resolver.resolve

    def resolve(descriptor, context, phase):
        resolved_shapes[descriptor.role] = context.spatial_shape
        return original(descriptor, context, phase)

    monkeypatch.setattr(resolver, "resolve", resolve)
    report = model.explain_optimization(
        input_shape=(1, 1, 17, 19, 21), dtype="bfloat16", device="cpu"
    )
    assert resolved_shapes == actual_shapes
    grn = [d for d in report.decisions if d.descriptor.family == "global_response_norm3d"]
    assert len(grn) == 17
    assert all(d.phase == "training" for d in grn)
    assert all("kernel_size" not in d.descriptor.to_primitive() for d in grn)
    depthwise = [d for d in report.decisions if d.descriptor.role == "upsamples.0.depthwise"]
    assert {d.phase for d in depthwise} == {"backward_input", "backward_weight"}


def test_external_grn_policy_report_and_native_eval(monkeypatch):
    from types import SimpleNamespace

    from mednext_accel.models import mednext_v2 as v2

    policy = {
        "version": 2,
        "kind": "mednext-accel-policy",
        "name": "v2-grn",
        "target": {"vendor": "nvidia"},
        "scope": {"model_family": "mednext_v2", "checkpointing": "all-expansion"},
        "rules": [
            {
                "id": "grn",
                "when": {"family": "global_response_norm3d"},
                "use": {"training": {"implementation": "triton_fused_grn"}},
                "confidence": "measured-exact-context",
            }
        ],
    }
    monkeypatch.setattr(v2, "get_mednext_v2_config", lambda *a, **kw: small_config())
    model = models.mednext_v2_base(
        in_channels=1,
        out_channels=3,
        optimization=policy,
        checkpointing=CheckpointConfig(style="expansion"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 6))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(total_memory=24 * 2**30)
    )
    report = model.explain_optimization(
        input_shape=(2, 1, 32, 32, 32), dtype="bfloat16", device="cuda"
    )
    grn = [d for d in report.decisions if d.descriptor.family == "global_response_norm3d"]
    assert len(grn) == 17
    assert all(d.implementation == "triton_fused_grn" and d.policy == "v2-grn" for d in grn)
    original = model._optimization_resolver.resolve

    def resolve(descriptor, context, phase):
        assert descriptor.family != "global_response_norm3d"
        return original(descriptor, context, phase)

    monkeypatch.setattr(model._optimization_resolver, "resolve", resolve)
    report = model.eval().explain_optimization(
        input_shape=(2, 1, 32, 32, 32), dtype="bfloat16", device="cuda"
    )
    grn = [d for d in report.decisions if d.descriptor.family == "global_response_norm3d"]
    assert len(grn) == 17
    assert all(d.implementation == "reference" and d.disposition == "native" for d in grn)


def test_reference_optimization_report_and_input_rank():
    model = models.MedNeXtV2(small_config())
    report = model.explain_optimization(
        input_shape=(2, 1, 32, 32, 32), dtype="float32", device="cpu"
    )
    assert report.decisions
    assert all(d.implementation == "reference" for d in report.decisions)
    assert (
        len([d for d in report.decisions if d.descriptor.family == "global_response_norm3d"]) == 17
    )
    with pytest.raises(ValueError, match="rank 5"):
        model.explain_optimization(input_shape=(1, 1, 32, 32), dtype="float32", device="cpu")
