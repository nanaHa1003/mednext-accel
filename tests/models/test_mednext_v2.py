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
