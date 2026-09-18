from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from mednext_accel import (
    CheckpointConfig,
    MedNeXtV1,
    mednext_base,
    mednext_large,
    mednext_medium,
    mednext_small,
)


@pytest.mark.parametrize(
    "factory",
    [mednext_small, mednext_base, mednext_medium, mednext_large],
)
def test_factories_build_declared_variant(
    factory: Callable[..., MedNeXtV1],
) -> None:
    model = factory(in_channels=1, out_channels=3, base_channels=2)

    assert isinstance(model, MedNeXtV1)
    assert model.config.variant in factory.__name__


@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_small_eval_returns_primary_logits(spatial_dims: int) -> None:
    model = mednext_small(
        in_channels=1,
        out_channels=3,
        spatial_dims=spatial_dims,
        base_channels=2,
        deep_supervision=True,
    ).eval()
    shape = (1, 1, *([32] * spatial_dims))

    with torch.no_grad():
        output = model(torch.randn(shape))

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, *([32] * spatial_dims))


def test_base_parameter_count_matches_official_without_dummy_tensor() -> None:
    model = mednext_base(in_channels=1, out_channels=3, deep_supervision=True)

    assert sum(parameter.numel() for parameter in model.parameters()) == 10_529_231


@pytest.mark.parametrize("output_format", ["tuple", "list", "stacked"])
def test_deep_supervision_training_output_formats(output_format: str) -> None:
    model = mednext_small(
        in_channels=1,
        out_channels=3,
        base_channels=2,
        deep_supervision=True,
        deep_supervision_output=output_format,
    ).train()

    output = model(torch.randn(1, 1, 32, 32, 32))

    expected_shapes = [
        (1, 3, 32, 32, 32),
        (1, 3, 16, 16, 16),
        (1, 3, 8, 8, 8),
        (1, 3, 4, 4, 4),
        (1, 3, 2, 2, 2),
    ]
    if output_format == "stacked":
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 5, 3, 32, 32, 32)
    else:
        assert isinstance(output, tuple if output_format == "tuple" else list)
        assert [tuple(item.shape) for item in output] == expected_shapes


def test_invalid_deep_supervision_output_is_rejected() -> None:
    with pytest.raises(ValueError, match="deep_supervision_output"):
        mednext_small(
            in_channels=1,
            out_channels=3,
            deep_supervision_output="tensor",
        )


@pytest.mark.parametrize("style", ["expansion", "block"])
def test_checkpoint_policy_preserves_state_and_gradients(style: str) -> None:
    torch.manual_seed(42)
    reference = mednext_small(
        in_channels=1,
        out_channels=2,
        base_channels=2,
    ).train()
    checkpointed = mednext_small(
        in_channels=1,
        out_channels=2,
        base_channels=2,
        checkpointing=CheckpointConfig(style=style, stages=(0, 1)),
    ).train()
    checkpointed.load_state_dict(reference.state_dict(), strict=True)
    reference_input = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
    checkpointed_input = reference_input.detach().clone().requires_grad_(True)

    reference_output = reference(reference_input)
    checkpointed_output = checkpointed(checkpointed_input)
    assert isinstance(reference_output, torch.Tensor)
    assert isinstance(checkpointed_output, torch.Tensor)
    reference_output.square().mean().backward()
    checkpointed_output.square().mean().backward()

    torch.testing.assert_close(checkpointed_output, reference_output)
    torch.testing.assert_close(checkpointed_input.grad, reference_input.grad)
    assert checkpointed.state_dict().keys() == reference.state_dict().keys()
    for expected, actual in zip(reference.parameters(), checkpointed.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("style", ["expansion", "block"])
def test_checkpoint_policy_compiles_fullgraph(style: str) -> None:
    model = (
        mednext_small(
            in_channels=1,
            out_channels=2,
            base_channels=2,
            checkpointing=CheckpointConfig(style=style),
        )
        .cuda()
        .train()
    )
    model.compile(mode="default", fullgraph=True)
    sample = torch.randn(2, 1, 32, 32, 32, device="cuda", requires_grad=True)

    output = model(sample)
    assert isinstance(output, torch.Tensor)
    output.square().mean().backward()

    assert sample.grad is not None


def test_approximate_gelu_is_eval_only() -> None:
    model = mednext_small(
        in_channels=1,
        out_channels=2,
        base_channels=2,
        approximate_gelu_eval=True,
    )
    activations = [module for module in model.modules() if isinstance(module, nn.GELU)]

    assert activations == []
    assert any(module.__class__.__name__ == "EvalModeGELU" for module in model.modules())
