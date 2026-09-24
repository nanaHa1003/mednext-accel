from __future__ import annotations

import copy

import pytest
import torch

from mednext_accel.models.blocks import EvalModeGELU
from mednext_accel.models.blocks_v2 import (
    MedNeXtV2Block,
    MedNeXtV2DownBlock,
    MedNeXtV2UpBlock,
)
from mednext_accel.ops.grn import GlobalResponseNorm3d


def test_standard_block_order_and_parameters() -> None:
    block = MedNeXtV2Block(in_channels=4, out_channels=4, expansion_ratio=3)

    assert block.depthwise.groups == 4
    assert block.depthwise.kernel_size == (3, 3, 3)
    assert block.depthwise.padding == (1, 1, 1)
    assert block.expand.out_channels == 12
    assert isinstance(block.activation, EvalModeGELU)
    assert block.activation.approximate_eval is False
    assert isinstance(block.grn, GlobalResponseNorm3d)
    assert block.project.in_channels == 12
    assert block(torch.randn(2, 4, 9, 11, 13)).shape == (2, 4, 9, 11, 13)


def test_resampling_shapes() -> None:
    x = torch.randn(2, 4, 9, 11, 13)
    down = MedNeXtV2DownBlock(in_channels=4, out_channels=8, expansion_ratio=4)
    y = down(x)

    assert y.shape == (2, 8, 5, 6, 7)
    assert MedNeXtV2UpBlock(in_channels=8, out_channels=4, expansion_ratio=8)(y).shape == (
        2,
        4,
        10,
        12,
        14,
    )


@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (MedNeXtV2Block, {"in_channels": 4, "out_channels": 4, "expansion_ratio": 3}),
        (MedNeXtV2DownBlock, {"in_channels": 4, "out_channels": 8, "expansion_ratio": 4}),
        (MedNeXtV2UpBlock, {"in_channels": 8, "out_channels": 4, "expansion_ratio": 8}),
    ],
)
def test_expansion_checkpoint_preserves_forward_input_and_parameter_gradients(
    factory, kwargs
) -> None:
    torch.manual_seed(42)
    reference = factory(**kwargs).double().train()
    checkpointed = copy.deepcopy(reference)
    checkpointed.checkpoint_expansion = True

    reference_input = torch.randn(2, kwargs["in_channels"], 5, 7, 9, dtype=torch.double)
    checkpointed_input = reference_input.detach().clone()
    reference_input.requires_grad_()
    checkpointed_input.requires_grad_()

    reference_output = reference(reference_input)
    checkpointed_output = checkpointed(checkpointed_input)
    reference_output.square().mean().backward()
    checkpointed_output.square().mean().backward()

    torch.testing.assert_close(checkpointed_output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(checkpointed_input.grad, reference_input.grad, rtol=0, atol=0)
    for expected, actual in zip(reference.parameters(), checkpointed.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
