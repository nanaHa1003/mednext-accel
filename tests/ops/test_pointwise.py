from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from mednext_accel import mednext_base
from mednext_accel.ops.pointwise import GemmPointwise3d, replace_pointwise_convs


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("bias", [False, True])
def test_outputs_gradients_and_parameter_identity(batch: int, bias: bool) -> None:
    torch.manual_seed(7)
    original = nn.Sequential(nn.Conv3d(3, 7, 1, bias=bias)).double()
    candidate = copy.deepcopy(original)
    parameters = list(candidate.parameters())

    assert replace_pointwise_convs(candidate) == 1
    assert isinstance(candidate[0], GemmPointwise3d)
    assert all(
        before is after for before, after in zip(parameters, candidate.parameters(), strict=True)
    )
    assert list(candidate.state_dict()) == list(original.state_dict())

    x = torch.randn(batch, 3, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    z = x.detach().clone().requires_grad_()
    expected, actual = original(x), candidate(z)
    gradient = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, (x, *original.parameters()), gradient)
    actual_gradients = torch.autograd.grad(actual, (z, *candidate.parameters()), gradient)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def test_shape_selection_falls_back_to_convolution() -> None:
    conv = nn.Conv3d(3, 7, 1).double()
    candidate = GemmPointwise3d(conv, selected_shapes={(3, 7, 8, 8, 8)})
    x = torch.randn(1, 3, 3, 4, 5, dtype=torch.float64)

    torch.testing.assert_close(candidate(x), conv(x), rtol=0, atol=0)


def test_ineligible_layers_are_preserved() -> None:
    layers = nn.Sequential(
        nn.Conv3d(4, 4, 3),
        nn.Conv3d(4, 4, 1, stride=2),
        nn.Conv3d(4, 4, 1, groups=4),
        nn.ConvTranspose3d(4, 4, 1),
        nn.Conv3d(4, 4, 1, padding=1),
    )
    before = list(layers)

    assert replace_pointwise_convs(layers) == 0
    assert list(layers) == before


def test_mednext_state_dict_paths_are_preserved() -> None:
    model = mednext_base(in_channels=1, out_channels=3, base_channels=2)
    keys = tuple(model.state_dict())

    count = replace_pointwise_convs(model)

    assert count == 53
    assert tuple(model.state_dict()) == keys
