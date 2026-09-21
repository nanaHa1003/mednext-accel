import copy

import torch
from torch import nn

from mednext_accel.ops.adaptive import (
    AdaptiveDepthwise3d,
    AdaptivePointwise3d,
    ModelOptimizationContext,
    install_adaptive_operators,
)
from mednext_accel.optimization.profiles import ProfileRegistry


def _resolver():
    return ProfileRegistry().resolver_for(sm=(12, 0))


def test_installation_preserves_parameters_and_state_dict_keys() -> None:
    model = nn.Sequential(
        nn.Conv3d(32, 32, 3, padding=1, groups=32),
        nn.Conv3d(32, 64, 1),
    )
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())
    count = install_adaptive_operators(
        model, resolver=_resolver(),
        model_context=ModelOptimizationContext("mednext_v1", "base", "none"),
    )
    assert count == 2
    assert tuple(model.parameters()) == parameters
    assert tuple(model.state_dict()) == keys
    assert isinstance(model[0], AdaptiveDepthwise3d)
    assert isinstance(model[1], AdaptivePointwise3d)


def test_cpu_forward_backward_matches_reference() -> None:
    torch.manual_seed(3)
    reference = nn.Sequential(
        nn.Conv3d(3, 3, 3, padding=1, groups=3), nn.Conv3d(3, 7, 1)
    ).double()
    candidate = copy.deepcopy(reference)
    install_adaptive_operators(
        candidate, resolver=_resolver(),
        model_context=ModelOptimizationContext("mednext_v1", "base", "none"),
    )
    x = torch.randn(2, 3, 5, 6, 7, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = reference(x), candidate(y)
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(expected)
    expected_grad = torch.autograd.grad(expected, (x, *reference.parameters()), gradient)
    actual_grad = torch.autograd.grad(actual, (y, *candidate.parameters()), gradient)
    for got, want in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(got, want)


def test_batch_three_can_be_inspected_without_cuda_execution() -> None:
    wrapper = AdaptivePointwise3d(
        nn.Conv3d(32, 64, 1), _resolver(),
        ModelOptimizationContext("mednext_v1", "base", "all-expansion"),
    )
    decision = wrapper.decision_for_shape(
        batch_size=3, spatial_shape=(128, 128, 128), dtype="bfloat16",
        device_type="cuda", sm=(12, 0), total_vram_bytes=32 * 2**30,
    )
    assert decision.implementation == "pointwise_gemm_per_sample"
