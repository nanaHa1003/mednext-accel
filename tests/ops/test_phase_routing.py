"""Real CUDA phase dispatch, native masks, autocast and compiled autograd."""

import copy

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from mednext_accel.ops.adaptive import AdaptiveDepthwise3d, ModelOptimizationContext
from mednext_accel.optimization.policy import parse_policy
from mednext_accel.optimization.policy_resolver import PolicyResolver

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def wrapper(conv, dx, dw):
    policy = parse_policy(
        {
            "version": 2,
            "kind": "mednext-accel-policy",
            "name": "phase-test",
            "target": {"vendor": "nvidia"},
            "rules": [
                {
                    "id": "regular",
                    "when": {"family": "depthwise_conv3d", "direction": "regular"},
                    "use": {
                        "backward_input": {
                            "implementation": "triton_depthwise_dx" if dx else "reference",
                            **({"parameters": "auto"} if dx else {}),
                        },
                        "backward_weight": {
                            "implementation": "triton_split_dw" if dw else "reference",
                            **({"parameters": "auto"} if dw else {}),
                        },
                    },
                    "confidence": "measured-exact-context",
                }
            ],
        }
    )
    return AdaptiveDepthwise3d(
        conv,
        PolicyResolver(external=policy),
        ModelOptimizationContext("mednext_v1", "base", "none"),
    )


class ObserveConvolution(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.masks = []
        self.operations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(func._schema.name)
        if func._schema.name == "aten::convolution_backward":
            self.masks.append(tuple(args[-1]))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize(("dx", "dw"), [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("autocast", [False, True])
def test_regular_decisions_execute_independently_with_native_phase_masks(dx, dw, autocast):
    torch.manual_seed(19)
    reference = nn.Conv3d(4, 4, 3, padding=1, groups=4, device="cuda")
    candidate = wrapper(copy.deepcopy(reference), dx, dw)
    x = torch.randn(2, 4, 8, 8, 8, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        expected = reference(x)
        with ObserveConvolution() as forward:
            actual = candidate(y)
    assert ("mednext_accel::depthwise_conv3d_odd_regular" in forward.operations) == (dx or dw)
    gradient = torch.randn_like(expected)
    wanted = torch.autograd.grad(expected, (x, *reference.parameters()), gradient)
    with ObserveConvolution() as backward:
        got = torch.autograd.grad(actual, (y, *candidate.parameters()), gradient)
    if dx or dw:
        assert backward.masks == [(not dx, not dw, True)]
        assert ("mednext_accel::depthwise_input_grad_odd_regular" in backward.operations) == dx
        assert ("mednext_accel::depthwise_weight_grad_odd_regular" in backward.operations) == dw
    torch.testing.assert_close(actual, expected)
    for value, target in zip(got, wanted, strict=True):
        assert value.dtype == torch.float32
        error = (value.float() - target.float()).norm() / target.float().norm()
        assert error < (0.02 if autocast else 0.0001)


@pytest.mark.parametrize(("dx", "dw"), [(True, True), (True, False), (False, True), (False, False)])
def test_phase_routing_compiles_with_fp32_parameters(dx, dw):
    torch.manual_seed(31)
    reference = nn.Conv3d(2, 2, 3, padding=1, groups=2, device="cuda")
    candidate = wrapper(copy.deepcopy(reference), dx, dw)
    compiled = torch.compile(candidate)
    x = torch.randn(1, 2, 5, 5, 5, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual, expected = compiled(y), reference(x)
    gradient = torch.randn_like(expected)
    got = torch.autograd.grad(actual, (y, *candidate.parameters()), gradient)
    wanted = torch.autograd.grad(expected, (x, *reference.parameters()), gradient)
    assert actual.dtype == expected.dtype
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.02
    for value, target in zip(got, wanted, strict=True):
        assert (value.float() - target.float()).norm() / target.float().norm() < 0.02


@pytest.mark.parametrize("guard", ["noncontiguous", "unbatched", "no-grad", "output-padding"])
def test_tensor_guards_prevent_custom_execution(guard):
    candidate = wrapper(nn.Conv3d(2, 2, 3, padding=1, groups=2, device="cuda"), True, True)
    x = torch.randn(1, 2, 5, 5, 5, device="cuda", requires_grad=True)
    if guard == "noncontiguous":
        x = x.transpose(2, 3)
    if guard == "unbatched":
        x = x.squeeze(0)
    if guard == "output-padding":
        candidate.output_padding = (1, 0, 0)
    with torch.set_grad_enabled(guard != "no-grad"), ObserveConvolution() as observed:
        candidate(x)
    assert "mednext_accel::depthwise_conv3d_odd_regular" not in observed.operations
