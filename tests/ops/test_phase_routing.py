"""Real CUDA phase dispatch, native masks, autocast and compiled autograd."""

import copy
import warnings

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from mednext_accel.ops.adaptive import (
    AdaptiveDepthwise3d,
    AdaptivePointwise3d,
    ModelOptimizationContext,
    _context_from_tensor,
)
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


@pytest.mark.parametrize("layout", ["contiguous", "channels_last_3d", "strided", "unbatched"])
def test_pointwise_gemm_requires_batched_contiguous_ncdhw(layout):
    torch.manual_seed(47)
    reference = nn.Conv3d(4, 8, 1, device="cuda")
    policy = parse_policy(
        {
            "version": 2,
            "kind": "mednext-accel-policy",
            "name": "pointwise-layout",
            "target": {"vendor": "nvidia"},
            "rules": [
                {
                    "id": "pointwise",
                    "when": {"family": "pointwise_conv3d"},
                    "use": {"training": {"implementation": "pointwise_gemm_per_sample"}},
                    "confidence": "measured-exact-context",
                }
            ],
        }
    )
    candidate = AdaptivePointwise3d(
        copy.deepcopy(reference),
        PolicyResolver(external=policy),
        ModelOptimizationContext("mednext_v1", "base", "none"),
    )
    sample = torch.randn(2, 4, 3, 4, 5, device="cuda", requires_grad=True)
    if layout == "channels_last_3d":
        sample = sample.contiguous(memory_format=torch.channels_last_3d)
    elif layout == "strided":
        sample = sample.transpose(-1, -2)
    elif layout == "unbatched":
        sample = sample[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = reference(sample)
        with ObserveConvolution() as observed:
            actual = candidate(sample)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert actual.stride() == expected.stride()
    assert ("aten::addmm" in observed.operations) == (layout == "contiguous")
    assert ("aten::convolution" in observed.operations) == (layout != "contiguous")


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
    compiled = torch.compile(candidate, fullgraph=True)
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


@pytest.mark.parametrize(
    "style, stages", [(None, None), ("block", None), ("expansion", None), ("expansion", (0, 1))]
)
def test_factory_checkpoint_modes_execute_shared_optimized_gradients(style, stages):
    from mednext_accel import CheckpointConfig, mednext_small
    from mednext_accel.optimization.policies import load_bundled_policy

    checkpointing = None if style is None else CheckpointConfig(style=style, stages=stages)
    model = mednext_small(in_channels=1, out_channels=3, checkpointing=checkpointing)
    candidate = next(
        module
        for module in model.modules()
        if isinstance(module, AdaptiveDepthwise3d)
        and module.in_channels == 128
        and module.descriptor.direction == "regular"
    ).cuda()
    # Exercise shared policy execution on this GPU without its exact-SM overlay.
    candidate.resolver = PolicyResolver((load_bundled_policy("shared-nvidia"),))
    reference = nn.Conv3d(128, 128, 3, padding=1, groups=128, device="cuda")
    reference.load_state_dict(candidate.state_dict())
    x = torch.randn(1, 128, 32, 32, 32, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = reference(x)
        with ObserveConvolution() as observed:
            actual = candidate(y)
    assert "mednext_accel::depthwise_conv3d_odd_regular" in observed.operations
    gradient = torch.randn_like(expected)
    wanted = torch.autograd.grad(expected, (x, *reference.parameters()), gradient)
    with ObserveConvolution() as backward:
        got = torch.autograd.grad(actual, (y, *candidate.parameters()), gradient)
    assert backward.masks == [(False, False, True)]
    for value, target in zip(got, wanted, strict=True):
        assert (value.float() - target.float()).norm() / target.float().norm() < 0.02


@pytest.mark.parametrize("kind", ["pointwise", "depthwise"])
@pytest.mark.parametrize("placement", ["construction", "move", "late-policy"])
def test_cross_sm_policy_first_fullgraph_call_warns_once_outside_trace(kind, placement):
    torch.manual_seed(59)
    actual_sm = torch.cuda.get_device_capability()
    target_sm = (8, 9) if actual_sm != (8, 9) else (8, 6)
    phase = "training" if kind == "pointwise" else "backward_input"
    implementation = "pointwise_gemm_per_sample" if kind == "pointwise" else "triton_depthwise_dx"
    policy = parse_policy(
        {
            "version": 2,
            "kind": "mednext-accel-policy",
            "name": "cross-sm-first-call",
            "target": {"vendor": "nvidia", "sm": list(target_sm)},
            "rules": [
                {
                    "id": "custom",
                    "when": {"family": f"{kind}_conv3d", "direction": "regular"},
                    "use": {phase: {"implementation": implementation}},
                    "confidence": "inferred-same-sm",
                }
            ],
        }
    )
    resolver = PolicyResolver(external=policy)
    cls = AdaptivePointwise3d if kind == "pointwise" else AdaptiveDepthwise3d
    reference = nn.Conv3d(
        2,
        2,
        1 if kind == "pointwise" else 3,
        padding=0 if kind == "pointwise" else 1,
        groups=1 if kind == "pointwise" else 2,
        device="cuda" if placement == "construction" else "cpu",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidate = cls(
            copy.deepcopy(reference),
            PolicyResolver() if placement == "late-policy" else resolver,
            ModelOptimizationContext("mednext_v1", "base", "none"),
        ).cuda()
        if placement == "late-policy":
            candidate.resolver = resolver
        assert sum("applying it as requested" in str(item.message) for item in caught) == (
            0 if placement == "late-policy" else 1
        )
        reference = reference.cuda()
        x = torch.randn(1, 2, 5, 5, 5, device="cuda", requires_grad=True)
        y = x.detach().clone().requires_grad_()
        # No prior eager forward or policy report is allowed here.
        actual = torch.compile(candidate, fullgraph=True)(y)
        expected = reference(x)
        gradient = torch.randn_like(expected)
        got = torch.autograd.grad(actual, (y, *candidate.parameters()), gradient)
        wanted = torch.autograd.grad(expected, (x, *reference.parameters()), gradient)
        decision = resolver.resolve(
            candidate.descriptor,
            _context_from_tensor(x, candidate.model_context, training=True),
            phase,
        )
        candidate.cpu().cuda()(x)
    notices = [item for item in caught if "applying it as requested" in str(item.message)]
    assert len(notices) == 1
    assert decision.implementation == implementation
    assert decision.warning == str(notices[0].message)
    for value, target in zip((actual, *got), (expected, *wanted), strict=True):
        assert (value.float() - target.float()).norm() / target.float().norm() < 0.02


def directional_wrapper(conv, direction):
    phases = {
        "regular": {"backward_input": "triton_depthwise_dx", "backward_weight": "triton_split_dw"},
        "downsample": {"backward_input": "triton_downsample_dx"},
        "transpose": {"backward_weight": "triton_transpose_split_dw"},
    }
    policy = parse_policy(
        {
            "version": 2,
            "kind": "mednext-accel-policy",
            "name": "mask-test",
            "target": {"vendor": "nvidia"},
            "rules": [
                {
                    "id": direction,
                    "when": {
                        "family": "depthwise_conv_transpose3d"
                        if direction == "transpose"
                        else "depthwise_conv3d",
                        "direction": direction,
                    },
                    "use": {
                        phase: {"implementation": implementation, "parameters": "auto"}
                        for phase, implementation in phases[direction].items()
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


@pytest.mark.parametrize("direction", ["regular", "downsample", "transpose"])
@pytest.mark.parametrize(
    "mask",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (False, True, True),
        (True, True, True),
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_requested_gradient_masks_match_native_and_skip_unused_phases(direction, mask, compiled):
    torch.manual_seed(47)
    cls = nn.ConvTranspose3d if direction == "transpose" else nn.Conv3d
    reference = cls(
        2, 2, 3, padding=1, stride=1 if direction == "regular" else 2, groups=2, device="cuda"
    )
    candidate = directional_wrapper(copy.deepcopy(reference), direction)
    for module in (reference, candidate):
        module.weight.requires_grad_(mask[1])
        module.bias.requires_grad_(mask[2])
    x = torch.randn(1, 2, 5, 5, 5, device="cuda", dtype=torch.bfloat16, requires_grad=mask[0])
    y = x.detach().clone().requires_grad_(mask[0])
    if compiled:
        torch.compiler.reset()
    forward = torch.compile(candidate, fullgraph=True) if compiled else candidate
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected, actual = reference(x), forward(y)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    with ObserveConvolution() as observed:
        actual.backward(gradient)
    torch.testing.assert_close(actual, expected)
    for wanted, got, needed in zip(
        (x, *reference.parameters()), (y, *candidate.parameters()), mask, strict=True
    ):
        if needed:
            assert got.grad is not None
            assert (
                got.grad.float() - wanted.grad.float()
            ).norm() / wanted.grad.float().norm() < 0.02
        else:
            assert got.grad is None
    if not compiled:
        custom_dx = direction != "transpose" and mask[0]
        custom_dw = direction != "downsample" and mask[1]
        native_mask = (mask[0] and not custom_dx, mask[1] and not custom_dw, mask[2])
        assert observed.masks == ([native_mask] if any(native_mask) else [])
        custom_ops = [op for op in observed.operations if op.startswith("mednext_accel::")]
        assert any("input_grad" in op for op in custom_ops) == custom_dx
        assert any("weight_grad" in op for op in custom_ops) == custom_dw
