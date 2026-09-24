import builtins
import copy
import sys
from dataclasses import replace
from types import ModuleType

import pytest
import torch
from torch import nn

import mednext_accel.ops.grn as grn
from mednext_accel.ops.adaptive import ModelOptimizationContext, execution_context_for_shape
from mednext_accel.optimization.policies import PolicyRegistry

POLICY = {
    "version": 2,
    "kind": "mednext-accel-policy",
    "name": "test-grn",
    "target": {"vendor": "nvidia"},
    "rules": [
        {
            "id": "grn-training",
            "when": {"family": "global_response_norm3d", "model_family": "mednext_v2"},
            "use": {"training": {"implementation": "triton_fused_grn"}},
            "confidence": "measured-exact-context",
        }
    ],
}
MODEL_CONTEXT = ModelOptimizationContext("mednext_v2", "base", "none")


def context(**changes):
    return replace(
        execution_context_for_shape(
            MODEL_CONTEXT,
            batch_size=2,
            spatial_shape=(3, 4, 5),
            dtype="bfloat16",
            device_type="cuda",
            sm=(8, 6),
            total_vram_bytes=24 * 2**30,
        ),
        **changes,
    )


def wrapper():
    return grn.AdaptiveGlobalResponseNorm3d(
        grn.GlobalResponseNorm3d(3),
        PolicyRegistry(external=POLICY).resolver(),
        MODEL_CONTEXT,
        role="encoder_stages.0.0.grn",
    )


def test_external_grn_policy_reports_tensor_guards():
    module = wrapper()
    decision = module.resolver.resolve(module.descriptor, context(), "training")
    assert decision.implementation == "triton_fused_grn"
    assert decision.execution_guards == ("grad_enabled", "rank_5", "contiguous")
    assert decision.descriptor.to_primitive() == {
        "family": "global_response_norm3d",
        "direction": "regular",
        "in_channels": 3,
        "out_channels": 3,
        "role": "encoder_stages.0.0.grn",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"device_type": "cpu"},
        {"dtype": "float32"},
        {"phase": "inference"},
        {"phase": "export", "export": True},
    ],
)
def test_static_guards_reject_unsupported_context(changes):
    module = wrapper()
    decision = module.resolver.resolve(module.descriptor, context(**changes), "training")
    assert decision.implementation == "reference"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_training_dispatches_one_fused_backend(monkeypatch, dtype):
    module = wrapper()
    sample = torch.randn(2, 3, 3, 4, 5, dtype=dtype, requires_grad=True)
    # Simulate CUDA context so routing is exercised on CPU-only test hosts.
    monkeypatch.setattr(grn, "_context_from_tensor", lambda *a, **kw: context(dtype=str(dtype)[6:]))
    backend = ModuleType("mednext_accel.ops._triton.grn")
    calls = []

    def fused(x, gamma, beta, eps):
        calls.append((x, gamma, beta, eps))
        return x + 7

    backend.fused_global_response_norm3d = fused
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    original = module.resolver.resolve
    phases = []

    def resolve(descriptor, execution_context, phase):
        phases.append(phase)
        return original(descriptor, execution_context, phase)

    monkeypatch.setattr(module.resolver, "resolve", resolve)
    torch.testing.assert_close(module(sample), sample + 7)
    assert phases == ["training"]
    assert len(calls) == 1
    assert calls[0][0] is sample
    assert calls[0][1] is module.gamma
    assert calls[0][2] is module.beta
    assert calls[0][3] == module.eps


@pytest.mark.parametrize("mode", ["eval", "no_grad", "inference_mode"])
def test_inference_bypasses_resolver_and_backend(monkeypatch, mode):
    module = wrapper().train(mode != "eval")
    sample = torch.randn(2, 3, 3, 4, 5)

    def forbidden(*args, **kwargs):
        pytest.fail("inference accessed the resolver or context")

    monkeypatch.setattr(module.resolver, "resolve", forbidden)
    monkeypatch.setattr(grn, "_context_from_tensor", forbidden)
    manager = torch.inference_mode if mode == "inference_mode" else torch.set_grad_enabled
    with manager() if mode == "inference_mode" else manager(mode == "eval"):
        torch.testing.assert_close(module(sample), sample)


@pytest.mark.parametrize("mode", ["cpu", "fp32", "noncontiguous", "export", "missing_backend"])
def test_unsupported_execution_uses_reference(monkeypatch, mode):
    module = wrapper()
    with torch.no_grad():
        module.gamma.fill_(0.3)
        module.beta.fill_(0.2)
    sample = torch.randn(2, 3, 3, 4, 5, dtype=torch.bfloat16)
    if mode == "fp32":
        sample = sample.float()
    if mode == "noncontiguous":
        sample = sample.transpose(-1, -2)
    changes = {"device_type": "cpu"} if mode == "cpu" else {}
    if mode == "fp32":
        changes["dtype"] = "float32"
    if mode == "export":
        changes.update(phase="export", export=True)
    monkeypatch.setattr(grn, "_context_from_tensor", lambda *a, **kw: context(**changes))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.endswith("_triton.grn"):
            if mode == "missing_backend":
                raise ImportError("backend not installed")
            pytest.fail("unsupported execution imported the backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    expected = grn.global_response_norm3d_reference(sample, module.gamma, module.beta, module.eps)
    torch.testing.assert_close(module(sample), expected)


@pytest.mark.parametrize("shape", [(), (3,), (2, 3, 4, 5), (2, 3, 1, 3, 4, 5)])
def test_wrong_rank_retains_reference_error(shape):
    with pytest.raises(ValueError, match="GRN expects an NCDHW tensor"):
        wrapper()(torch.randn(shape))


def test_installation_preserves_parameters_state_training_and_gradients():
    reference = nn.Sequential(nn.Sequential(grn.GlobalResponseNorm3d(3))).double().eval()
    with torch.no_grad():
        reference[0][0].gamma.normal_()
        reference[0][0].beta.normal_()
    model = copy.deepcopy(reference)
    parameters = dict(model.named_parameters())
    keys = tuple(model.state_dict())
    count = grn.install_adaptive_grn(
        model, resolver=PolicyRegistry().resolver(), model_context=MODEL_CONTEXT
    )
    assert count == 1
    assert (
        grn.install_adaptive_grn(
            model, resolver=PolicyRegistry().resolver(), model_context=MODEL_CONTEXT
        )
        == 0
    )
    assert tuple(model.state_dict()) == keys
    assert all(model.get_parameter(name) is p for name, p in parameters.items())
    assert model[0][0].descriptor.role == "0.0"
    assert not model[0][0].training
    x = torch.randn(2, 3, 3, 4, 5, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = reference(x), model(y)
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected.square().sum(), (x, *reference.parameters()))
    actual_grads = torch.autograd.grad(actual.square().sum(), (y, *model.parameters()))
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want)


def test_fp32_tensor_under_autocast_still_uses_reference(monkeypatch):
    module = wrapper()
    sample = torch.randn(2, 3, 3, 4, 5)
    # Convolution context describes autocast dtype, but GRN receives the actual tensor.
    monkeypatch.setattr(grn, "_context_from_tensor", lambda *a, **kw: context())
    backend = ModuleType("mednext_accel.ops._triton.grn")

    def forbidden(*args):
        pytest.fail("FP32 GRN reached the reduced-precision backend")

    backend.fused_global_response_norm3d = forbidden
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(module(sample), sample)
