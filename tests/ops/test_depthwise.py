from __future__ import annotations

import copy
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mednext_accel.ops.depthwise import replace_depthwise_convs


def test_importing_dispatch_does_not_import_triton() -> None:
    code = (
        "import sys; import mednext_accel.ops.depthwise; "
        "print(any(x == 'triton' or x.startswith('triton.') for x in sys.modules))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_replacement_preserves_parameters_and_state_dict_paths() -> None:
    model = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1, groups=32))
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())

    assert replace_depthwise_convs(model) == 1

    assert tuple(model.parameters()) == parameters
    assert tuple(model.state_dict()) == keys


def test_stride_two_downsample_is_explicitly_selected() -> None:
    native = nn.Sequential(nn.Conv3d(32, 32, 3, stride=2, padding=1, groups=32))
    selected = copy.deepcopy(native)

    assert replace_depthwise_convs(native) == 0
    assert replace_depthwise_convs(selected, include_stride2_input_grad=True) == 1


def test_stride_two_transpose_is_selected() -> None:
    model = nn.Sequential(nn.ConvTranspose3d(64, 64, 3, stride=2, padding=1, groups=64))

    assert replace_depthwise_convs(model) == 1


def test_eval_uses_traceable_native_convolution(monkeypatch: pytest.MonkeyPatch) -> None:
    import mednext_accel.ops._triton.depthwise as backend

    conv = nn.Conv3d(32, 32, 3, padding=1, groups=32)
    reference = copy.deepcopy(conv).eval()
    model = nn.Sequential(conv).eval()
    replace_depthwise_convs(model)
    monkeypatch.setitem(backend._REGULAR_CONFIGS, (32, 8), (2, 128, 128))

    def fail(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("custom training operator was called during evaluation")

    monkeypatch.setattr(backend, "depthwise_conv3d_regular", fail)
    example = torch.randn(1, 32, 8, 8, 8)

    with torch.no_grad():
        expected = reference(example)
        actual = model(example)
        traced = torch.jit.trace(model, example)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "mednext_accel::" not in str(traced.inlined_graph)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [1, 2])
def test_regular_wrapper_matches_forward_and_all_gradients(
    batch_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mednext_accel.ops._triton.depthwise as backend

    torch.manual_seed(5)
    original = (
        nn.Sequential(nn.Conv3d(128, 128, 3, padding=1, groups=128)).cuda().to(torch.bfloat16)
    )
    candidate = copy.deepcopy(original)
    replace_depthwise_convs(candidate)
    calls = 0
    custom_op = backend.depthwise_conv3d_regular

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        return custom_op(*args, **kwargs)

    monkeypatch.setattr(backend, "depthwise_conv3d_regular", tracked)
    x = torch.randn(
        batch_size,
        128,
        32,
        32,
        32,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    z = x.detach().clone().requires_grad_()

    expected, actual = original(x), candidate(z)
    gradient = torch.randn_like(expected)
    expected_all = (
        expected,
        *torch.autograd.grad(expected, (x, *original.parameters()), gradient),
    )
    actual_all = (
        actual,
        *torch.autograd.grad(actual, (z, *candidate.parameters()), gradient),
    )

    for actual_tensor, expected_tensor in zip(actual_all, expected_all, strict=True):
        relative_error = (
            actual_tensor.float() - expected_tensor.float()
        ).norm() / expected_tensor.float().norm().clamp_min(1e-12)
        assert relative_error.item() < 0.01
    assert calls == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["transpose", "downsample"])
def test_resampling_wrapper_matches_batch_two_gradients(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mednext_accel.ops._triton.depthwise as backend

    torch.manual_seed(11)
    if kind == "transpose":
        conv = nn.ConvTranspose3d(3, 3, 3, stride=2, padding=1, groups=3)
        wrapper_type = backend.SplitDepthwiseConvTranspose3d
        monkeypatch.setitem(backend._TRANSPOSE_CONFIGS, (3, 4), (2, 128))
        operation_name = "depthwise_conv_transpose3d_regular"
        spatial = 4
    else:
        conv = nn.Conv3d(3, 3, 3, stride=2, padding=1, groups=3)
        wrapper_type = backend.SplitDepthwiseDownsampleConv3d
        monkeypatch.setitem(backend._DOWNSAMPLE_CONFIGS, (3, 8), 128)
        operation_name = "depthwise_conv3d_stride2_regular"
        spatial = 8
    original = nn.Sequential(conv).cuda().to(torch.bfloat16)
    candidate = nn.Sequential(wrapper_type(copy.deepcopy(conv))).cuda().to(torch.bfloat16)
    calls = 0
    custom_op = getattr(backend, operation_name)

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        return custom_op(*args, **kwargs)

    monkeypatch.setattr(backend, operation_name, tracked)
    x = torch.randn(
        2, 3, spatial, spatial, spatial, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    z = x.detach().clone().requires_grad_()
    expected, actual = original(x), candidate(z)
    gradient = torch.randn_like(expected)
    expected_all = (expected, *torch.autograd.grad(expected, (x, *original.parameters()), gradient))
    actual_all = (actual, *torch.autograd.grad(actual, (z, *candidate.parameters()), gradient))

    for actual_tensor, expected_tensor in zip(actual_all, expected_all, strict=True):
        relative_error = (
            actual_tensor.float() - expected_tensor.float()
        ).norm() / expected_tensor.float().norm().clamp_min(1e-12)
        assert relative_error.item() < 0.01
    assert calls == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_custom_operator_registration_with_opcheck() -> None:
    from mednext_accel.ops._triton.depthwise import depthwise_conv3d_regular

    x = torch.randn(1, 2, 5, 5, 5, device="cuda", requires_grad=True)
    weight = torch.randn(2, 1, 3, 3, 3, device="cuda", requires_grad=True)
    bias = torch.randn(2, device="cuda", requires_grad=True)

    results = torch.library.opcheck(
        depthwise_conv3d_regular,
        (x, weight, bias, 3, 2, 5, 2, 128, 128),
        test_utils=("test_schema", "test_autograd_registration", "test_faketensor"),
        raise_exception=False,
    )

    assert results == {
        "test_schema": "SUCCESS",
        "test_autograd_registration": "SUCCESS",
        "test_faketensor": "SUCCESS",
    }


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_low_level_regular_gradients_against_double_precision(kernel_size: int) -> None:
    from mednext_accel.ops._triton.depthwise import (
        depthwise_input_grad,
        depthwise_weight_grad,
    )

    torch.manual_seed(17 + kernel_size)
    x = torch.randn(2, 3, 5, 6, 7, device="cuda", dtype=torch.float64, requires_grad=True)
    weight = torch.randn(
        3,
        1,
        kernel_size,
        kernel_size,
        kernel_size,
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    output = F.conv3d(x, weight, padding=kernel_size // 2, groups=3)
    gradient = torch.randn_like(output)
    expected_input, expected_weight = torch.autograd.grad(output, (x, weight), gradient)

    actual_input = depthwise_input_grad(gradient, weight.detach(), block=128)
    actual_weight = depthwise_weight_grad(
        x.detach(), gradient, splits=4, block=128, kernel_size=kernel_size
    )

    torch.testing.assert_close(actual_input, expected_input, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(actual_weight, expected_weight, rtol=2e-5, atol=2e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_low_level_stride_two_input_gradient_against_double_precision() -> None:
    from mednext_accel.ops._triton.depthwise import depthwise_stride2_input_grad

    torch.manual_seed(47)
    x = torch.randn(2, 3, 5, 6, 7, device="cuda", dtype=torch.float64, requires_grad=True)
    weight = torch.randn(3, 1, 3, 3, 3, device="cuda", dtype=torch.float64)
    output = F.conv3d(x, weight, stride=2, padding=1, groups=3)
    gradient = torch.randn_like(output)
    (expected,) = torch.autograd.grad(output, x, gradient)

    actual = depthwise_stride2_input_grad(gradient, weight, x.shape[2:], block=128)

    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_low_level_transpose_weight_gradient_against_double_precision() -> None:
    from mednext_accel.ops._triton.depthwise import depthwise_transpose_weight_grad

    torch.manual_seed(41)
    x = torch.randn(2, 3, 4, 5, 6, device="cuda", dtype=torch.float64)
    weight = torch.randn(3, 1, 3, 3, 3, device="cuda", dtype=torch.float64, requires_grad=True)
    output = F.conv_transpose3d(x, weight, stride=2, padding=1, groups=3)
    gradient = torch.randn_like(output)
    (expected,) = torch.autograd.grad(output, weight, gradient)

    actual = depthwise_transpose_weight_grad(x, gradient, splits=4, block=128)

    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
