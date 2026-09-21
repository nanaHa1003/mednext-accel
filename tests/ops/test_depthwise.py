from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F


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
