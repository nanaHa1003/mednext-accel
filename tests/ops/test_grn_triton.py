"""Fused GRN contract, numerical oracle, and PyTorch integration coverage."""

import importlib

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from mednext_accel.ops.grn import global_response_norm3d_reference

pytest.importorskip("triton")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DTYPES = (torch.float16, torch.bfloat16)


def backend():
    return importlib.import_module("mednext_accel.ops._triton.grn")


def inputs(shape, dtype, *, device="cuda", requires_grad=True):
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    gamma = torch.randn(1, shape[1], 1, 1, 1, device=device, requires_grad=requires_grad)
    beta = torch.randn_like(gamma, requires_grad=requires_grad)
    return x, gamma, beta


def compare_with_reference(x, gamma, beta, upstream, *, eps=1e-6):
    expected_inputs = tuple(t.detach().clone().requires_grad_() for t in (x, gamma, beta))
    expected = global_response_norm3d_reference(*expected_inputs, eps)
    actual = backend().fused_global_response_norm3d(x, gamma, beta, eps)
    expected_grads = torch.autograd.grad(expected, expected_inputs, upstream)
    actual_grads = torch.autograd.grad(actual, (x, gamma, beta), upstream)
    activation_tolerance = dict(rtol=2e-3, atol=2e-3)
    if x.dtype == torch.bfloat16:
        activation_tolerance = dict(rtol=1e-2, atol=1.6e-2)
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, **activation_tolerance)
    for index, (got, want) in enumerate(zip(actual_grads, expected_grads, strict=True)):
        assert torch.isfinite(got).all()
        assert got.dtype == (x.dtype if index == 0 else torch.float32)
        tolerance = activation_tolerance if index == 0 else dict(rtol=2e-4, atol=3e-5)
        torch.testing.assert_close(got, want, **tolerance)
    assert torch.isfinite(actual).all()
    return actual, actual_grads


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("batch", (1, 2, 3))
@pytest.mark.parametrize("channels", (3, 32, 96))
@pytest.mark.parametrize("spatial", ((2, 3, 5), (8, 8, 8), (17, 9, 5)))
@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_and_all_gradients_match_fp32_oracle(batch, channels, spatial, dtype):
    torch.manual_seed(902)
    x, gamma, beta = inputs((batch, channels, *spatial), dtype)
    compare_with_reference(x, gamma, beta, torch.randn_like(x))


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "case", ("zero", "tiny", "one_zero_channel", "multiple_tiles", "split_loop")
)
def test_zero_norm_tiny_values_and_spatial_reduction_tails(dtype, case):
    torch.manual_seed(903)
    spatial = (17, 13, 11) if case == "multiple_tiles" else (2, 3, 5)
    if case == "split_loop":
        spatial = (41, 41, 41)  # More than 64 * 1024 elements, with a final masked tile.
    x, gamma, beta = inputs((2, 3, *spatial), dtype)
    with torch.no_grad():
        if case == "zero":
            x.zero_()
        elif case == "tiny":
            x.copy_(torch.sign(x) * torch.finfo(dtype).tiny)
        elif case == "one_zero_channel":
            x[:, 1].zero_()
    upstream = torch.randn_like(x)
    _, grads = compare_with_reference(x, gamma, beta, upstream)
    if case == "zero":
        torch.testing.assert_close(grads[0], upstream, rtol=0, atol=0)
        torch.testing.assert_close(grads[1], torch.zeros_like(gamma), rtol=0, atol=0)


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_couples_channels_through_denominator(dtype):
    x = torch.tensor([1.0, 2.0, 4.0], device="cuda", dtype=dtype).view(1, 3, 1, 1, 1)
    x.requires_grad_()
    gamma = torch.tensor([2.0, 3.0, -4.0], device="cuda").view(1, 3, 1, 1, 1)
    gamma.requires_grad_()
    beta = torch.zeros_like(gamma, requires_grad=True)
    upstream = torch.zeros_like(x)
    upstream[:, 0] = 1
    _, grads = compare_with_reference(x, gamma, beta, upstream)
    # With q only in channel 0, other channels still receive -2 / (7 + eps)**2.
    torch.testing.assert_close(
        grads[0][:, 1:].float(),
        torch.full_like(grads[0][:, 1:].float(), -2 / (7 + 1e-6) ** 2),
        rtol=5e-3,
        atol=1e-4,
    )


def test_public_entry_rejects_cpu():
    args = inputs((1, 3, 2, 3, 5), torch.float16, device="cpu", requires_grad=False)
    with pytest.raises(ValueError, match="^fused GRN requires CUDA tensors$"):
        backend().fused_global_response_norm3d(*args)


@pytest.mark.parametrize(
    "case,message",
    [
        ("rank", "fused GRN expects a rank-5 NCDHW input"),
        ("fp32", "fused GRN requires a float16 or bfloat16 input"),
        ("strided", "fused GRN requires a contiguous NCDHW input"),
        ("affine_dtype", "fused GRN requires float32 gamma and beta"),
        ("affine_shape", "fused GRN requires gamma and beta with shape"),
        ("affine_stride", "fused GRN requires contiguous gamma and beta"),
        ("device", "fused GRN requires all tensors on the same CUDA device"),
        ("empty", "fused GRN requires nonempty input dimensions"),
        ("epsilon", "fused GRN requires eps to be finite and positive"),
    ],
)
def test_public_entry_rejects_unsupported_cuda_metadata(case, message):
    with FakeTensorMode():
        x, gamma, beta = inputs((2, 3, 2, 3, 5), torch.float16, requires_grad=False)
        eps = 1e-6
        if case == "rank":
            x = x[0]
        elif case == "fp32":
            x = x.float()
        elif case == "strided":
            x = x.transpose(-1, -2)
        elif case == "affine_dtype":
            gamma = gamma.half()
        elif case == "affine_shape":
            beta = beta.flatten()
        elif case == "affine_stride":
            gamma = torch.empty(1, 6, 1, 1, 1, device="cuda")[:, ::2]
        elif case == "device":
            beta = torch.empty_like(beta, device="cuda:1")
        elif case == "empty":
            x = x[:0]
        elif case == "epsilon":
            eps = 0.0
        with pytest.raises(ValueError, match=message):
            backend().fused_global_response_norm3d(x, gamma, beta, eps)


@pytest.mark.parametrize("dtype", DTYPES)
def test_fake_forward_and_backward_preserve_metadata(dtype):
    module = backend()
    with FakeTensorMode():
        x, gamma, beta = inputs((2, 3, 2, 3, 5), dtype, requires_grad=False)
        output = module.fused_global_response_norm3d(x, gamma, beta)
        _, norm, denominator = module._grn_forward(x, gamma, beta, 1e-6)
        grad_input, grad_gamma, grad_beta = module._grn_backward(
            torch.empty_like(x), x, gamma, norm, denominator
        )
    assert output.shape == x.shape and output.dtype == dtype and output.device == x.device
    assert norm.shape == (2, 3) and norm.dtype == torch.float32
    assert denominator.shape == (2,) and denominator.dtype == torch.float32
    for got, want in zip((grad_input, grad_gamma, grad_beta), (x, gamma, beta), strict=True):
        assert got.shape == want.shape and got.dtype == want.dtype and got.device == want.device


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_operator_registration_with_opcheck(dtype):
    torch.manual_seed(904)
    args = inputs((2, 3, 2, 3, 5), dtype)
    result = torch.library.opcheck(backend()._grn_forward, (*args, 1e-6), raise_exception=False)
    assert result and all(value == "SUCCESS" for value in result.values()), result
    x, gamma, beta = (t.detach() for t in args)
    _, norm, denominator = backend()._grn_forward(x, gamma, beta, 1e-6)
    result = torch.library.opcheck(
        backend()._grn_backward,
        (torch.randn_like(x), x, gamma, norm, denominator),
        raise_exception=False,
    )
    assert result and all(value == "SUCCESS" for value in result.values()), result


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_fullgraph_compile_forward_and_autograd_backward(dtype):
    torch.manual_seed(905)
    fused = backend().fused_global_response_norm3d

    def loss_and_output(x, gamma, beta, upstream):
        output = fused(x, gamma, beta)
        return (output * upstream).sum(), output

    compiled = torch.compile(loss_and_output, mode="default", fullgraph=True)
    args = inputs((2, 3, 17, 13, 11), dtype)
    upstream = torch.randn_like(args[0])
    expected_inputs = tuple(t.detach().clone().requires_grad_() for t in args)
    expected_loss, expected_output = loss_and_output(*expected_inputs, upstream)
    expected_grads = torch.autograd.grad(expected_loss, expected_inputs)
    actual_loss, actual_output = compiled(*args, upstream)
    actual_grads = torch.autograd.grad(actual_loss, args)
    torch.testing.assert_close(actual_output, expected_output)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want)


@CUDA
@pytest.mark.cuda
def test_autograd_saves_only_input_gamma_and_compact_statistics():
    args = inputs((2, 3, 17, 13, 11), torch.float16)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = backend().fused_global_response_norm3d(*args)
    assert len(saved) == 4
    assert saved[0] is args[0] and saved[1] is args[1]
    assert [tuple(t.shape) for t in saved[2:]] == [(2, 3), (2,)]
    # A sum supplies an expanded, noncontiguous upstream gradient to backward.
    gradients = torch.autograd.grad(output.sum(), args)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@CUDA
@pytest.mark.cuda
@pytest.mark.parametrize(
    "name",
    (
        "_partial_norm_kernel",
        "_affine_kernel",
        "_partial_backward_kernel",
        "_input_backward_kernel",
    ),
)
@pytest.mark.parametrize("large", (False, True))
def test_activation_addresses_widen_before_multiplication(name, large):
    from triton.runtime.jit import MockTensor

    kernel = getattr(backend(), name)
    constants = dict(N=64 if large else 2, C=32, SPATIAL=128**3 if large else 2431, BLOCK=1024)
    if "partial" in name:
        constants["SPLITS"] = 64 if large else 3
    pointers = [MockTensor(torch.float32) for arg in kernel.arg_names if arg not in constants]
    compiled = kernel.warmup(*pointers, **constants, grid=(1,), num_warps=4)
    ir = compiled.asm["ttir"]
    addresses = [line for line in ir.splitlines() if "tt.addptr" in line]
    assert addresses
    assert all(("i64" in line) == large for line in addresses), addresses
    if large:
        multiplies = [line for line in ir.splitlines() if "arith.muli" in line]
        assert all("i32" not in line for line in multiplies), multiplies
