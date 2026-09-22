"""Compile large address spaces without allocating or launching large tensors."""

import re

import pytest
import torch

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compiler required"),
]


def compile_kernel(name, **constants):
    from triton.runtime.jit import MockTensor

    from mednext_accel.ops._triton import depthwise

    kernel = getattr(depthwise, name)
    pointers = [MockTensor(torch.float32)] * (2 if name == "_finish_dw_kernel" else 3)
    return kernel.warmup(*pointers, **constants, grid=(1,), num_warps=4).asm["ttir"]


@pytest.mark.parametrize(
    "name,constants",
    [
        (
            "_depthwise_input_grad_kernel",
            dict(N=64, C=32, D=128, H=128, W=128, KERNEL_SIZE=3, BLOCK=128, ACCUMULATE_FP64=False),
        ),
        (
            "_depthwise_stride2_input_grad_kernel",
            dict(
                N=64,
                C=32,
                D=128,
                H=128,
                W=128,
                OD=64,
                OH=64,
                OW=64,
                KERNEL_SIZE=3,
                BLOCK=128,
                ACCUMULATE_FP64=False,
            ),
        ),
        (
            "_partial_dw_kernel",
            dict(N=64, C=32, D=128, H=128, W=128, KERNEL_SIZE=3, SPLITS=64, BLOCK=128),
        ),
        # Only the transpose output exceeds the signed 32-bit element boundary.
        (
            "_partial_transpose_dw_kernel",
            dict(
                N=40,
                C=64,
                D=64,
                H=64,
                W=64,
                OD=127,
                OH=127,
                OW=127,
                KERNEL_SIZE=3,
                SPLITS=64,
                BLOCK=128,
                ACCUMULATE_FP64=False,
            ),
        ),
        # The reduction itself crosses 2**31, not only the channel-strided pointers.
        (
            "_partial_dw_kernel",
            dict(N=2**22, C=1, D=8, H=8, W=8, KERNEL_SIZE=3, SPLITS=64, BLOCK=128),
        ),
        ("_finish_dw_kernel", dict(SPLITS=64, BLOCK=64)),
    ],
)
def test_large_tensor_addresses_are_wide_before_multiplication(name, constants):
    ir = compile_kernel(name, **constants)
    pointers = [line for line in ir.splitlines() if "tt.addptr" in line]
    assert pointers
    assert ir.count("tt.load") >= (1 if name == "_finish_dw_kernel" else 2), ir
    assert all(re.search(r"(?:x|[, :] )i64[ >]", line) for line in pointers), pointers
    # A late cast after an i32 multiplication does not repair overflow.
    assert not re.search(r"arith.muli[^\n]*(?:x|: )i32[ >]", ir), ir


@pytest.mark.parametrize("batch, wide", [(31, False), (32, True), (33, True)])
def test_dx_index_width_at_signed_element_boundary(batch, wide):
    ir = compile_kernel(
        "_depthwise_input_grad_kernel",
        N=batch,
        C=32,
        D=128,
        H=128,
        W=128,
        KERNEL_SIZE=3,
        BLOCK=128,
        ACCUMULATE_FP64=False,
    )
    pointers = [line for line in ir.splitlines() if "tt.addptr" in line]
    assert pointers
    assert all(("xi64>" in line) == wide for line in pointers)
    assert ir.count("tt.load") == 2
    if wide:
        # Preserve the positive element-count bound, including the exact boundary.
        assert f"dense<{batch * 67108864}>" in ir


@pytest.mark.parametrize(
    "name",
    ["_partial_dw_kernel", "_partial_transpose_dw_kernel", "_depthwise_stride2_input_grad_kernel"],
)
def test_small_tensor_compute_kernels_keep_int32_indexing(name):
    constants = dict(N=2, C=32, D=16, H=16, W=16, KERNEL_SIZE=3, BLOCK=128)
    if "partial" in name:
        constants["SPLITS"] = 64
    if "transpose" in name:
        constants.update(OD=31, OH=31, OW=31)
    if "stride2" in name:
        constants.update(OD=8, OH=8, OW=8)
    if name != "_partial_dw_kernel":
        constants["ACCUMULATE_FP64"] = False
    ir = compile_kernel(name, **constants)
    pointers = [line for line in ir.splitlines() if "tt.addptr" in line]
    assert pointers
    assert all(re.search(r"(?:x|[, :] )i32[ >]", line) for line in pointers)
