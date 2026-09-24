"""Fused training GRN for contiguous CUDA NCDHW FP16/BF16 activations.

For spatial norms g and s = sum_channel(g) + eps, r = g / s and
Y = X + gamma * X * r + beta. All arithmetic is FP32; only Y and dX
are cast to the activation dtype at their final stores. Affine parameters
and their gradients remain FP32.

Spatial reductions use at most 64 independent splits. Only X, gamma,
(N, C) norms and (N,) denominators are saved for autograd; response maps
are never materialized. The backward includes the shared denominator's
cross-channel derivative, with an explicit zero subgradient at g == 0.
"""

import math

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _partial_norm_kernel(
    x,
    partial,
    N: tl.constexpr,
    C: tl.constexpr,
    SPATIAL: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Widen before multiplying program ids, including the masked split tails.
    index_dtype: tl.constexpr = (
        tl.int64 if N * C * (SPATIAL + SPLITS + BLOCK) >= 2**31 else tl.int32
    )
    program = tl.program_id(0).to(index_dtype)
    nc = program // SPLITS
    split = program % SPLITS
    chunk: tl.constexpr = triton.cdiv(SPATIAL, SPLITS)
    start = split * chunk
    end = tl.minimum(start + chunk, SPATIAL)
    offsets = start + tl.arange(0, BLOCK).to(index_dtype)
    accumulator = tl.zeros((BLOCK,), tl.float32)
    loop_begin = tl.full((), 0, index_dtype)
    loop_end = tl.full((), chunk, index_dtype)
    loop_step = tl.full((), BLOCK, index_dtype)
    for step in range(loop_begin, loop_end, loop_step):
        positions = offsets + step
        value = tl.load(x + nc * SPATIAL + positions, positions < end, 0).to(tl.float32)
        accumulator += value * value
    tl.store(partial + program, tl.sum(accumulator, 0))


@triton.jit
def _finish_norm_kernel(
    partial,
    norm,
    denominator,
    C: tl.constexpr,
    SPLITS: tl.constexpr,
    EPS: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
    SPLIT_BLOCK: tl.constexpr,
):
    # Completion buffers are small; use wide indices for arbitrary batch/channel counts.
    batch = tl.program_id(0).to(tl.int64)
    channels = tl.arange(0, CHANNEL_BLOCK).to(tl.int64)
    splits = tl.arange(0, SPLIT_BLOCK).to(tl.int64)
    nc = batch * C + channels
    squares = tl.load(
        partial + nc[:, None] * SPLITS + splits[None, :],
        (channels[:, None] < C) & (splits[None, :] < SPLITS),
        0,
    )
    norms = tl.sqrt(tl.sum(squares, 1))
    tl.store(norm + nc, norms, channels < C)
    tl.store(denominator + batch, tl.sum(norms, 0) + EPS)


@triton.jit
def _affine_kernel(
    x,
    gamma,
    beta,
    norm,
    denominator,
    output,
    N: tl.constexpr,
    C: tl.constexpr,
    SPATIAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index_dtype: tl.constexpr = tl.int64 if N * C * SPATIAL + BLOCK >= 2**31 else tl.int32
    offsets = tl.program_id(0).to(index_dtype) * BLOCK + tl.arange(0, BLOCK).to(index_dtype)
    mask = offsets < N * C * SPATIAL
    nc = offsets // SPATIAL
    channel = nc % C
    batch = nc // C
    value = tl.load(x + offsets, mask, 0).to(tl.float32)
    scale = tl.load(gamma + channel, mask, 0)
    shift = tl.load(beta + channel, mask, 0)
    g = tl.load(norm + nc, mask, 0)
    s = tl.load(denominator + batch, mask, 1)
    ratio = g / s
    # Preserve the reference's FP32 affine operation order and single final cast.
    result = value + scale * value * ratio + shift
    tl.store(output + offsets, result, mask)


@triton.jit
def _partial_backward_kernel(
    q,
    x,
    partial_qx,
    partial_q,
    N: tl.constexpr,
    C: tl.constexpr,
    SPATIAL: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index_dtype: tl.constexpr = (
        tl.int64 if N * C * (SPATIAL + SPLITS + BLOCK) >= 2**31 else tl.int32
    )
    program = tl.program_id(0).to(index_dtype)
    nc = program // SPLITS
    split = program % SPLITS
    chunk: tl.constexpr = triton.cdiv(SPATIAL, SPLITS)
    start = split * chunk
    end = tl.minimum(start + chunk, SPATIAL)
    offsets = start + tl.arange(0, BLOCK).to(index_dtype)
    qx_accumulator = tl.zeros((BLOCK,), tl.float32)
    q_accumulator = tl.zeros((BLOCK,), tl.float32)
    loop_begin = tl.full((), 0, index_dtype)
    loop_end = tl.full((), chunk, index_dtype)
    loop_step = tl.full((), BLOCK, index_dtype)
    for step in range(loop_begin, loop_end, loop_step):
        positions = offsets + step
        index = nc * SPATIAL + positions
        mask = positions < end
        value = tl.load(x + index, mask, 0).to(tl.float32)
        upstream = tl.load(q + index, mask, 0).to(tl.float32)
        qx_accumulator += upstream * value
        q_accumulator += upstream
    tl.store(partial_qx + program, tl.sum(qx_accumulator, 0))
    tl.store(partial_q + program, tl.sum(q_accumulator, 0))


@triton.jit
def _finish_backward_kernel(
    partial_qx,
    partial_q,
    gamma,
    norm,
    denominator,
    grad_norm,
    batch_dgamma,
    batch_dbeta,
    C: tl.constexpr,
    SPLITS: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
    SPLIT_BLOCK: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    channels = tl.arange(0, CHANNEL_BLOCK).to(tl.int64)
    splits = tl.arange(0, SPLIT_BLOCK).to(tl.int64)
    nc = batch * C + channels
    offsets = nc[:, None] * SPLITS + splits[None, :]
    mask = (channels[:, None] < C) & (splits[None, :] < SPLITS)
    qx = tl.sum(tl.load(partial_qx + offsets, mask, 0), 1)
    q = tl.sum(tl.load(partial_q + offsets, mask, 0), 1)
    scale = tl.load(gamma + channels, channels < C, 0)
    g = tl.load(norm + nc, channels < C, 0)
    s = tl.load(denominator + batch)
    ratio = g / s
    h = scale * qx
    coupled = tl.sum(h * ratio, 0)
    # dL/dg_c = (h_c - sum_j(h_j * r_j)) / s, including eps through s.
    tl.store(grad_norm + nc, (h - coupled) / s, channels < C)
    tl.store(batch_dgamma + nc, qx * ratio, channels < C)
    tl.store(batch_dbeta + nc, q, channels < C)


@triton.jit
def _input_backward_kernel(
    q,
    x,
    gamma,
    norm,
    denominator,
    grad_norm,
    grad_input,
    N: tl.constexpr,
    C: tl.constexpr,
    SPATIAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index_dtype: tl.constexpr = tl.int64 if N * C * SPATIAL + BLOCK >= 2**31 else tl.int32
    offsets = tl.program_id(0).to(index_dtype) * BLOCK + tl.arange(0, BLOCK).to(index_dtype)
    mask = offsets < N * C * SPATIAL
    nc = offsets // SPATIAL
    channel = nc % C
    batch = nc // C
    value = tl.load(x + offsets, mask, 0).to(tl.float32)
    upstream = tl.load(q + offsets, mask, 0).to(tl.float32)
    scale = tl.load(gamma + channel, mask, 0)
    g = tl.load(norm + nc, mask, 0)
    s = tl.load(denominator + batch, mask, 1)
    dg = tl.load(grad_norm + nc, mask, 0)
    # Avoid division by zero even in the unselected branch of tl.where.
    norm_derivative = tl.where(g > 0, value / tl.where(g > 0, g, 1.0), 0.0)
    result = upstream * (1.0 + scale * (g / s)) + norm_derivative * dg
    tl.store(grad_input + offsets, result, mask)


@triton.jit
def _affine_backward_kernel(
    batch_dgamma,
    batch_dbeta,
    grad_gamma,
    grad_beta,
    N: tl.constexpr,
    C: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
):
    channel = tl.program_id(0).to(tl.int64)
    batches = tl.arange(0, BATCH_BLOCK).to(tl.int64)
    offsets = batches * C + channel
    dgamma = tl.load(batch_dgamma + offsets, batches < N, 0)
    dbeta = tl.load(batch_dbeta + offsets, batches < N, 0)
    tl.store(grad_gamma + channel, tl.sum(dgamma, 0))
    tl.store(grad_beta + channel, tl.sum(dbeta, 0))


@triton_op("mednext_accel::grn_forward", mutates_args={})
def _grn_forward(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the activation and compact statistics consumed by autograd."""
    n, channels, depth, height, width = x.shape
    spatial = depth * height * width
    block = 1024
    splits = min(64, triton.cdiv(spatial, block))
    partial = torch.empty((n, channels, splits), device=x.device, dtype=torch.float32)
    norm = torch.empty((n, channels), device=x.device, dtype=torch.float32)
    denominator = torch.empty((n,), device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)
    wrap_triton(_partial_norm_kernel)[(n * channels * splits,)](
        x,
        partial,
        N=n,
        C=channels,
        SPATIAL=spatial,
        SPLITS=splits,
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    wrap_triton(_finish_norm_kernel)[(n,)](
        partial,
        norm,
        denominator,
        C=channels,
        SPLITS=splits,
        EPS=eps,
        CHANNEL_BLOCK=triton.next_power_of_2(channels),
        SPLIT_BLOCK=triton.next_power_of_2(splits),
        num_warps=4,
        enable_fp_fusion=False,
    )
    wrap_triton(_affine_kernel)[(triton.cdiv(x.numel(), block),)](
        x,
        gamma,
        beta,
        norm,
        denominator,
        output,
        N=n,
        C=channels,
        SPATIAL=spatial,
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, norm, denominator


@_grn_forward.register_fake
def _grn_forward_fake(x, gamma, beta, eps):
    return (
        torch.empty_like(x),
        torch.empty(x.shape[:2], device=x.device, dtype=torch.float32),
        torch.empty((x.shape[0],), device=x.device, dtype=torch.float32),
    )


@triton_op("mednext_accel::grn_backward", mutates_args={})
def _grn_backward(
    q: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
    norm: torch.Tensor,
    denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dX, dGamma and dBeta using FP32 split partials and channel coupling."""
    n, channels, depth, height, width = x.shape
    spatial = depth * height * width
    block = 1024
    splits = min(64, triton.cdiv(spatial, block))
    partial_qx = torch.empty((n, channels, splits), device=x.device, dtype=torch.float32)
    partial_q = torch.empty_like(partial_qx)
    grad_norm = torch.empty_like(norm)
    batch_dgamma = torch.empty_like(norm)
    batch_dbeta = torch.empty_like(norm)
    grad_input = torch.empty_like(x)
    grad_gamma = torch.empty_like(gamma)
    grad_beta = torch.empty_like(gamma)
    wrap_triton(_partial_backward_kernel)[(n * channels * splits,)](
        q,
        x,
        partial_qx,
        partial_q,
        N=n,
        C=channels,
        SPATIAL=spatial,
        SPLITS=splits,
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    wrap_triton(_finish_backward_kernel)[(n,)](
        partial_qx,
        partial_q,
        gamma,
        norm,
        denominator,
        grad_norm,
        batch_dgamma,
        batch_dbeta,
        C=channels,
        SPLITS=splits,
        CHANNEL_BLOCK=triton.next_power_of_2(channels),
        SPLIT_BLOCK=triton.next_power_of_2(splits),
        num_warps=4,
        enable_fp_fusion=False,
    )
    wrap_triton(_input_backward_kernel)[(triton.cdiv(x.numel(), block),)](
        q,
        x,
        gamma,
        norm,
        denominator,
        grad_norm,
        grad_input,
        N=n,
        C=channels,
        SPATIAL=spatial,
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    wrap_triton(_affine_backward_kernel)[(channels,)](
        batch_dgamma,
        batch_dbeta,
        grad_gamma,
        grad_beta,
        N=n,
        C=channels,
        BATCH_BLOCK=triton.next_power_of_2(n),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return grad_input, grad_gamma, grad_beta


@_grn_backward.register_fake
def _grn_backward_fake(q, x, gamma, norm, denominator):
    return torch.empty_like(x), torch.empty_like(gamma), torch.empty_like(gamma)


def _setup_context(ctx, inputs, output):
    x, gamma, _beta, _eps = inputs
    _activation, norm, denominator = output
    ctx.save_for_backward(x, gamma, norm, denominator)
    ctx.mark_non_differentiable(norm, denominator)
    ctx.set_materialize_grads(False)


def _backward(ctx, grad_output, _grad_norm, _grad_denominator):
    if grad_output is None:
        return None, None, None, None
    x, gamma, norm, denominator = ctx.saved_tensors
    gradients = _grn_backward(grad_output.contiguous(), x, gamma, norm, denominator)
    return tuple(
        g if needed else None for g, needed in zip(gradients, ctx.needs_input_grad[:3], strict=True)
    ) + (None,)


torch.library.register_autograd(_grn_forward, _backward, setup_context=_setup_context)


def fused_global_response_norm3d(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply fused GRN; reject unsupported direct calls instead of silently falling back.

    Inputs must be nonempty, contiguous CUDA NCDHW tensors in FP16 or BF16.
    Gamma and beta must be contiguous FP32 tensors of shape (1, C, 1, 1, 1)
    on the same device. The caller's epsilon must be finite and positive.
    """
    if x.ndim != 5:
        raise ValueError("fused GRN expects a rank-5 NCDHW input")
    if not (x.is_cuda and gamma.is_cuda and beta.is_cuda):
        raise ValueError("fused GRN requires CUDA tensors")
    if gamma.device != x.device or beta.device != x.device:
        raise ValueError("fused GRN requires all tensors on the same CUDA device")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("fused GRN requires a float16 or bfloat16 input")
    if not x.is_contiguous():
        raise ValueError("fused GRN requires a contiguous NCDHW input")
    if any(size == 0 for size in x.shape):
        raise ValueError("fused GRN requires nonempty input dimensions")
    if gamma.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("fused GRN requires float32 gamma and beta")
    expected_shape = (1, x.shape[1], 1, 1, 1)
    if gamma.shape != expected_shape or beta.shape != expected_shape:
        raise ValueError("fused GRN requires gamma and beta with shape (1, C, 1, 1, 1)")
    if not gamma.is_contiguous() or not beta.is_contiguous():
        raise ValueError("fused GRN requires contiguous gamma and beta")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("fused GRN requires eps to be finite and positive")
    return _grn_forward(x, gamma, beta, float(eps))[0]
