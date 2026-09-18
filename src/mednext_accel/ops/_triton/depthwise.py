"""Triton dW and dX paths for selected odd depthwise Conv3d shapes.

The optimized paths support contiguous CUDA tensors with fixed same-padding
stride-one, stride-two, or transpose geometries. PyTorch computes bias gradients.
"""

import torch
import triton
import triton.language as tl
from torch.library import custom_op, triton_op, wrap_triton


@triton.jit
def _partial_dw_kernel(
    x,
    grad_output,
    partial,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    weight_index = tl.program_id(0)
    split = tl.program_id(1)
    kernel_plane = KERNEL_SIZE * KERNEL_SIZE
    kernel_volume = kernel_plane * KERNEL_SIZE
    channel = weight_index // kernel_volume
    offset = weight_index % kernel_volume
    kd = offset // kernel_plane
    kh = (offset % kernel_plane) // KERNEL_SIZE
    kw = offset % KERNEL_SIZE
    padding = KERNEL_SIZE // 2

    spatial = D * H * W
    total = N * spatial
    chunk = tl.cdiv(total, SPLITS)
    split_start = split * chunk
    split_end = tl.minimum(split_start + chunk, total)
    positions = split_start + tl.arange(0, BLOCK)
    accumulator = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, chunk, BLOCK):
        p = positions + start
        valid_output = p < split_end
        n = p // spatial
        rem = p % spatial
        od = rem // (H * W)
        rem = rem % (H * W)
        oh = rem // W
        ow = rem % W
        id_ = od + kd - padding
        ih = oh + kh - padding
        iw = ow + kw - padding
        valid_input = (id_ >= 0) & (id_ < D) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        input_index = ((n * C + channel) * D + id_) * H * W + ih * W + iw
        output_index = ((n * C + channel) * D + od) * H * W + oh * W + ow
        xv = tl.load(x + input_index, mask=valid_output & valid_input, other=0.0)
        gv = tl.load(grad_output + output_index, mask=valid_output, other=0.0)
        accumulator += xv.to(tl.float32) * gv.to(tl.float32)
    tl.store(partial + weight_index * SPLITS + split, tl.sum(accumulator, axis=0))


@triton.jit
def _finish_dw_kernel(partial, output, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    weight_index = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(partial + weight_index * SPLITS + offsets, mask=offsets < SPLITS, other=0.0)
    tl.store(output + weight_index, tl.sum(values, axis=0))


@triton.jit
def _partial_transpose_dw_kernel(
    x,
    grad_output,
    partial,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    ACCUMULATE_FP64: tl.constexpr,
):
    weight_index = tl.program_id(0)
    split = tl.program_id(1)
    kernel_plane = KERNEL_SIZE * KERNEL_SIZE
    kernel_volume = kernel_plane * KERNEL_SIZE
    channel = weight_index // kernel_volume
    offset = weight_index % kernel_volume
    kd = offset // kernel_plane
    kh = (offset % kernel_plane) // KERNEL_SIZE
    kw = offset % KERNEL_SIZE
    padding = KERNEL_SIZE // 2

    spatial = D * H * W
    total = N * spatial
    chunk = tl.cdiv(total, SPLITS)
    split_start = split * chunk
    split_end = tl.minimum(split_start + chunk, total)
    positions = split_start + tl.arange(0, BLOCK)
    accumulator = tl.zeros((BLOCK,), tl.float64 if ACCUMULATE_FP64 else tl.float32)
    for start in range(0, chunk, BLOCK):
        p = positions + start
        valid_input = p < split_end
        n = p // spatial
        rem = p % spatial
        id_ = rem // (H * W)
        rem = rem % (H * W)
        ih = rem // W
        iw = rem % W
        od = id_ * 2 - padding + kd
        oh = ih * 2 - padding + kh
        ow = iw * 2 - padding + kw
        valid_output = (od >= 0) & (od < OD) & (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW)
        input_index = ((n * C + channel) * D + id_) * H * W + ih * W + iw
        output_index = ((n * C + channel) * OD + od) * OH * OW + oh * OW + ow
        xv = tl.load(x + input_index, mask=valid_input, other=0.0)
        gv = tl.load(grad_output + output_index, mask=valid_input & valid_output, other=0.0)
        accumulator += xv.to(accumulator.dtype) * gv.to(accumulator.dtype)
    tl.store(partial + weight_index * SPLITS + split, tl.sum(accumulator, axis=0))


@triton.jit
def _depthwise_input_grad_kernel(
    grad_output,
    weight,
    grad_input,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    ACCUMULATE_FP64: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    spatial = D * H * W
    total = N * C * spatial
    valid_output = offsets < total
    n = offsets // (C * spatial)
    rem = offsets % (C * spatial)
    channel = rem // spatial
    rem = rem % spatial
    id_ = rem // (H * W)
    rem = rem % (H * W)
    ih = rem // W
    iw = rem % W
    padding = KERNEL_SIZE // 2
    accumulator = tl.zeros((BLOCK,), tl.float64 if ACCUMULATE_FP64 else tl.float32)
    for kd in range(KERNEL_SIZE):
        for kh in range(KERNEL_SIZE):
            for kw in range(KERNEL_SIZE):
                od = id_ - kd + padding
                oh = ih - kh + padding
                ow = iw - kw + padding
                valid_grad = (od >= 0) & (od < D) & (oh >= 0) & (oh < H) & (ow >= 0) & (ow < W)
                grad_index = ((n * C + channel) * D + od) * H * W + oh * W + ow
                weight_index = ((channel * KERNEL_SIZE + kd) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw
                gv = tl.load(grad_output + grad_index, mask=valid_output & valid_grad, other=0.0)
                wv = tl.load(weight + weight_index, mask=valid_output, other=0.0)
                accumulator += gv.to(accumulator.dtype) * wv.to(accumulator.dtype)
    tl.store(grad_input + offsets, accumulator, mask=valid_output)


@triton.jit
def _depthwise_stride2_input_grad_kernel(
    grad_output,
    weight,
    grad_input,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    ACCUMULATE_FP64: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    spatial = D * H * W
    total = N * C * spatial
    valid_input = offsets < total
    n = offsets // (C * spatial)
    rem = offsets % (C * spatial)
    channel = rem // spatial
    rem = rem % spatial
    id_ = rem // (H * W)
    rem = rem % (H * W)
    ih = rem // W
    iw = rem % W
    padding = KERNEL_SIZE // 2
    accumulator = tl.zeros((BLOCK,), tl.float64 if ACCUMULATE_FP64 else tl.float32)
    for kd in range(KERNEL_SIZE):
        for kh in range(KERNEL_SIZE):
            for kw in range(KERNEL_SIZE):
                od_numerator = id_ + padding - kd
                oh_numerator = ih + padding - kh
                ow_numerator = iw + padding - kw
                divisible = (
                    (od_numerator % 2 == 0) & (oh_numerator % 2 == 0) & (ow_numerator % 2 == 0)
                )
                od = od_numerator // 2
                oh = oh_numerator // 2
                ow = ow_numerator // 2
                valid_grad = (
                    divisible
                    & (od >= 0)
                    & (od < OD)
                    & (oh >= 0)
                    & (oh < OH)
                    & (ow >= 0)
                    & (ow < OW)
                )
                grad_index = ((n * C + channel) * OD + od) * OH * OW + oh * OW + ow
                weight_index = ((channel * KERNEL_SIZE + kd) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw
                gv = tl.load(grad_output + grad_index, mask=valid_input & valid_grad, other=0.0)
                wv = tl.load(weight + weight_index, mask=valid_input, other=0.0)
                accumulator += gv.to(accumulator.dtype) * wv.to(accumulator.dtype)
    tl.store(grad_input + offsets, accumulator, mask=valid_input)


def depthwise_input_grad(grad_output, weight, block=256):
    """Return stride-one, same-padding depthwise dX for cubic odd kernels."""
    if not (grad_output.is_cuda and weight.is_cuda):
        raise ValueError("CUDA tensors are required")
    if grad_output.ndim != 5 or weight.ndim != 5:
        raise ValueError("grad_output and weight must be NCDHW and C1KKK tensors")
    if not (grad_output.is_contiguous() and weight.is_contiguous()):
        raise ValueError("contiguous tensors are required")
    n, channels, depth, height, width = grad_output.shape
    if weight.shape[0] != channels or weight.shape[1] != 1 or len(set(weight.shape[2:])) != 1:
        raise ValueError("weight must have shape [C, 1, K, K, K]")
    kernel_size = weight.shape[2]
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if block < 1 or triton.next_power_of_2(block) != block:
        raise ValueError("block must be a positive power of two")
    output = torch.empty_like(grad_output)
    _depthwise_input_grad_kernel[(triton.cdiv(grad_output.numel(), block),)](
        grad_output,
        weight,
        output,
        N=n,
        C=channels,
        D=depth,
        H=height,
        W=width,
        KERNEL_SIZE=kernel_size,
        BLOCK=block,
        ACCUMULATE_FP64=grad_output.dtype == torch.float64,
        num_warps=4,
    )
    return output


def depthwise_stride2_input_grad(grad_output, weight, input_spatial_size, block=256):
    """Return stride-two, same-padding depthwise Conv3d dX."""
    if not (grad_output.is_cuda and weight.is_cuda):
        raise ValueError("CUDA tensors are required")
    if grad_output.ndim != 5 or weight.ndim != 5:
        raise ValueError("grad_output and weight must be NCDHW and C1KKK tensors")
    if not (grad_output.is_contiguous() and weight.is_contiguous()):
        raise ValueError("contiguous tensors are required")
    if len(input_spatial_size) != 3:
        raise ValueError("input_spatial_size must contain D, H, and W")
    n, channels, out_depth, out_height, out_width = grad_output.shape
    if weight.shape[0] != channels or weight.shape[1] != 1 or len(set(weight.shape[2:])) != 1:
        raise ValueError("weight must have shape [C, 1, K, K, K]")
    kernel_size = weight.shape[2]
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if block < 1 or triton.next_power_of_2(block) != block:
        raise ValueError("block must be a positive power of two")
    depth, height, width = (int(size) for size in input_spatial_size)
    expected = tuple((size + 1) // 2 for size in (depth, height, width))
    if (out_depth, out_height, out_width) != expected:
        raise ValueError("grad_output shape does not match stride-two same padding")
    output = torch.empty(
        (n, channels, depth, height, width), device=grad_output.device, dtype=grad_output.dtype
    )
    total = output.numel()
    _depthwise_stride2_input_grad_kernel[(triton.cdiv(total, block),)](
        grad_output,
        weight,
        output,
        N=n,
        C=channels,
        D=depth,
        H=height,
        W=width,
        OD=out_depth,
        OH=out_height,
        OW=out_width,
        KERNEL_SIZE=kernel_size,
        BLOCK=block,
        ACCUMULATE_FP64=grad_output.dtype == torch.float64,
        num_warps=4,
    )
    return output


def depthwise_weight_grad(x, grad_output, splits=64, block=512, kernel_size=3):
    """Return cubic odd-kernel dW using FP32 partial reductions."""
    if not (x.is_cuda and grad_output.is_cuda):
        raise ValueError("CUDA tensors are required")
    if x.shape != grad_output.shape or x.ndim != 5:
        raise ValueError("x and grad_output must have the same NCDHW shape")
    if not (x.is_contiguous() and grad_output.is_contiguous()):
        raise ValueError("contiguous NCDHW tensors are required")
    if splits < 1 or block < 1 or not triton.next_power_of_2(block) == block:
        raise ValueError("splits must be positive and block must be a power of two")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    n, channels, depth, height, width = x.shape
    kernel_volume = kernel_size**3
    weight_elements = channels * kernel_volume
    partial = torch.empty((weight_elements, splits), device=x.device, dtype=torch.float32)
    output = torch.empty((weight_elements,), device=x.device, dtype=x.dtype)
    _partial_dw_kernel[(weight_elements, splits)](
        x,
        grad_output,
        partial,
        N=n,
        C=channels,
        D=depth,
        H=height,
        W=width,
        KERNEL_SIZE=kernel_size,
        SPLITS=splits,
        BLOCK=block,
        num_warps=8,
    )
    finish_block = triton.next_power_of_2(splits)
    _finish_dw_kernel[(weight_elements,)](
        partial,
        output,
        SPLITS=splits,
        BLOCK=finish_block,
        num_warps=min(8, max(1, finish_block // 32)),
    )
    return output.reshape(channels, 1, kernel_size, kernel_size, kernel_size)


def depthwise_transpose_weight_grad(x, grad_output, splits=64, block=512, kernel_size=3):
    """Return stride-two, same-padding depthwise ConvTranspose3d dW."""
    if not (x.is_cuda and grad_output.is_cuda):
        raise ValueError("CUDA tensors are required")
    if x.ndim != 5 or grad_output.ndim != 5:
        raise ValueError("x and grad_output must be NCDHW tensors")
    if x.shape[:2] != grad_output.shape[:2]:
        raise ValueError("x and grad_output must have matching batch and channels")
    if not (x.is_contiguous() and grad_output.is_contiguous()):
        raise ValueError("contiguous NCDHW tensors are required")
    if splits < 1 or block < 1 or triton.next_power_of_2(block) != block:
        raise ValueError("splits must be positive and block must be a power of two")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    n, channels, depth, height, width = x.shape
    expected_output = tuple(2 * size - 1 for size in (depth, height, width))
    if tuple(grad_output.shape[2:]) != expected_output:
        raise ValueError("grad_output must have spatial shape 2 * x.shape[2:] - 1")
    kernel_volume = kernel_size**3
    weight_elements = channels * kernel_volume
    partial_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    partial = torch.empty((weight_elements, splits), device=x.device, dtype=partial_dtype)
    output = torch.empty((weight_elements,), device=x.device, dtype=x.dtype)
    _partial_transpose_dw_kernel[(weight_elements, splits)](
        x,
        grad_output,
        partial,
        N=n,
        C=channels,
        D=depth,
        H=height,
        W=width,
        OD=grad_output.shape[2],
        OH=grad_output.shape[3],
        OW=grad_output.shape[4],
        KERNEL_SIZE=kernel_size,
        SPLITS=splits,
        BLOCK=block,
        ACCUMULATE_FP64=x.dtype == torch.float64,
        num_warps=8,
    )
    finish_block = triton.next_power_of_2(splits)
    _finish_dw_kernel[(weight_elements,)](
        partial,
        output,
        SPLITS=splits,
        BLOCK=finish_block,
        num_warps=min(8, max(1, finish_block // 32)),
    )
    return output.reshape(channels, 1, kernel_size, kernel_size, kernel_size)


@triton_op("mednext_accel::depthwise_weight_grad_odd_regular", mutates_args={})
def depthwise_weight_grad_regular(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    kernel_size: int,
    channels: int,
    spatial_size: int,
    splits: int,
    block: int,
) -> torch.Tensor:
    """Compiled dW for one statically configured regular depthwise shape."""
    kernel_volume = kernel_size**3
    weight_elements = channels * kernel_volume
    partial = torch.empty((weight_elements, splits), device=x.device, dtype=torch.float32)
    output = torch.empty((weight_elements,), device=x.device, dtype=x.dtype)
    wrap_triton(_partial_dw_kernel)[(weight_elements, splits)](
        x,
        grad_output,
        partial,
        N=1,
        C=channels,
        D=spatial_size,
        H=spatial_size,
        W=spatial_size,
        KERNEL_SIZE=kernel_size,
        SPLITS=splits,
        BLOCK=block,
        num_warps=8,
    )
    finish_block = triton.next_power_of_2(splits)
    wrap_triton(_finish_dw_kernel)[(weight_elements,)](
        partial,
        output,
        SPLITS=splits,
        BLOCK=finish_block,
        num_warps=min(8, max(1, finish_block // 32)),
    )
    return output.reshape(channels, 1, kernel_size, kernel_size, kernel_size)


@triton_op("mednext_accel::depthwise_input_grad_odd_regular", mutates_args={})
def depthwise_input_grad_regular(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    kernel_size: int,
    channels: int,
    spatial_size: int,
    block: int,
) -> torch.Tensor:
    """Compiled dX for one statically configured regular depthwise shape."""
    output = torch.empty_like(grad_output)
    total = channels * spatial_size**3
    wrap_triton(_depthwise_input_grad_kernel)[(triton.cdiv(total, block),)](
        grad_output,
        weight,
        output,
        N=1,
        C=channels,
        D=spatial_size,
        H=spatial_size,
        W=spatial_size,
        KERNEL_SIZE=kernel_size,
        BLOCK=block,
        ACCUMULATE_FP64=False,
        num_warps=4,
    )
    return output


@triton_op("mednext_accel::depthwise_input_grad_stride2_regular", mutates_args={})
def depthwise_input_grad_stride2_regular(
    grad_output: torch.Tensor, weight: torch.Tensor, channels: int, spatial_size: int, block: int
) -> torch.Tensor:
    """Compiled dX for one statically configured stride-two depthwise shape."""
    output = torch.empty(
        (1, channels, spatial_size, spatial_size, spatial_size),
        device=grad_output.device,
        dtype=grad_output.dtype,
    )
    output_size = (spatial_size + 1) // 2
    total = channels * spatial_size**3
    wrap_triton(_depthwise_stride2_input_grad_kernel)[(triton.cdiv(total, block),)](
        grad_output,
        weight,
        output,
        N=1,
        C=channels,
        D=spatial_size,
        H=spatial_size,
        W=spatial_size,
        OD=output_size,
        OH=output_size,
        OW=output_size,
        KERNEL_SIZE=weight.shape[2],
        BLOCK=block,
        ACCUMULATE_FP64=False,
        num_warps=4,
    )
    return output


@custom_op("mednext_accel::depthwise_conv3d_odd_regular", mutates_args=())
def depthwise_conv3d_regular(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    kernel_size: int,
    channels: int,
    spatial_size: int,
    dw_splits: int,
    dw_block: int,
    dx_block: int,
) -> torch.Tensor:
    return torch.nn.functional.conv3d(x, weight, bias, padding=kernel_size // 2, groups=channels)


@depthwise_conv3d_regular.register_fake
def _(x, weight, bias, kernel_size, channels, spatial_size, dw_splits, dw_block, dx_block):
    return torch.empty_like(x)


def _setup_regular_context(ctx, inputs, output):
    x, weight, bias, kernel_size, channels, spatial_size, dw_splits, dw_block, dx_block = inputs
    ctx.save_for_backward(x, weight)
    ctx.has_bias = bias is not None
    ctx.kernel_size = kernel_size
    ctx.channels = channels
    ctx.spatial_size = spatial_size
    ctx.dw_splits = dw_splits
    ctx.dw_block = dw_block
    ctx.dx_block = dx_block


def _regular_backward(ctx, grad_output):
    x, weight = ctx.saved_tensors
    grad_output = grad_output.contiguous()
    grad_input = depthwise_input_grad_regular(
        grad_output, weight, ctx.kernel_size, ctx.channels, ctx.spatial_size, ctx.dx_block
    )
    grad_weight = depthwise_weight_grad_regular(
        x, grad_output, ctx.kernel_size, ctx.channels, ctx.spatial_size, ctx.dw_splits, ctx.dw_block
    )
    grad_bias = None
    if ctx.has_bias:
        padding = ctx.kernel_size // 2
        _, _, grad_bias = torch.ops.aten.convolution_backward(
            grad_output,
            x,
            weight,
            [ctx.channels],
            [1, 1, 1],
            [padding, padding, padding],
            [1, 1, 1],
            False,
            [0, 0, 0],
            ctx.channels,
            [False, False, True],
        )
    return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


depthwise_conv3d_regular.register_autograd(_regular_backward, setup_context=_setup_regular_context)


@custom_op("mednext_accel::depthwise_conv3d_stride2_regular", mutates_args=())
def depthwise_conv3d_stride2_regular(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    kernel_size: int,
    channels: int,
    spatial_size: int,
    dx_block: int,
) -> torch.Tensor:
    return torch.nn.functional.conv3d(
        x, weight, bias, stride=2, padding=kernel_size // 2, groups=channels
    )


@depthwise_conv3d_stride2_regular.register_fake
def _(x, weight, bias, kernel_size, channels, spatial_size, dx_block):
    output_size = (spatial_size + 1) // 2
    return torch.empty(
        (x.shape[0], channels, output_size, output_size, output_size),
        device=x.device,
        dtype=x.dtype,
    )


def _setup_stride2_context(ctx, inputs, output):
    x, weight, bias, kernel_size, channels, spatial_size, dx_block = inputs
    ctx.save_for_backward(x, weight)
    ctx.has_bias = bias is not None
    ctx.kernel_size = kernel_size
    ctx.channels = channels
    ctx.spatial_size = spatial_size
    ctx.dx_block = dx_block


def _stride2_backward(ctx, grad_output):
    x, weight = ctx.saved_tensors
    grad_output = grad_output.contiguous()
    grad_input = depthwise_input_grad_stride2_regular(
        grad_output, weight, ctx.channels, ctx.spatial_size, ctx.dx_block
    )
    padding = ctx.kernel_size // 2
    _, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
        grad_output,
        x,
        weight,
        [ctx.channels] if ctx.has_bias else None,
        [2, 2, 2],
        [padding, padding, padding],
        [1, 1, 1],
        False,
        [0, 0, 0],
        ctx.channels,
        [False, True, ctx.has_bias],
    )
    return grad_input, grad_weight, grad_bias if ctx.has_bias else None, None, None, None, None


depthwise_conv3d_stride2_regular.register_autograd(
    _stride2_backward, setup_context=_setup_stride2_context
)


@triton_op("mednext_accel::depthwise_transpose_weight_grad_odd_regular", mutates_args={})
def depthwise_transpose_weight_grad_regular(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    kernel_size: int,
    channels: int,
    spatial_size: int,
    splits: int,
    block: int,
) -> torch.Tensor:
    """Compiled dW for one configured stride-two depthwise transpose shape."""
    kernel_volume = kernel_size**3
    weight_elements = channels * kernel_volume
    partial = torch.empty((weight_elements, splits), device=x.device, dtype=torch.float32)
    output = torch.empty((weight_elements,), device=x.device, dtype=x.dtype)
    output_size = 2 * spatial_size - 1
    wrap_triton(_partial_transpose_dw_kernel)[(weight_elements, splits)](
        x,
        grad_output,
        partial,
        N=1,
        C=channels,
        D=spatial_size,
        H=spatial_size,
        W=spatial_size,
        OD=output_size,
        OH=output_size,
        OW=output_size,
        KERNEL_SIZE=kernel_size,
        SPLITS=splits,
        BLOCK=block,
        ACCUMULATE_FP64=False,
        num_warps=8,
    )
    finish_block = triton.next_power_of_2(splits)
    wrap_triton(_finish_dw_kernel)[(weight_elements,)](
        partial,
        output,
        SPLITS=splits,
        BLOCK=finish_block,
        num_warps=min(8, max(1, finish_block // 32)),
    )
    return output.reshape(channels, 1, kernel_size, kernel_size, kernel_size)


@custom_op("mednext_accel::depthwise_conv_transpose3d_odd_regular", mutates_args=())
def depthwise_conv_transpose3d_regular(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    kernel_size: int,
    channels: int,
    spatial_size: int,
    dw_splits: int,
    dw_block: int,
) -> torch.Tensor:
    return torch.nn.functional.conv_transpose3d(
        x, weight, bias, stride=2, padding=kernel_size // 2, groups=channels
    )


@depthwise_conv_transpose3d_regular.register_fake
def _(x, weight, bias, kernel_size, channels, spatial_size, dw_splits, dw_block):
    output_size = 2 * spatial_size - 1
    return torch.empty(
        (x.shape[0], channels, output_size, output_size, output_size),
        device=x.device,
        dtype=x.dtype,
    )


def _setup_transpose_regular_context(ctx, inputs, output):
    x, weight, bias, kernel_size, channels, spatial_size, dw_splits, dw_block = inputs
    ctx.save_for_backward(x, weight)
    ctx.has_bias = bias is not None
    ctx.kernel_size = kernel_size
    ctx.channels = channels
    ctx.spatial_size = spatial_size
    ctx.dw_splits = dw_splits
    ctx.dw_block = dw_block


def _transpose_regular_backward(ctx, grad_output):
    x, weight = ctx.saved_tensors
    grad_output = grad_output.contiguous()
    padding = ctx.kernel_size // 2
    grad_input, _, grad_bias = torch.ops.aten.convolution_backward(
        grad_output,
        x,
        weight,
        [ctx.channels] if ctx.has_bias else None,
        [2, 2, 2],
        [padding, padding, padding],
        [1, 1, 1],
        True,
        [0, 0, 0],
        ctx.channels,
        [True, False, ctx.has_bias],
    )
    grad_weight = depthwise_transpose_weight_grad_regular(
        x, grad_output, ctx.kernel_size, ctx.channels, ctx.spatial_size, ctx.dw_splits, ctx.dw_block
    )
    return (
        grad_input,
        grad_weight,
        grad_bias if ctx.has_bias else None,
        None,
        None,
        None,
        None,
        None,
    )


depthwise_conv_transpose3d_regular.register_autograd(
    _transpose_regular_backward, setup_context=_setup_transpose_regular_context
)


_REGULAR_CONFIGS = {
    (32, 128): (64, 1024, 128),
    (64, 64): (8, 1024, 128),
    (128, 32): (2, 1024, 128),
    (256, 16): (1, 512, 128),
    (512, 8): (1, 128, 128),
}
_REGULAR_CHANNELS = frozenset(channels for channels, _ in _REGULAR_CONFIGS)
_TRANSPOSE_CONFIGS = {(64, 64): (64, 1024)}
_TRANSPOSE_CHANNELS = frozenset(channels for channels, _ in _TRANSPOSE_CONFIGS)
_DOWNSAMPLE_CONFIGS = {
    (32, 128): 128,
    (64, 64): 128,
    (128, 32): 128,
    (256, 16): 128,
}
_DOWNSAMPLE_CHANNELS = frozenset(channels for channels, _ in _DOWNSAMPLE_CONFIGS)


class SplitDepthwiseConv3d(torch.nn.Module):
    """Drop-in module for tuned regular depthwise backward configurations."""

    _mednext_accel_backend_kind = "depthwise_regular"

    def __init__(self, conv, selected_shapes=None):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.groups = conv.groups
        self.kernel_size = conv.kernel_size[0]
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.selected_shapes = None if selected_shapes is None else frozenset(selected_shapes)
        self.train(conv.training)

    def forward(self, x):
        config = _REGULAR_CONFIGS.get((x.shape[1], x.shape[2]))
        if (
            self.training
            and torch.is_grad_enabled()
            and config is not None
            and (self.selected_shapes is None or (x.shape[1], x.shape[2]) in self.selected_shapes)
            and x.shape == (1, x.shape[1], x.shape[2], x.shape[2], x.shape[2])
            and x.is_contiguous()
        ):
            # Custom operators are opaque to autocast. Match Conv3d's AMP
            # boundary explicitly; autograd carries gradients through casts
            # back to the FP32 master parameters.
            weight = self.weight.to(x.dtype)
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            dw_splits, dw_block, dx_block = config
            return depthwise_conv3d_regular(
                x,
                weight,
                bias,
                self.kernel_size,
                x.shape[1],
                x.shape[2],
                dw_splits,
                dw_block,
                dx_block,
            )
        return torch.nn.functional.conv3d(
            x, self.weight, self.bias, padding=self.kernel_size // 2, groups=x.shape[1]
        )


class SplitDepthwiseConvTranspose3d(torch.nn.Module):
    """Drop-in module for tuned stride-two depthwise transpose dW."""

    _mednext_accel_backend_kind = "depthwise_transpose"

    def __init__(self, conv, selected_shapes=None):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size[0]
        self.stride = conv.stride
        self.padding = conv.padding
        self.output_padding = conv.output_padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.selected_shapes = None if selected_shapes is None else frozenset(selected_shapes)
        self.train(conv.training)

    def forward(self, x):
        config = _TRANSPOSE_CONFIGS.get((x.shape[1], x.shape[2]))
        if (
            self.training
            and torch.is_grad_enabled()
            and config is not None
            and (self.selected_shapes is None or (x.shape[1], x.shape[2]) in self.selected_shapes)
            and x.shape == (1, x.shape[1], x.shape[2], x.shape[2], x.shape[2])
            and x.is_contiguous()
        ):
            weight = self.weight.to(x.dtype)
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            dw_splits, dw_block = config
            return depthwise_conv_transpose3d_regular(
                x, weight, bias, self.kernel_size, x.shape[1], x.shape[2], dw_splits, dw_block
            )
        return torch.nn.functional.conv_transpose3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


class SplitDepthwiseDownsampleConv3d(torch.nn.Module):
    """Drop-in module for a measured stride-two depthwise dX prototype."""

    _mednext_accel_backend_kind = "depthwise_downsample"

    def __init__(self, conv, selected_shapes=None):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size[0]
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        self.selected_shapes = None if selected_shapes is None else frozenset(selected_shapes)
        self.train(conv.training)

    def forward(self, x):
        dx_block = _DOWNSAMPLE_CONFIGS.get((x.shape[1], x.shape[2]))
        if (
            self.training
            and torch.is_grad_enabled()
            and dx_block is not None
            and (self.selected_shapes is None or (x.shape[1], x.shape[2]) in self.selected_shapes)
            and x.shape == (1, x.shape[1], x.shape[2], x.shape[2], x.shape[2])
            and x.is_contiguous()
        ):
            weight = self.weight.to(x.dtype)
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            return depthwise_conv3d_stride2_regular(
                x, weight, bias, self.kernel_size, x.shape[1], x.shape[2], dx_block
            )
        return torch.nn.functional.conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


def replace_highres_depthwise_convs(model, include_stride2_input_grad=False, policy=None):
    """Replace eligible children; stride-two dX is opt-in pending integration tuning."""
    count = 0
    policy = policy or {}
    regular_shapes = policy.get("depthwise_regular")
    transpose_shapes = policy.get("depthwise_transpose")
    downsample_shapes = policy.get("depthwise_downsample")
    for name, child in list(model.named_children()):
        is_conv3d = type(child) is torch.nn.Conv3d
        is_transpose3d = type(child) is torch.nn.ConvTranspose3d
        cubic = len(set(child.kernel_size)) == 1 if (is_conv3d or is_transpose3d) else False
        kernel_size = child.kernel_size[0] if cubic else 0
        regular = (
            is_conv3d
            and child.in_channels in _REGULAR_CHANNELS
            and child.out_channels == child.in_channels
            and child.groups == child.in_channels
            and cubic
            and kernel_size % 2 == 1
            and child.stride == (1, 1, 1)
            and child.padding == (kernel_size // 2,) * 3
            and child.dilation == (1, 1, 1)
            and child.padding_mode == "zeros"
        )
        transpose = (
            is_transpose3d
            and child.in_channels in _TRANSPOSE_CHANNELS
            and child.out_channels == child.in_channels
            and child.groups == child.in_channels
            and cubic
            and kernel_size % 2 == 1
            and child.stride == (2, 2, 2)
            and child.padding == (kernel_size // 2,) * 3
            and child.output_padding == (0, 0, 0)
            and child.dilation == (1, 1, 1)
        )
        downsample = (
            is_conv3d
            and child.in_channels in _DOWNSAMPLE_CHANNELS
            and child.out_channels == child.in_channels
            and child.groups == child.in_channels
            and cubic
            and kernel_size % 2 == 1
            and child.stride == (2, 2, 2)
            and child.padding == (kernel_size // 2,) * 3
            and child.dilation == (1, 1, 1)
            and child.padding_mode == "zeros"
        )
        if regular:
            setattr(model, name, SplitDepthwiseConv3d(child, regular_shapes))
            count += 1
        elif transpose:
            setattr(model, name, SplitDepthwiseConvTranspose3d(child, transpose_shapes))
            count += 1
        elif include_stride2_input_grad and downsample:
            setattr(model, name, SplitDepthwiseDownsampleConv3d(child, downsample_shapes))
            count += 1
        else:
            count += replace_highres_depthwise_convs(child, include_stride2_input_grad, policy)
    return count
