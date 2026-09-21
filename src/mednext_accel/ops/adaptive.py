"""Profile-driven wrappers that preserve native module parameters and keys."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..optimization.descriptors import ExecutionContext, OperatorDescriptor
from ..optimization.report import Decision
from ..optimization.resolver import OptimizationResolver


@dataclass(frozen=True, slots=True)
class ModelOptimizationContext:
    family: str
    variant: str
    checkpointing: str
    allow_approximate: bool = False


def _dtype_name(dtype: torch.dtype | str) -> str:
    return str(dtype).removeprefix("torch.")


def execution_context_for_shape(
    model_context: ModelOptimizationContext,
    *,
    batch_size: int,
    spatial_shape: tuple[int, int, int],
    dtype: torch.dtype | str,
    device_type: str,
    sm: tuple[int, int] | None,
    total_vram_bytes: int,
    training: bool = True,
    export: bool = False,
) -> ExecutionContext:
    return ExecutionContext(
        phase="export" if export else ("training" if training else "inference"),
        device_type=device_type,
        sm=sm,
        total_vram_bytes=total_vram_bytes,
        dtype=_dtype_name(dtype),
        batch_size=batch_size,
        spatial_shape=spatial_shape,
        model_family=model_context.family,
        variant=model_context.variant,
        checkpointing=model_context.checkpointing,
        export=export,
        allow_approximate=model_context.allow_approximate,
    )


def _context_from_tensor(
    x: Tensor, model_context: ModelOptimizationContext, *, training: bool
) -> ExecutionContext:
    export = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
    sm = None
    total_vram = 0
    if x.device.type == "cuda" and not export:
        sm = torch.cuda.get_device_capability(x.device)
        total_vram = torch.cuda.get_device_properties(x.device).total_memory
    batch = 1 if x.ndim == 4 else int(x.shape[0])
    return execution_context_for_shape(
        model_context,
        batch_size=batch,
        spatial_shape=tuple(int(size) for size in x.shape[-3:]),
        dtype=x.dtype,
        device_type=x.device.type,
        sm=sm,
        total_vram_bytes=total_vram,
        training=training,
        export=export,
    )


def _conv_descriptor(
    conv: nn.Conv3d | nn.ConvTranspose3d, *, role: str | None
) -> OperatorDescriptor:
    transpose = isinstance(conv, nn.ConvTranspose3d)
    depthwise = conv.groups == conv.in_channels == conv.out_channels
    family = (
        "depthwise_conv_transpose3d"
        if transpose and depthwise
        else "depthwise_conv3d"
        if depthwise
        else "pointwise_conv3d"
    )
    direction = "transpose" if transpose else ("downsample" if conv.stride[0] == 2 else "regular")
    return OperatorDescriptor(
        family=family,
        direction=direction,
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        role=role,
    )


class AdaptivePointwise3d(nn.Module):
    _mednext_accel_backend_kind = "pointwise_gemm"

    def __init__(
        self,
        conv: nn.Conv3d,
        resolver: OptimizationResolver,
        model_context: ModelOptimizationContext,
        *,
        role: str | None = None,
    ) -> None:
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.descriptor = _conv_descriptor(conv, role=role)
        self.resolver = resolver
        self.model_context = model_context
        self.train(conv.training)

    def decision_for_shape(
        self,
        *,
        batch_size: int,
        spatial_shape: tuple[int, int, int],
        dtype: torch.dtype | str,
        device_type: str,
        sm: tuple[int, int] | None,
        total_vram_bytes: int,
    ) -> Decision:
        context = execution_context_for_shape(
            self.model_context,
            batch_size=batch_size,
            spatial_shape=spatial_shape,
            dtype=dtype,
            device_type=device_type,
            sm=sm,
            total_vram_bytes=total_vram_bytes,
            training=self.training,
        )
        return self.resolver.resolve(self.descriptor, context, context.phase)

    def _gemm(self, x: Tensor) -> Tensor:
        unbatched = x.ndim == 4
        if unbatched:
            x = x.unsqueeze(0)
        spatial = x.shape[2:]
        flattened = x.flatten(2)
        weight = self.weight.flatten(1)
        outputs = [
            torch.mm(weight, sample)
            if self.bias is None
            else torch.addmm(self.bias[:, None], weight, sample)
            for sample in flattened.unbind()
        ]
        output = torch.stack(outputs).reshape(x.shape[0], self.out_channels, *spatial)
        return output.squeeze(0) if unbatched else output

    def forward(self, x: Tensor) -> Tensor:
        context = _context_from_tensor(x, self.model_context, training=self.training)
        decision = self.resolver.resolve(self.descriptor, context, context.phase)
        if decision.implementation == "pointwise_gemm_per_sample" and torch.is_grad_enabled():
            return self._gemm(x)
        return F.conv3d(x, self.weight, self.bias)


class AdaptiveDepthwise3d(nn.Module):
    _mednext_accel_backend_kind = "depthwise"

    def __init__(
        self,
        conv: nn.Conv3d | nn.ConvTranspose3d,
        resolver: OptimizationResolver,
        model_context: ModelOptimizationContext,
        *,
        role: str | None = None,
    ) -> None:
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        for name in (
            "in_channels", "out_channels", "kernel_size", "stride", "padding",
            "dilation", "groups",
        ):
            setattr(self, name, getattr(conv, name))
        self.output_padding = getattr(conv, "output_padding", (0, 0, 0))
        self.transpose = isinstance(conv, nn.ConvTranspose3d)
        self.descriptor = _conv_descriptor(conv, role=role)
        self.resolver = resolver
        self.model_context = model_context
        self.train(conv.training)

    def _reference(self, x: Tensor) -> Tensor:
        if self.transpose:
            return F.conv_transpose3d(
                x, self.weight, self.bias, self.stride, self.padding,
                self.output_padding, self.groups, self.dilation,
            )
        return F.conv3d(
            x, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups,
        )

    def forward(self, x: Tensor) -> Tensor:
        context = _context_from_tensor(x, self.model_context, training=self.training)
        if (
            context.phase != "training"
            or context.device_type != "cuda"
            or not torch.is_grad_enabled()
            or x.ndim != 5
            or not x.is_contiguous()
            or tuple(x.shape[2:]) != (x.shape[2],) * 3
        ):
            return self._reference(x)
        dx = self.resolver.resolve(self.descriptor, context, "backward_input")
        dw = self.resolver.resolve(self.descriptor, context, "backward_weight")
        from ._triton import depthwise as backend

        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        size = int(x.shape[2])
        kernel = self.kernel_size[0]
        if self.transpose and dw.implementation == "triton_transpose_split_dw":
            return backend.depthwise_conv_transpose3d_regular(
                x, weight, bias, kernel, self.in_channels, size,
                int(dw.parameters.get("dw_splits", 8)),
                int(dw.parameters.get("dw_block", 256)),
            )
        if self.stride == (2, 2, 2) and dx.implementation == "triton_downsample_dx":
            return backend.depthwise_conv3d_stride2_regular(
                x, weight, bias, kernel, self.in_channels, size,
                int(dx.parameters.get("dx_block", 128)),
            )
        if (
            self.stride == (1, 1, 1)
            and dx.implementation == "triton_depthwise_dx"
            and dw.implementation == "triton_split_dw"
        ):
            return backend.depthwise_conv3d_regular(
                x, weight, bias, kernel, self.in_channels, size,
                int(dw.parameters.get("dw_splits", 8)),
                int(dw.parameters.get("dw_block", 256)),
                int(dx.parameters.get("dx_block", 256)),
            )
        return self._reference(x)


def _pointwise_eligible(module: nn.Module) -> bool:
    return (
        type(module) is nn.Conv3d
        and module.kernel_size == (1, 1, 1)
        and module.stride == (1, 1, 1)
        and module.padding == (0, 0, 0)
        and module.dilation == (1, 1, 1)
        and module.groups == 1
        and module.padding_mode == "zeros"
    )


def _depthwise_eligible(module: nn.Module) -> bool:
    if type(module) not in (nn.Conv3d, nn.ConvTranspose3d):
        return False
    kernel = module.kernel_size[0]
    return (
        len(set(module.kernel_size)) == 1
        and kernel % 2 == 1
        and module.in_channels == module.out_channels == module.groups
        and module.stride in ((1, 1, 1), (2, 2, 2))
        and module.padding == (kernel // 2,) * 3
        and module.dilation == (1, 1, 1)
        and getattr(module, "padding_mode", "zeros") == "zeros"
        and getattr(module, "output_padding", (0, 0, 0)) == (0, 0, 0)
    )


def install_adaptive_operators(
    model: nn.Module,
    *,
    resolver: OptimizationResolver,
    model_context: ModelOptimizationContext,
) -> int:
    """Install wrappers in place while retaining every Parameter object."""

    count = 0
    for name, child in list(model.named_children()):
        role = name
        if _pointwise_eligible(child):
            setattr(model, name, AdaptivePointwise3d(child, resolver, model_context, role=role))
            count += 1
        elif _depthwise_eligible(child):
            setattr(model, name, AdaptiveDepthwise3d(child, resolver, model_context, role=role))
            count += 1
        else:
            count += install_adaptive_operators(
                child, resolver=resolver, model_context=model_context
            )
    return count
