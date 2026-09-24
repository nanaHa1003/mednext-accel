"""Static discovery of optimizable model operators."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from torch import nn

from ..models.blocks import EvalModeGELU
from ..optimization.descriptors import OperatorDescriptor


@dataclass(frozen=True, slots=True)
class DiscoveredOperator:
    descriptor: OperatorDescriptor
    occurrences: int
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    operators: tuple[DiscoveredOperator, ...]
    architecture_fingerprint: str


def _descriptor(module: nn.Module, role: str) -> OperatorDescriptor | None:
    if type(module) in (nn.Conv3d, nn.ConvTranspose3d):
        transpose = type(module) is nn.ConvTranspose3d
        depthwise = module.groups == module.in_channels == module.out_channels
        if not depthwise and module.kernel_size != (1, 1, 1):
            return None
        family = (
            "depthwise_conv_transpose3d"
            if transpose and depthwise
            else "depthwise_conv3d"
            if depthwise
            else "pointwise_conv3d"
        )
        direction = (
            "transpose" if transpose else "downsample" if module.stride[0] == 2 else "regular"
        )
        return OperatorDescriptor(
            family,
            direction,
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            role,
        )
    if isinstance(module, nn.GroupNorm):
        return OperatorDescriptor.normalization(
            family="group_norm", channels=module.num_channels, role=role
        )
    if isinstance(module, (nn.GELU, EvalModeGELU)):
        return OperatorDescriptor(
            "gelu",
            "forward",
            1,
            1,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
            1,
            role,
        )
    return None


def discover_operators(model: nn.Module) -> DiscoveryResult:
    grouped: dict[tuple[object, ...], list[object]] = {}
    for role, module in model.named_modules():
        descriptor = _descriptor(module, role)
        if descriptor is None:
            continue
        key = (
            *descriptor.signature,
            descriptor.padding,
            descriptor.dilation,
            descriptor.groups,
        )
        if key not in grouped:
            grouped[key] = [replace(descriptor, role=None), []]
        grouped[key][1].append(role)  # type: ignore[union-attr]
    operators = tuple(
        DiscoveredOperator(item[0], len(item[1]), tuple(item[1]))  # type: ignore[arg-type]
        for _, item in sorted(grouped.items(), key=lambda pair: repr(pair[0]))
    )
    config = getattr(model, "config", None)
    identity = (
        asdict(config)
        if config is not None
        else {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "operators": [item.descriptor.to_primitive() for item in operators],
        }
    )
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return DiscoveryResult(operators, hashlib.sha256(encoded).hexdigest()[:16])
