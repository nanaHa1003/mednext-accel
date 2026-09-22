"""Ordinary Python launch recipes shared by policies and profiling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .descriptors import ExecutionContext, OperatorDescriptor

if TYPE_CHECKING:
    from .policy import PolicySelection

_RECIPE_IMPLEMENTATIONS = frozenset(
    {
        "triton_depthwise_dx",
        "triton_downsample_dx",
        "triton_split_dw",
        "triton_transpose_split_dw",
    }
)


def has_parameter_recipe(implementation: str) -> bool:
    return implementation in _RECIPE_IMPLEMENTATIONS


def validate_parameters(implementation: str, parameters: Mapping[str, object]) -> None:
    """Validate literal launch overrides without evaluating user expressions."""
    if implementation in ("triton_depthwise_dx", "triton_downsample_dx"):
        allowed = {"dx_block"}
    elif implementation in ("triton_split_dw", "triton_transpose_split_dw"):
        allowed = {"dw_block", "dw_splits"}
    else:
        allowed = None
    for name, value in parameters.items():
        if not isinstance(name, str):
            raise ValueError("parameters keys must be strings")
        if allowed is not None and name not in allowed:
            raise ValueError(f"unknown parameters field {name!r} for {implementation}")
        if allowed is None:
            if value is not None and not isinstance(value, (str, bool, int, float)):
                raise ValueError(f"parameters.{name} must be a literal scalar")
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(f"parameters.{name} must be a positive integer")
        if name == "dw_splits" and value > 512:
            raise ValueError("parameters.dw_splits must be <= 512")
        if name.endswith("_block") and (value & (value - 1) or value > 1024):
            raise ValueError(f"parameters.{name} must be a power of two <= 1024")


def parameter_defaults(
    implementation: str,
    descriptor: OperatorDescriptor,
    context: ExecutionContext,
) -> dict[str, int]:
    """Compute launch defaults; these formulas do not select an implementation.

    Shared SM86/SM89 reduction recipes scale with every spatial dimension.
    SM120's existing base launches are preserved only in their shape/channel
    regions; its two batch exceptions remain explicit policy overrides.
    """
    if not has_parameter_recipe(implementation):
        raise ValueError(f"{implementation!r} has no parameter recipe")
    if implementation in ("triton_depthwise_dx", "triton_downsample_dx"):
        return {"dx_block": 128}
    transpose = implementation == "triton_transpose_split_dw"
    reduction_work = context.batch_size * context.spatial_volume
    splits = min(512, max(1, round(reduction_work / (4096 if transpose else 32768))))
    block = 512
    if context.sm == (12, 0) and descriptor.in_channels == descriptor.out_channels:
        region = (context.spatial_shape, descriptor.in_channels)
        if transpose and region == ((64, 64, 64), 64):
            splits, block = 64, 1024
        elif not transpose:
            # (splits per sample, block); integer batches preserve legacy scaling.
            regions = {
                ((128, 128, 128), 32): (64, 1024),
                ((64, 64, 64), 64): (8, 1024),
                ((32, 32, 32), 128): (2, 1024),
                ((16, 16, 16), 256): (1, 512),
                ((8, 8, 8), 512): (1, 128),
            }
            if region in regions:
                per_sample, block = regions[region]
                splits = min(512, per_sample * context.batch_size)
    return {"dw_splits": splits, "dw_block": block}


def resolve_parameters(
    selection: PolicySelection,
    descriptor: OperatorDescriptor,
    context: ExecutionContext,
) -> dict[str, object]:
    """Omission is empty; explicit mappings override available recipe defaults."""
    supplied = selection.parameters
    if supplied is None:
        return {}
    if supplied == "auto":
        return parameter_defaults(selection.implementation, descriptor, context)
    result = (
        parameter_defaults(selection.implementation, descriptor, context)
        if has_parameter_recipe(selection.implementation)
        else {}
    )
    result.update(supplied)
    validate_parameters(selection.implementation, result)
    return result
