"""Model-level optimization policy application."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from os import PathLike
from pathlib import Path

import torch
from torch import nn

from ..ops.depthwise import replace_depthwise_convs
from ..ops.pointwise import replace_pointwise_convs
from .autotune import autotune_selections
from .policy import conservative_selections
from .report import (
    BackendSelections,
    CompileMode,
    OptimizationPolicy,
    OptimizationReport,
)

_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}


def _default_cache_path() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(root) if root else Path.home() / ".cache"
    return cache_root / "mednext_accel" / "autotune-v1.json"


_DEFAULT_CACHE_PATH = _default_cache_path()


def _device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must have parameters") from error


def _set_selections(model: nn.Module, selections: BackendSelections) -> None:
    for module in model.modules():
        kind = getattr(module, "_mednext_accel_backend_kind", None)
        if kind is not None:
            module.selected_shapes = frozenset(getattr(selections, kind))


def _apply_selections(model: nn.Module, selections: BackendSelections) -> int:
    replacements = 0
    if selections.pointwise_gemm:
        replacements += replace_pointwise_convs(model, selected_shapes=selections.pointwise_gemm)
    if any(
        (
            selections.depthwise_regular,
            selections.depthwise_transpose,
            selections.depthwise_downsample,
        )
    ):
        replacements += replace_depthwise_convs(
            model,
            include_stride2_input_grad=bool(selections.depthwise_downsample),
            policy={
                "depthwise_regular": selections.depthwise_regular,
                "depthwise_transpose": selections.depthwise_transpose,
                "depthwise_downsample": selections.depthwise_downsample,
            },
        )
    _set_selections(model, selections)
    return replacements


def optimize(
    model: nn.Module,
    *,
    input_shape: tuple[int, ...],
    dtype: torch.dtype,
    policy: OptimizationPolicy,
    compile_mode: CompileMode = "default",
    cache_path: str | PathLike[str] | None = _DEFAULT_CACHE_PATH,
    warmup: int = 10,
    repetitions: int = 50,
    include_stride2_input_grad: bool = False,
) -> OptimizationReport:
    """Apply an execution policy in place and return a reviewable report."""

    if policy not in ("torch", "conservative", "autotune"):
        raise ValueError("policy must be 'torch', 'conservative', or 'autotune'")
    if compile_mode not in _COMPILE_MODES:
        raise ValueError(f"unknown torch.compile mode {compile_mode!r}")
    input_shape = tuple(input_shape)
    device = _device(model)
    measurements = ()
    cache_hit = False
    cache_key = None
    notes: tuple[str, ...] = ()

    if policy == "torch":
        selections = BackendSelections()
    elif policy == "conservative":
        capability = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
        selections = conservative_selections(
            device_type=device.type,
            capability=capability,
            dtype=dtype,
            input_shape=input_shape,
        )
        if selections.is_empty:
            notes = ("no conservative choices are validated for this execution context",)
    else:
        tuned = autotune_selections(
            model,
            input_shape=input_shape,
            dtype=dtype,
            warmup=warmup,
            repetitions=repetitions,
            cache_path=cache_path,
            include_stride2_input_grad=include_stride2_input_grad,
            compile_mode=compile_mode,
        )
        selections = tuned.selections
        measurements = tuned.measurements
        cache_hit = tuned.cache_hit
        cache_key = tuned.cache_key

    replacements = _apply_selections(model, selections)
    report = OptimizationReport(
        policy=policy,
        compile_mode=compile_mode,
        input_shape=input_shape,
        dtype=str(dtype),
        device=str(device),
        selections=selections,
        measurements=measurements,
        replacements=replacements,
        cache_hit=cache_hit,
        cache_key=cache_key,
        notes=notes,
    )
    model.__dict__["_mednext_accel_optimization_report"] = report
    return report


@contextmanager
def use_backend(model: nn.Module, backend: str) -> Iterator[nn.Module]:
    """Temporarily force existing acceleration wrappers to native PyTorch."""

    if backend != "torch":
        raise ValueError("the context manager currently supports only the 'torch' backend")
    wrappers = [
        module
        for module in model.modules()
        if getattr(module, "_mednext_accel_backend_kind", None) is not None
    ]
    previous = [module.selected_shapes for module in wrappers]
    try:
        for module in wrappers:
            module.selected_shapes = frozenset()
        yield model
    finally:
        for module, selected_shapes in zip(wrappers, previous, strict=True):
            module.selected_shapes = selected_shapes
