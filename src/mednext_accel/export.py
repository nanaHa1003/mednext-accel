"""Evaluation-only export helpers for MedNeXt models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def _require_eval(model: nn.Module) -> None:
    if model.training:
        raise ValueError("model must be in eval mode before tracing or export")


def trace(
    model: nn.Module,
    example: Tensor,
    *,
    check_trace: bool = True,
) -> torch.jit.ScriptModule:
    """Trace an eval model using the ordinary TorchScript trace API."""

    _require_eval(model)
    return torch.jit.trace(model, example, check_trace=check_trace)


def export_onnx(
    model: nn.Module,
    example: Tensor,
    *,
    opset_version: int = 18,
    dynamic_batch: bool = False,
) -> Any:
    """Export an eval model with PyTorch's dynamo-based ONNX exporter."""

    _require_eval(model)
    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1)}}
    return torch.onnx.export(
        model,
        (example,),
        f=None,
        opset_version=opset_version,
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
    )
