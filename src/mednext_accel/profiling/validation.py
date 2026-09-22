"""Numerical kernel checks and whole-model validation-boundary handling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .synthesize import Measurement, Objective, candidate_wins


def validate_components(
    actual: Mapping[str, Tensor], expected: Mapping[str, Tensor]
) -> dict[str, object]:
    """Compare named tensors independently using finite values and relative L2.

    The 2% relative-L2 threshold tolerates different BF16 reduction orders near
    zero without allowing a bad component to hide behind a larger tensor.
    Returned metrics contain only JSON primitives and retain no tensors.
    """
    import torch

    if not actual or actual.keys() != expected.keys():
        raise ValueError("validation requires the same nonempty component names")
    metrics = {}
    reason = None
    with torch.no_grad():
        for name in sorted(expected):
            got, want = actual[name].detach().float(), expected[name].detach().float()
            finite = bool(torch.isfinite(got).all() and torch.isfinite(want).all())
            relative = maximum = None
            component_reason = None
            if got.shape != want.shape:
                component_reason = f"{name}: shape mismatch"
            elif not finite:
                component_reason = f"{name}: non-finite values"
            else:
                difference = got - want
                relative = (difference.norm() / want.norm().clamp_min(1e-12)).item()
                maximum = difference.abs().max().item() if difference.numel() else 0.0
                if not isfinite(relative) or not isfinite(maximum):
                    component_reason = f"{name}: non-finite error metrics"
                    relative = relative if isfinite(relative) else None
                    maximum = maximum if isfinite(maximum) else None
                elif relative >= 0.02:
                    component_reason = f"{name}: relative L2 error must be below 0.02"
            metrics[name] = {"finite": finite, "relative_l2": relative, "max_absolute": maximum}
            if reason is None:
                reason = component_reason
    return {
        "valid": reason is None,
        "validator": "component-relative-l2-v1",
        "validation_metrics": metrics,
        "rejection_reason": reason,
    }


DecisionEntry = tuple[
    str,
    str,
    str,
    tuple[int, int, int],
    int,
    int,
    str,
    tuple[tuple[str, int], ...],
]
DecisionSignature = tuple[DecisionEntry, ...]


@dataclass(frozen=True, slots=True)
class DecisionSegment:
    """A consecutive run of batches sharing one optimization decision."""

    batches: tuple[int, ...]
    signature: DecisionSignature


def decision_signature(
    measurements: Sequence[Measurement], objective: Objective
) -> DecisionSignature:
    """Return the deterministic optimization decision for one batch.

    Workload and checkpoint contexts are segmented independently by the caller,
    so the signature contains the measurement identity and selected
    implementation parameters only.
    """
    entries = []
    for item in measurements:
        winner = candidate_wins(item, objective)
        entries.append(
            (
                item.family,
                item.direction,
                item.phase,
                item.spatial_shape,
                item.in_channels,
                item.out_channels,
                item.implementation if winner else "reference",
                item.parameters if winner else (),
            )
        )
    return tuple(sorted(entries))


def segment_decisions(
    decisions: Mapping[int, DecisionSignature],
) -> tuple[DecisionSegment, ...]:
    """Group equal decisions across exactly consecutive sampled batches."""
    segments: list[DecisionSegment] = []
    for batch, signature in sorted(decisions.items()):
        if segments:
            previous = segments[-1]
            if batch == previous.batches[-1] + 1 and signature == previous.signature:
                segments[-1] = DecisionSegment((*previous.batches, batch), signature)
                continue
        segments.append(DecisionSegment((batch,), signature))
    return tuple(segments)


def validation_batches(segments: Sequence[DecisionSegment]) -> tuple[int, ...]:
    """Return unique segment endpoints in deterministic ascending order."""
    endpoints = {
        endpoint
        for segment in segments
        if segment.batches
        for endpoint in (segment.batches[0], segment.batches[-1])
    }
    return tuple(sorted(endpoints))


def apply_validation_results(
    measurements: Sequence[Measurement],
    segments: Sequence[DecisionSegment],
    accepted: Mapping[int, bool],
) -> tuple[Measurement, ...]:
    """Propagate failed boundary checks to every measurement in that segment.

    A segment is accepted only when each of its selected endpoint batches has
    an explicit ``True`` result. Kernel evidence stays unchanged; the separate
    whole-model outcome explains why an otherwise valid kernel was rejected.
    """
    outcomes: dict[int, bool] = {}
    for segment in segments:
        endpoints = (
            (
                (segment.batches[0],)
                if len(segment.batches) == 1
                else (segment.batches[0], segment.batches[-1])
            )
            if segment.batches
            else ()
        )
        segment_valid = all(accepted.get(batch) is True for batch in endpoints)
        for batch in segment.batches:
            outcomes[batch] = outcomes.get(batch, True) and segment_valid

    return tuple(
        replace(
            item,
            valid=item.valid and outcomes[item.batch],
            whole_model_valid=outcomes[item.batch],
            rejection_reason=item.rejection_reason
            or (None if outcomes[item.batch] else "whole-model validation rejected segment"),
        )
        if item.batch in outcomes
        else item
        for item in measurements
    )
