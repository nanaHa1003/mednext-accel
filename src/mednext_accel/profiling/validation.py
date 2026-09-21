"""CPU-only decision segmentation and validation-boundary handling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .synthesize import Measurement, Objective, candidate_wins

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
    an explicit ``True`` result. Measurements in accepted or unrelated
    segments retain their original validity.
    """
    rejected_batches: set[int] = set()
    for segment in segments:
        endpoints = (
            (segment.batches[0],)
            if len(segment.batches) == 1
            else (segment.batches[0], segment.batches[-1])
        ) if segment.batches else ()
        if not all(accepted.get(batch) is True for batch in endpoints):
            rejected_batches.update(segment.batches)

    return tuple(
        replace(item, valid=False) if item.batch in rejected_batches else item
        for item in measurements
    )
