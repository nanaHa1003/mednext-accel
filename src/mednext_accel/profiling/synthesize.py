"""Convert benchmark measurements into reusable profile rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from ..optimization.schema import OptimizationProfile, parse_profile

Objective = Literal["balanced", "throughput", "memory"]


@dataclass(frozen=True, slots=True)
class Measurement:
    family: str
    direction: str
    phase: str
    implementation: str
    batch: int
    spatial_shape: tuple[int, int, int]
    in_channels: int
    out_channels: int
    dtype: str
    checkpointing: str
    reference_ms: float
    candidate_ms: float
    reference_peak_bytes: int
    candidate_peak_bytes: int
    valid: bool

    def to_primitive(self) -> dict[str, object]:
        return asdict(self)


def _wins(item: Measurement, objective: Objective) -> bool:
    if not item.valid or item.reference_ms <= 0 or item.candidate_ms <= 0:
        return False
    if objective == "throughput":
        return item.candidate_ms < item.reference_ms
    if objective == "memory":
        return (
            item.candidate_peak_bytes < item.reference_peak_bytes
            and item.candidate_ms <= item.reference_ms * 1.10
        )
    return (
        item.candidate_ms <= item.reference_ms * 0.97
        and item.candidate_peak_bytes <= item.reference_peak_bytes * 1.15
    )


def _defaults() -> dict[str, dict[str, dict[str, object]]]:
    phases = (
        "training", "inference", "export", "backward_input", "backward_weight",
        "backward_bias",
    )
    return {
        family: {
            phase: {"implementation": "reference", "parameters": {}}
            for phase in phases
        }
        for family in (
            "pointwise_conv3d", "depthwise_conv3d", "depthwise_conv_transpose3d",
            "group_norm", "gelu",
        )
    }


def synthesize_profile(
    measurements: list[Measurement] | tuple[Measurement, ...],
    *,
    name: str,
    sm: tuple[int, int],
    objective: Objective,
) -> OptimizationProfile:
    winners = sorted((item for item in measurements if _wins(item, objective)), key=lambda item: (
        item.family, item.direction, item.phase, item.implementation,
        item.spatial_shape, item.in_channels, item.out_channels, item.dtype,
        item.checkpointing, item.batch,
    ))
    groups: list[list[Measurement]] = []
    for item in winners:
        identity = (
            item.family, item.direction, item.phase, item.implementation,
            item.spatial_shape, item.in_channels, item.out_channels, item.dtype,
            item.checkpointing,
        )
        if groups:
            previous = groups[-1][-1]
            previous_identity = (
                previous.family, previous.direction, previous.phase, previous.implementation,
                previous.spatial_shape, previous.in_channels, previous.out_channels,
                previous.dtype, previous.checkpointing,
            )
            if identity == previous_identity and item.batch == previous.batch + 1:
                groups[-1].append(item)
                continue
        groups.append([item])
    rules = []
    for index, group in enumerate(groups):
        first, last = group[0], group[-1]
        rules.append({
            "id": f"generated-{index:04d}",
            "family": first.family,
            "direction": first.direction,
            "match": {
                "batch": {"min": first.batch, "max": last.batch},
                "spatial_shape": list(first.spatial_shape),
                "in_channels": first.in_channels,
                "out_channels": first.out_channels,
                "dtype": first.dtype,
                "checkpointing": first.checkpointing,
            },
            "phases": {
                first.phase: {"implementation": first.implementation, "parameters": {}}
            },
            "confidence": "measured" if len(group) == 1 else "interpolated",
        })
    return parse_profile({
        "schema_version": 1,
        "profile": {
            "name": name, "target": {"vendor": "nvidia", "sm": list(sm)},
            "provenance": {"generator": "mednext-accel", "objective": objective,
                           "measurement_count": len(measurements)},
        },
        "defaults": _defaults(), "rules": rules, "overrides": [],
        "measurements": [item.to_primitive() for item in measurements],
    })
