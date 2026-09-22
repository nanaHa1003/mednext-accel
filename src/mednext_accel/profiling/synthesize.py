"""Convert benchmark measurements into reusable profile rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal

from ..optimization.schema import OptimizationProfile, parse_profile
from .matrix import KernelCase

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
    parameters: tuple[tuple[str, int], ...] = ()

    def to_primitive(self) -> dict[str, object]:
        return asdict(self)


def measurement_from_result(
    case: KernelCase,
    result: Mapping[str, object],
    *,
    checkpointing: str,
) -> Measurement:
    """Materialize context-free evidence for one workload reference.

    Failed or absent results retain their planned identity with zero metrics,
    ensuring synthesis explicitly falls back to the reference implementation.
    """
    ok = result.get("status") == "ok"
    key = case.key
    return Measurement(
        family=key.family,
        direction=key.direction,
        phase=key.phase,
        implementation=key.implementation,
        batch=key.batch,
        spatial_shape=key.spatial_shape,
        in_channels=key.in_channels,
        out_channels=key.out_channels,
        dtype=key.dtype,
        checkpointing=checkpointing,
        reference_ms=float(result.get("reference_ms", 0.0)) if ok else 0.0,
        candidate_ms=float(result.get("candidate_ms", 0.0)) if ok else 0.0,
        reference_peak_bytes=int(result.get("reference_peak_bytes", 0)) if ok else 0,
        candidate_peak_bytes=int(result.get("candidate_peak_bytes", 0)) if ok else 0,
        valid=ok and bool(result.get("valid", False)),
        parameters=key.parameters,
    )


def candidate_wins(item: Measurement, objective: Objective) -> bool:
    """Return whether a validated candidate satisfies the profile objective."""
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


def reconcile_measurements(measurements: Sequence[Measurement]) -> tuple[Measurement, ...]:
    """Invalidate candidates that schema-v1 rules cannot distinguish from a rejection.

    The identity includes exactly the emitted match fields plus family, direction,
    and phase. Implementation and launch parameters select a candidate; they do not
    constrain which workload matches it.
    """

    def identity(item: Measurement) -> tuple[object, ...]:
        return (
            item.family,
            item.direction,
            item.phase,
            item.batch,
            item.spatial_shape,
            item.in_channels,
            item.out_channels,
            item.dtype,
            item.checkpointing,
        )

    rejected = {identity(item) for item in measurements if not item.valid}
    return tuple(
        replace(item, valid=False) if item.valid and identity(item) in rejected else item
        for item in measurements
    )


def _defaults() -> dict[str, dict[str, dict[str, object]]]:
    phases = (
        "training",
        "inference",
        "export",
        "backward_input",
        "backward_weight",
        "backward_bias",
    )
    return {
        family: {phase: {"implementation": "reference", "parameters": {}} for phase in phases}
        for family in (
            "pointwise_conv3d",
            "depthwise_conv3d",
            "depthwise_conv_transpose3d",
            "group_norm",
            "gelu",
        )
    }


def synthesize_profile(
    measurements: list[Measurement] | tuple[Measurement, ...],
    *,
    name: str,
    sm: tuple[int, int],
    objective: Objective,
    environment: dict[str, object] | None = None,
    compile_mode: str | None = None,
    execution: Mapping[str, int] | None = None,
    campaign: Mapping[str, object] | None = None,
) -> OptimizationProfile:
    measurements = reconcile_measurements(measurements)
    winners = sorted(
        (item for item in measurements if candidate_wins(item, objective)),
        key=lambda item: (
            item.family,
            item.direction,
            item.phase,
            item.implementation,
            item.spatial_shape,
            item.in_channels,
            item.out_channels,
            item.dtype,
            item.checkpointing,
            item.parameters,
            item.batch,
        ),
    )
    groups: list[list[Measurement]] = []
    for item in winners:
        identity = (
            item.family,
            item.direction,
            item.phase,
            item.implementation,
            item.spatial_shape,
            item.in_channels,
            item.out_channels,
            item.dtype,
            item.checkpointing,
            item.parameters,
        )
        if groups:
            previous = groups[-1][-1]
            previous_identity = (
                previous.family,
                previous.direction,
                previous.phase,
                previous.implementation,
                previous.spatial_shape,
                previous.in_channels,
                previous.out_channels,
                previous.dtype,
                previous.checkpointing,
                previous.parameters,
            )
            if identity == previous_identity and item.batch == previous.batch + 1:
                groups[-1].append(item)
                continue
        groups.append([item])
    rules = []
    for index, group in enumerate(groups):
        first, last = group[0], group[-1]
        rules.append(
            {
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
                    first.phase: {
                        "implementation": first.implementation,
                        "parameters": dict(first.parameters),
                    }
                },
                "confidence": "measured" if len(group) == 1 else "interpolated",
            }
        )
    return parse_profile(
        {
            "schema_version": 1,
            "profile": {
                "name": name,
                "target": {"vendor": "nvidia", "sm": list(sm)},
                "provenance": {
                    "generator": "mednext-accel",
                    "objective": objective,
                    "measurement_count": len(measurements),
                    **({"environment": environment} if environment is not None else {}),
                    **({"compile_mode": compile_mode} if compile_mode is not None else {}),
                    **({"execution": dict(execution)} if execution is not None else {}),
                    **({"campaign": dict(campaign)} if campaign is not None else {}),
                },
            },
            "defaults": _defaults(),
            "rules": rules,
            "overrides": [],
            "measurements": [item.to_primitive() for item in measurements],
        }
    )
