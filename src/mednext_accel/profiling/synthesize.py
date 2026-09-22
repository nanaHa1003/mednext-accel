"""Convert benchmark measurements into reusable profile rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Literal

from ..optimization.policy import OptimizationPolicy, parse_policy
from .evidence import (
    OBJECTIVES,
    freeze,
    primitive,
    record_fields,
    require_bool,
    require_int,
    require_mapping,
    require_number,
    require_sequence,
    require_string,
    validate_json,
    validate_workload,
)
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
    kernel_valid: bool | None
    parameters: tuple[tuple[str, int], ...] = ()
    probe_status: str = "ok"
    probe_message: str | None = None
    objective_winner: bool | None = None
    objective: str | None = None
    failure_stage: str | None = None
    seed: int | None = None
    kernel_size: int = 1
    workload: Mapping[str, object] = field(default_factory=dict)
    validator: str | None = None
    validation_metrics: Mapping[str, Mapping[str, float | bool | None]] = field(
        default_factory=dict
    )
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        require_bool(self.kernel_valid, "kernel_valid", nullable=True)
        require_bool(self.objective_winner, "objective_winner", nullable=True)
        for name in (
            "family",
            "direction",
            "phase",
            "implementation",
            "dtype",
            "checkpointing",
            "probe_status",
        ):
            require_string(getattr(self, name), name)
        for name in ("probe_message", "failure_stage", "validator", "rejection_reason"):
            require_string(getattr(self, name), name, nullable=True)
        for name in ("batch", "in_channels", "out_channels", "kernel_size"):
            require_int(getattr(self, name), name, minimum=1)
        if self.seed is not None:
            require_int(self.seed, "seed", maximum=2**63 - 1)
        require_sequence(self.spatial_shape, "spatial_shape")
        if len(self.spatial_shape) != 3:
            raise ValueError("spatial_shape must contain three dimensions")
        for dimension in self.spatial_shape:
            require_int(dimension, "spatial_shape dimension", minimum=1)
        require_sequence(self.parameters, "parameters")
        names = set()
        for parameter in self.parameters:
            require_sequence(parameter, "parameter")
            if len(parameter) != 2:
                raise ValueError("parameter must contain a name and value")
            name, value = parameter
            require_string(name, "parameter name")
            require_int(value, "parameter value", minimum=1)
            if name in names:
                raise ValueError("duplicate parameter name")
            names.add(name)
        validate_workload(self.workload)
        require_mapping(self.validation_metrics, "validation_metrics")
        for name, metrics in self.validation_metrics.items():
            record_fields(
                metrics,
                {"finite", "relative_l2", "max_absolute"},
                f"validation_metrics.{name}",
                required=set(),
            )
            if "finite" in metrics:
                require_bool(metrics["finite"], f"validation_metrics.{name}.finite")
            for metric in ("relative_l2", "max_absolute"):
                if metric in metrics:
                    require_number(metrics[metric], f"validation_metrics.{name}.{metric}")
                    if metrics[metric] is not None and metrics[metric] < 0:
                        raise ValueError(f"validation_metrics.{name}.{metric} must be nonnegative")
            if self.kernel_valid is True:
                relative = metrics.get("relative_l2")
                if (
                    metrics.get("finite") is not True
                    or relative is None
                    or not isfinite(relative)
                    or relative >= 0.02
                ):
                    raise ValueError("kernel_valid contradicts validation_metrics")
                if "max_absolute" in metrics and (
                    metrics["max_absolute"] is None or not isfinite(metrics["max_absolute"])
                ):
                    raise ValueError("kernel_valid contradicts validation_metrics")
        if self.kernel_valid is True and self.rejection_reason is not None:
            raise ValueError("kernel_valid contradicts numerical rejection_reason")
        object.__setattr__(self, "spatial_shape", tuple(self.spatial_shape))
        object.__setattr__(self, "parameters", tuple(tuple(item) for item in self.parameters))
        object.__setattr__(self, "validation_metrics", freeze(self.validation_metrics))
        object.__setattr__(self, "workload", freeze(self.workload))
        for name in (
            "reference_ms",
            "candidate_ms",
            "reference_peak_bytes",
            "candidate_peak_bytes",
        ):
            require_number(getattr(self, name), name)
            object.__setattr__(self, name, freeze(getattr(self, name)))
        validate_json(self.workload)
        validate_json(self.validation_metrics)
        if self.objective is None:
            if self.objective_winner is not None:
                raise ValueError("objective_winner requires an objective")
        elif self.objective not in OBJECTIVES:
            raise ValueError("unsupported objective")
        elif self.objective_winner is not candidate_wins(self, self.objective):
            raise ValueError("objective_winner contradicts the objective and kernel measurements")

    def to_primitive(self) -> dict[str, object]:
        return primitive(self)


def measurement_from_result(
    case: KernelCase,
    result: Mapping[str, object],
    *,
    checkpointing: str,
    objective: Objective = "balanced",
    workload: Mapping[str, object] | None = None,
) -> Measurement:
    """Retain observed kernel facts, including partial results after a later failure."""
    record_fields(
        result,
        {
            "valid",
            "status",
            "message",
            "reference_ms",
            "candidate_ms",
            "reference_peak_bytes",
            "candidate_peak_bytes",
            "seed",
            "failure_stage",
            "validator",
            "validation_metrics",
            "rejection_reason",
            "case_id",
            "implementation",
            "parameters",
            "checkpointing",
        },
        "kernel result",
        required=set(),
    )
    if "valid" in result:
        require_bool(result["valid"], "kernel result valid")
    key = case.key
    item = Measurement(
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
        reference_ms=result.get("reference_ms", 0.0),
        candidate_ms=result.get("candidate_ms", 0.0),
        reference_peak_bytes=result.get("reference_peak_bytes", 0),
        candidate_peak_bytes=result.get("candidate_peak_bytes", 0),
        parameters=key.parameters,
        probe_status=result.get("status", "missing"),
        probe_message=result.get("message"),
        kernel_valid=result.get("valid"),
        failure_stage=result.get("failure_stage"),
        seed=result.get("seed"),
        kernel_size=key.kernel_size,
        workload={} if workload is None else workload,
        validator=result.get("validator"),
        validation_metrics=result.get("validation_metrics", {}),
        rejection_reason=result.get("rejection_reason"),
    )
    return replace(item, objective=objective, objective_winner=candidate_wins(item, objective))


def candidate_wins(item: Measurement, objective: Objective) -> bool:
    """Return whether a validated candidate satisfies the profile objective."""
    if objective not in ("balanced", "throughput", "memory"):
        raise ValueError(f"unknown profiling objective {objective!r}")
    if item.kernel_valid is not True or item.probe_status != "ok":
        return False
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
        for value in (
            item.reference_ms,
            item.candidate_ms,
            item.reference_peak_bytes,
            item.candidate_peak_bytes,
        )
    ):
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


def measurement_identity(item: Measurement) -> tuple[object, ...]:
    """The policy match fields; candidate parameters do not constrain matching."""
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


def selectable_measurements(measurements: Sequence[Measurement]) -> tuple[Measurement, ...]:
    """Exclude contradictory policy matches without modifying source evidence."""
    rejected = {measurement_identity(item) for item in measurements if item.kernel_valid is False}
    return tuple(item for item in measurements if measurement_identity(item) not in rejected)


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
    merge_adjacent_batches: bool = True,
) -> OptimizationPolicy:
    measurements = selectable_measurements(measurements)
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
            if (
                merge_adjacent_batches
                and identity == previous_identity
                and item.batch == previous.batch + 1
            ):
                groups[-1].append(item)
                continue
        groups.append([item])
    rules = []
    for index, group in enumerate(groups):
        first, last = group[0], group[-1]
        rules.append(
            {
                "id": f"generated-{index:04d}",
                "when": {
                    "family": first.family,
                    "direction": first.direction,
                    "batch": {"min": first.batch, "max": last.batch},
                    "spatial_shape": list(first.spatial_shape),
                    "channels": [first.in_channels, first.out_channels],
                    "dtype": first.dtype,
                    "checkpointing": first.checkpointing,
                },
                "use": {
                    first.phase: {
                        "implementation": first.implementation,
                        "parameters": dict(first.parameters),
                    }
                },
                "confidence": "measured-exact-context"
                if len(group) == 1
                else "interpolated-bounded",
            }
        )
    # Runtime documents contain only dispatch. The measurements remain on the
    # campaign result pending the separate evidence artifact integration.
    return parse_policy(
        {
            "version": 2,
            "kind": "mednext-accel-policy",
            "name": name,
            "target": {"vendor": "nvidia", "sm": list(sm)},
            "rules": rules,
        }
    )
