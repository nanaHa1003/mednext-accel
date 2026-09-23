"""Convert benchmark measurements into reusable profile rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Literal

from ..optimization.policy import OptimizationPolicy, parse_policy
from .benchmark import normalize_probe_message
from .evidence import (
    OBJECTIVES,
    freeze,
    primitive,
    ratio_at_most,
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
    benchmark_kind: str = "raw_kernel_diagnostic"
    compile_mode: str | None = None
    gradient_mask: tuple[bool, bool, bool] | None = None
    raw_diagnostic: Mapping[str, object] = field(default_factory=dict)
    # None preserves older evidence whose memory accounting was not recorded.
    memory_measured: bool | None = None

    def __post_init__(self) -> None:
        if self.benchmark_kind not in ("raw_kernel_diagnostic", "integrated_operator"):
            raise ValueError("unsupported benchmark_kind")
        require_string(self.compile_mode, "compile_mode", nullable=True)
        require_bool(self.memory_measured, "memory_measured", nullable=True)
        if self.gradient_mask is not None:
            require_sequence(self.gradient_mask, "gradient_mask")
            if len(self.gradient_mask) != 3:
                raise ValueError("gradient_mask must contain input, weight and bias flags")
            for value in self.gradient_mask:
                require_bool(value, "gradient_mask flag")
            object.__setattr__(self, "gradient_mask", tuple(self.gradient_mask))
        require_mapping(self.raw_diagnostic, "raw_diagnostic")
        validate_json(self.raw_diagnostic)
        object.__setattr__(self, "raw_diagnostic", freeze(self.raw_diagnostic))
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
        elif (
            self.benchmark_kind == "integrated_operator"
            # Historical verdicts may predate memory-accounting metadata. They
            # remain diagnostic; current dispatch always recomputes eligibility.
            and (self.objective == "throughput" or self.memory_measured is not None)
            and self.objective_winner is not candidate_wins(self, self.objective)
        ):
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
            "benchmark_kind",
            "compile_mode",
            "gradient_mask",
            "raw_diagnostic",
            "memory_measured",
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
        probe_message=normalize_probe_message(
            result.get("status", "missing"), result.get("message")
        ),
        kernel_valid=result.get("valid"),
        failure_stage=result.get("failure_stage"),
        seed=result.get("seed"),
        kernel_size=key.kernel_size,
        workload={} if workload is None else workload,
        validator=result.get("validator"),
        validation_metrics=result.get("validation_metrics", {}),
        rejection_reason=result.get("rejection_reason"),
        benchmark_kind=result.get("benchmark_kind", "raw_kernel_diagnostic"),
        compile_mode=result.get("compile_mode"),
        gradient_mask=result.get("gradient_mask"),
        raw_diagnostic=result.get("raw_diagnostic", {}),
        memory_measured=result.get("memory_measured"),
    )
    return replace(item, objective=objective, objective_winner=candidate_wins(item, objective))


def candidate_wins(item: Measurement, objective: Objective) -> bool:
    """Return whether a validated candidate satisfies the profile objective."""
    if objective not in ("balanced", "throughput", "memory"):
        raise ValueError(f"unknown profiling objective {objective!r}")
    if item.benchmark_kind != "integrated_operator":
        return False
    if item.kernel_valid is not True or item.probe_status != "ok":
        return False
    if not complete_metrics(item, objective):
        return False
    if objective == "throughput":
        return item.candidate_ms < item.reference_ms
    if objective == "memory":
        return item.candidate_peak_bytes < item.reference_peak_bytes and ratio_at_most(
            item.candidate_ms, item.reference_ms, 11, 10
        )
    return ratio_at_most(item.candidate_ms, item.reference_ms, 97, 100) and ratio_at_most(
        item.candidate_peak_bytes, item.reference_peak_bytes, 23, 20
    )


def complete_metrics(item: Measurement, objective: Objective = "balanced") -> bool:
    """Require measured, finite metrics for this objective before calling a loser."""
    metrics = (item.reference_ms, item.candidate_ms)
    if objective != "throughput":
        if item.memory_measured is not True:
            return False
        metrics += (item.reference_peak_bytes, item.candidate_peak_bytes)
    return item.probe_status == "ok" and all(
        type(value) in (int, float) and isfinite(value) and value > 0 for value in metrics
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
        item.kernel_size,
        item.workload.get("model_family"),
        item.workload.get("variant"),
    )


def synthesize_profile(
    measurements: Sequence[Measurement],
    *,
    name: str,
    sm: tuple[int, int],
    objective: Objective,
    include_winners: bool = True,
) -> OptimizationPolicy:
    """Build an exact local overlay; missing observations remain policy gaps.

    Numerical failures and complete objective losers explicitly select reference.
    A later aggregate rejection removes winners, never the negative evidence.
    """
    by_match: dict[tuple[object, ...], list[Measurement]] = {}
    for item in measurements:
        if item.benchmark_kind != "integrated_operator":
            continue
        by_match.setdefault(measurement_identity(item), []).append(item)
    selected = []
    for items in by_match.values():
        # A failed observation rejects its recipe, not other candidate recipes.
        rejected = {
            (item.implementation, tuple(sorted(item.parameters)))
            for item in items
            if item.kernel_valid is False
        }
        winners = [
            item
            for item in items
            if (item.implementation, tuple(sorted(item.parameters))) not in rejected
            and candidate_wins(item, objective)
        ]
        if winners and include_winners:
            winner = min(
                winners,
                key=lambda item: (
                    item.candidate_peak_bytes if objective == "memory" else item.candidate_ms,
                    item.implementation,
                    tuple(sorted(item.parameters)),
                ),
            )
            selected.append((winner, winner.implementation, tuple(sorted(winner.parameters))))
            continue
        # Suppressed winners cannot protect against inherited rejected recipes.
        # Preserve conclusive negative evidence in a negative-only overlay.
        negative = next(
            (
                item
                for item in items
                if item.kernel_valid is False
                or (
                    item.kernel_valid is True
                    and complete_metrics(item, objective)
                    and not candidate_wins(item, objective)
                )
            ),
            None,
        )
        if negative is not None:
            selected.append((negative, "reference", ()))
    # Stable ordering keeps regenerated policies and effective identities deterministic.
    selected.sort(
        key=lambda row: (
            row[0].family,
            row[0].direction,
            row[0].phase,
            row[0].spatial_shape,
            row[0].in_channels,
            row[0].out_channels,
            row[0].dtype,
            row[0].checkpointing,
            row[0].kernel_size,
            row[0].workload.get("model_family", ""),
            row[0].workload.get("variant", ""),
            row[0].batch,
        )
    )
    rules = []
    for item, implementation, parameters in selected:
        when = {
            "family": item.family,
            "direction": item.direction,
            "batch": item.batch,
            "spatial_shape": list(item.spatial_shape),
            "channels": [item.in_channels, item.out_channels],
            "kernel_size": [item.kernel_size] * 3,
            "stride": [2 if item.direction in ("downsample", "transpose") else 1] * 3,
            "dtype": item.dtype,
            "checkpointing": item.checkpointing,
        }
        for name_key in ("model_family", "variant"):
            if name_key in item.workload:
                when[name_key] = item.workload[name_key]
        selection = {"implementation": implementation}
        if parameters:
            selection["parameters"] = dict(parameters)
        rules.append(
            {
                "id": f"generated-{len(rules):04d}",
                "when": when,
                "use": {item.phase: selection},
                "confidence": "measured-exact-context",
            }
        )
    return parse_policy(
        {
            "version": 2,
            "kind": "mednext-accel-policy",
            "name": name,
            "target": {"vendor": "nvidia", "sm": list(sm)},
            "rules": rules,
        }
    )
