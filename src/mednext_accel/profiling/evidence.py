"""Immutable measurements, independent from runtime policy acceptance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .synthesize import Measurement


def freeze(value):
    """Copy JSON-shaped data so callers cannot mutate recorded evidence."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def primitive(value):
    """Make strict JSON data; nonfinite observations are represented as null."""
    if is_dataclass(value):
        return {item.name: primitive(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [primitive(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


@dataclass(frozen=True, slots=True)
class EnvironmentEvidence:
    data: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "data", freeze(self.data))


@dataclass(frozen=True, slots=True)
class CampaignEvidence:
    data: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "data", freeze(self.data))


@dataclass(frozen=True, slots=True)
class ModelProbeEvidence:
    status: str
    seed: int
    step_ms: float | None = None
    peak_bytes: int | float | None = None
    message: str | None = None
    failure_stage: str | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))
        object.__setattr__(self, "step_ms", freeze(self.step_ms))
        object.__setattr__(self, "peak_bytes", freeze(self.peak_bytes))

    @classmethod
    def from_result(cls, result: Mapping[str, object], *, seed: int):
        return cls(
            status=str(result.get("status", "missing")),
            seed=int(result.get("seed", seed)),
            step_ms=result.get("step_ms"),
            peak_bytes=result.get("peak_bytes"),
            message=result.get("message"),
            failure_stage=result.get("failure_stage"),
            diagnostics=result.get(
                "diagnostics", {"loss": result["loss"]} if "loss" in result else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class BatchProbeEvidence:
    batch: int
    result: ModelProbeEvidence
    within_budget: bool
    feasible: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BatchSearchEvidence:
    workload: Mapping[str, object]
    budget_bytes: int
    attempts: tuple[BatchProbeEvidence, ...]
    selected_maximum: int

    def __post_init__(self):
        object.__setattr__(self, "workload", freeze(self.workload))
        object.__setattr__(self, "attempts", tuple(self.attempts))

    @property
    def by_batch(self) -> Mapping[int, BatchProbeEvidence]:
        return MappingProxyType({item.batch: item for item in self.attempts})

    def to_primitive(self):
        return {
            **primitive(self),
            "by_batch": {str(key): primitive(value) for key, value in self.by_batch.items()},
        }


@dataclass(frozen=True, slots=True)
class ModelComparisonEvidence:
    workload: Mapping[str, object]
    batch: int
    objective: str
    reference: ModelProbeEvidence
    candidate: ModelProbeEvidence
    effective_policy: tuple[Mapping[str, object], ...]
    performance_accepted: bool
    memory_accepted: bool
    policy_accepted: bool
    reason: str
    numerical_equivalence: str = "not-measured"
    protocol: str = (
        "seed-before-initialization-and-input;adamw;bf16;mean-square-loss;1-warmup;1-step"
    )

    def __post_init__(self):
        object.__setattr__(self, "workload", freeze(self.workload))
        object.__setattr__(self, "effective_policy", freeze(self.effective_policy))

    def to_primitive(self):
        return primitive(self)


@dataclass(frozen=True, slots=True)
class ProfilingEvidence:
    environment: EnvironmentEvidence
    campaign: CampaignEvidence
    batch_searches: tuple[BatchSearchEvidence, ...] = ()
    kernel_measurements: tuple[Measurement, ...] = ()
    model_comparisons: tuple[ModelComparisonEvidence, ...] = ()
    execution: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("batch_searches", "kernel_measurements", "model_comparisons"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "execution", freeze(self.execution))

    def to_primitive(self) -> dict[str, object]:
        return {
            "version": 1,
            "kind": "mednext-accel-evidence",
            "environment": primitive(self.environment.data),
            "campaign": primitive(self.campaign.data),
            "batch_searches": [item.to_primitive() for item in self.batch_searches],
            "kernel_measurements": [item.to_primitive() for item in self.kernel_measurements],
            "model_comparisons": [item.to_primitive() for item in self.model_comparisons],
            "execution": primitive(self.execution),
        }

    @classmethod
    def from_primitive(cls, data: Mapping[str, object]) -> ProfilingEvidence:
        from .synthesize import Measurement

        if data.get("version") != 1 or data.get("kind") != "mednext-accel-evidence":
            raise ValueError("expected mednext-accel-evidence version 1")
        searches = []
        for item in data.get("batch_searches", ()):
            attempts = tuple(
                BatchProbeEvidence(**{**probe, "result": ModelProbeEvidence(**probe["result"])})
                for probe in item["attempts"]
            )
            searches.append(
                BatchSearchEvidence(
                    item["workload"], item["budget_bytes"], attempts, item["selected_maximum"]
                )
            )
        comparisons = tuple(
            ModelComparisonEvidence(
                **{
                    **item,
                    "reference": ModelProbeEvidence(**item["reference"]),
                    "candidate": ModelProbeEvidence(**item["candidate"]),
                }
            )
            for item in data.get("model_comparisons", ())
        )
        return cls(
            EnvironmentEvidence(data["environment"]),
            CampaignEvidence(data["campaign"]),
            tuple(searches),
            tuple(Measurement(**item) for item in data.get("kernel_measurements", ())),
            comparisons,
            data.get("execution", {}),
        )
