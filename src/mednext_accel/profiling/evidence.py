"""Immutable measurements, independent from runtime policy acceptance."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .synthesize import Measurement


MODEL_PROTOCOL = "seed-before-initialization-and-input;adamw;bf16;mean-square-loss;1-warmup;1-step"
OBJECTIVES = ("balanced", "throughput", "memory")


def require_bool(value, path: str, *, nullable: bool = False) -> None:
    if type(value) is not bool and not (nullable and value is None):
        raise ValueError(f"{path} must be a boolean" + (" or null" if nullable else ""))


def require_int(value, path: str, *, minimum: int = 0, maximum: int | None = None) -> None:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{path} must be an integer in the supported range")


def require_string(value, path: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a nonempty string" + (" or null" if nullable else ""))


def require_number(value, path: str) -> None:
    if value is not None and type(value) not in (int, float):
        raise ValueError(f"{path} must be a number or null")


def require_mapping(value, path: str) -> None:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be a mapping with string keys")


def require_sequence(value, path: str) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be an array")


def record_fields(data, expected: set[str], path: str, *, required: set[str] | None = None):
    require_mapping(data, path)
    unknown = set(data) - expected
    if unknown:
        raise ValueError(f"{path}: unknown field(s): {', '.join(sorted(unknown))}")
    missing = (expected if required is None else required) - set(data)
    if missing:
        raise ValueError(f"{path}: missing field(s): {', '.join(sorted(missing))}")
    return data


def validate_json(value, path: str = "evidence") -> None:
    """Reject non-JSON values before a loader could normalize or coerce them."""
    if isinstance(value, Mapping):
        require_mapping(value, path)
        for key, item in value.items():
            validate_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_json(item, f"{path}[{index}]")
    elif type(value) not in (str, bool, int, float, type(None)):
        raise ValueError(f"{path} must contain JSON values")
    elif type(value) is float and not isfinite(value):
        raise ValueError(f"{path} must use null for a nonfinite observation")


WORKLOAD_FIELDS = {
    "model_family",
    "variant",
    "spatial",
    "dtypes",
    "checkpointing",
    "in_channels",
    "out_channels",
}


def validate_workload(value, *, complete: bool = False) -> None:
    record_fields(
        value, WORKLOAD_FIELDS, "workload", required=WORKLOAD_FIELDS if complete else set()
    )
    for name in ("model_family", "variant", "checkpointing"):
        if name in value:
            require_string(value[name], f"workload.{name}")
    for name in ("in_channels", "out_channels"):
        if name in value:
            require_int(value[name], f"workload.{name}", minimum=1)
    if "spatial" in value:
        require_sequence(value["spatial"], "workload.spatial")
        if len(value["spatial"]) != 3:
            raise ValueError("workload.spatial must have three dimensions")
        for size in value["spatial"]:
            require_int(size, "workload.spatial dimension", minimum=1)
    if "dtypes" in value:
        require_sequence(value["dtypes"], "workload.dtypes")
        if not value["dtypes"]:
            raise ValueError("workload.dtypes must not be empty")
        for dtype in value["dtypes"]:
            require_string(dtype, "workload.dtype")


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
        record_fields(
            self.data,
            {"collected_at", "platform", "python", "driver", "gpu", "software"},
            "environment",
            required=set(),
        )
        for name in ("collected_at", "platform", "python", "driver"):
            if name in self.data:
                require_string(self.data[name], f"environment.{name}", nullable=name == "driver")
        if "gpu" in self.data:
            gpu = self.data["gpu"]
            record_fields(
                gpu,
                {"index", "name", "sm", "total_memory_bytes"},
                "environment.gpu",
                required=set(),
            )
            for name in ("index", "total_memory_bytes"):
                if name in gpu:
                    require_int(gpu[name], f"gpu.{name}")
            if "name" in gpu:
                require_string(gpu["name"], "gpu.name")
            if "sm" in gpu:
                require_sequence(gpu["sm"], "gpu.sm")
                if len(gpu["sm"]) != 2:
                    raise ValueError("gpu.sm must contain two integers")
                for number in gpu["sm"]:
                    require_int(number, "gpu.sm")
        if "software" in self.data:
            software = self.data["software"]
            record_fields(
                software,
                {"mednext_accel", "torch", "cuda", "cudnn", "triton"},
                "environment.software",
                required=set(),
            )
            for name, version in software.items():
                if name == "cudnn":
                    if version is not None:
                        require_int(version, "software.cudnn")
                else:
                    require_string(version, f"software.{name}", nullable=True)
        object.__setattr__(self, "data", freeze(self.data))
        validate_json(self.data)


@dataclass(frozen=True, slots=True)
class CampaignEvidence:
    data: Mapping[str, object]

    def __post_init__(self):
        record_fields(
            self.data,
            {"phase", "seed", "preset", "objective", "compile_mode", "batch_search", "workloads"},
            "campaign",
        )
        require_string(self.data["preset"], "campaign.preset")
        require_string(self.data["compile_mode"], "campaign.compile_mode")
        search = self.data["batch_search"]
        record_fields(search, {"memory_fraction", "maximum"}, "campaign.batch_search")
        require_number(search["memory_fraction"], "memory_fraction")
        if search["memory_fraction"] is None or not 0 < search["memory_fraction"] <= 1:
            raise ValueError("memory_fraction must be in (0, 1]")
        if search["maximum"] is not None:
            require_int(search["maximum"], "maximum", minimum=1)
        require_sequence(self.data["workloads"], "campaign.workloads")
        for workload in self.data["workloads"]:
            validate_workload(workload, complete=True)
        if self.data.get("phase") != "training":
            raise ValueError("campaign.phase must be training")
        require_int(self.data.get("seed"), "campaign.seed", maximum=2**63 - 1)
        if self.data.get("objective") not in OBJECTIVES:
            raise ValueError("campaign.objective is unsupported")
        object.__setattr__(self, "data", freeze(self.data))
        validate_json(self.data)


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
        require_string(self.status, "status")
        require_int(self.seed, "seed", maximum=2**63 - 1)
        require_string(self.message, "message", nullable=True)
        require_string(self.failure_stage, "failure_stage", nullable=True)
        require_mapping(self.diagnostics, "diagnostics")
        require_number(self.step_ms, "step_ms")
        require_number(self.peak_bytes, "peak_bytes")
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))
        validate_json(self.diagnostics)
        object.__setattr__(self, "step_ms", freeze(self.step_ms))
        object.__setattr__(self, "peak_bytes", freeze(self.peak_bytes))

    @classmethod
    def from_result(cls, result: Mapping[str, object], *, seed: int):
        record_fields(
            result,
            {
                "status",
                "seed",
                "step_ms",
                "peak_bytes",
                "message",
                "failure_stage",
                "diagnostics",
                "loss",
            },
            "model result",
            required=set(),
        )
        if "diagnostics" in result and "loss" in result:
            raise ValueError("model result must not supply both diagnostics and legacy loss")
        return cls(
            status=result.get("status", "missing"),
            seed=result.get("seed", seed),
            step_ms=result.get("step_ms"),
            peak_bytes=result.get("peak_bytes"),
            message=result.get("message"),
            failure_stage=result.get("failure_stage"),
            diagnostics=result.get(
                "diagnostics", {"loss": result["loss"]} if "loss" in result else {}
            ),
        )


def model_probe_failure(side: str, result: ModelProbeEvidence) -> str | None:
    if result.status != "ok":
        return f"{side}: status {result.status}"
    for name in ("step_ms", "peak_bytes"):
        value = getattr(result, name)
        if type(value) not in (int, float) or not isfinite(value) or value <= 0:
            return f"{side}: invalid {name}"
    return None


def model_acceptance(
    objective: str, reference: ModelProbeEvidence, candidate: ModelProbeEvidence, seed: int
) -> tuple[bool, bool, str]:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown profiling objective {objective!r}")
    failure = model_probe_failure("reference", reference) or model_probe_failure(
        "candidate", candidate
    )
    if reference.seed != seed or candidate.seed != seed:
        failure = failure or "reference/candidate seed mismatch"
    performance = memory = False
    if failure is None:
        if objective == "memory":
            performance = candidate.step_ms <= reference.step_ms * 1.10
            memory = candidate.peak_bytes < reference.peak_bytes
        else:
            performance = candidate.step_ms < reference.step_ms
            memory = candidate.peak_bytes <= reference.peak_bytes * (
                1.25 if objective == "throughput" else 1.15
            )
    reason = failure or (
        "accepted"
        if performance and memory
        else "performance rejected"
        if not performance
        else "memory rejected"
    )
    return performance, memory, reason


@dataclass(frozen=True, slots=True)
class BatchProbeEvidence:
    batch: int
    result: ModelProbeEvidence
    within_budget: bool
    feasible: bool
    reason: str | None = None

    def __post_init__(self):
        require_int(self.batch, "batch", minimum=1)
        if not isinstance(self.result, ModelProbeEvidence):
            raise ValueError("batch result must be ModelProbeEvidence")
        require_bool(self.within_budget, "within_budget")
        require_bool(self.feasible, "feasible")
        require_string(self.reason, "reason", nullable=True)


@dataclass(frozen=True, slots=True)
class BatchSearchEvidence:
    workload: Mapping[str, object]
    budget_bytes: int
    attempts: tuple[BatchProbeEvidence, ...]
    selected_maximum: int

    def __post_init__(self):
        validate_workload(self.workload)
        require_int(self.budget_bytes, "budget_bytes")
        require_int(self.selected_maximum, "selected_maximum")
        require_sequence(self.attempts, "attempts")
        seen = set()
        for attempt in self.attempts:
            if not isinstance(attempt, BatchProbeEvidence):
                raise ValueError("attempts must contain BatchProbeEvidence records")
            if attempt.batch in seen:
                raise ValueError("duplicate batch in ordered attempts")
            seen.add(attempt.batch)
            peak = attempt.result.peak_bytes
            within_budget = type(peak) in (int, float) and 0 < peak <= self.budget_bytes
            failure = model_probe_failure("probe", attempt.result)
            reason = failure or (None if within_budget else "memory_fraction")
            if (
                attempt.within_budget is not within_budget
                or attempt.feasible is not (reason is None)
                or attempt.reason != reason
            ):
                raise ValueError(
                    "batch attempt budget, feasibility or reason contradicts its result"
                )
        maximum = max((attempt.batch for attempt in self.attempts if attempt.feasible), default=0)
        if self.selected_maximum != maximum:
            raise ValueError("selected_maximum must be the largest feasible attempted batch")
        object.__setattr__(self, "workload", freeze(self.workload))
        object.__setattr__(self, "attempts", tuple(self.attempts))
        validate_json(self.workload)

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
    seed: int = 0
    numerical_equivalence: str = "not-measured"
    protocol: str = MODEL_PROTOCOL

    def __post_init__(self):
        validate_workload(self.workload)
        require_int(self.batch, "batch", minimum=1)
        require_int(self.seed, "seed", maximum=2**63 - 1)
        if not isinstance(self.reference, ModelProbeEvidence) or not isinstance(
            self.candidate, ModelProbeEvidence
        ):
            raise ValueError("comparison sides must be ModelProbeEvidence records")
        for name in ("performance_accepted", "memory_accepted", "policy_accepted"):
            require_bool(getattr(self, name), name)
        if self.numerical_equivalence != "not-measured":
            raise ValueError("numerical_equivalence must be exactly not-measured")
        if self.protocol != MODEL_PROTOCOL:
            raise ValueError("unsupported model comparison protocol")
        performance, memory, reason = model_acceptance(
            self.objective, self.reference, self.candidate, self.seed
        )
        if (self.performance_accepted, self.memory_accepted, self.policy_accepted, self.reason) != (
            performance,
            memory,
            performance and memory,
            reason,
        ):
            raise ValueError(
                "model comparison gates, final acceptance or reason contradicts its metrics"
            )
        require_sequence(self.effective_policy, "effective_policy")
        for layer in self.effective_policy:
            record_fields(layer, {"name", "sha256"}, "effective_policy layer")
            require_string(layer["name"], "policy name")
            digest = layer["sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError("effective_policy sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "workload", freeze(self.workload))
        object.__setattr__(self, "effective_policy", freeze(self.effective_policy))
        validate_json(self.workload)

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
        from .synthesize import Measurement

        if not isinstance(self.environment, EnvironmentEvidence) or not isinstance(
            self.campaign, CampaignEvidence
        ):
            raise ValueError("environment/campaign must be immutable evidence records")
        for name, record_type in (
            ("batch_searches", BatchSearchEvidence),
            ("kernel_measurements", Measurement),
            ("model_comparisons", ModelComparisonEvidence),
        ):
            records = getattr(self, name)
            require_sequence(records, name)
            if any(not isinstance(item, record_type) for item in records):
                raise ValueError(f"{name} contains an invalid record")
            object.__setattr__(self, name, tuple(records))
        for comparison in self.model_comparisons:
            if (
                comparison.seed != self.campaign.data["seed"]
                or comparison.objective != self.campaign.data["objective"]
            ):
                raise ValueError("model comparison seed/objective contradicts campaign")
        require_mapping(self.execution, "execution")
        for name, count in self.execution.items():
            require_int(count, f"execution.{name}")
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

        validate_json(data)
        record_fields(
            data,
            {
                "version",
                "kind",
                "environment",
                "campaign",
                "batch_searches",
                "kernel_measurements",
                "model_comparisons",
                "execution",
            },
            "evidence",
        )
        if (
            type(data["version"]) is not int
            or data["version"] != 1
            or data["kind"] != "mednext-accel-evidence"
        ):
            raise ValueError("expected mednext-accel-evidence version 1")

        def load_record(record_type, value, **updates):
            record_fields(value, {item.name for item in fields(record_type)}, record_type.__name__)
            return record_type(**{**value, **updates})

        for name in ("batch_searches", "kernel_measurements", "model_comparisons"):
            require_sequence(data[name], name)
        searches = []
        for item in data["batch_searches"]:
            record_fields(
                item,
                {field.name for field in fields(BatchSearchEvidence)} | {"by_batch"},
                "batch search",
            )
            require_sequence(item["attempts"], "attempts")
            attempts = []
            for attempt in item["attempts"]:
                record_fields(
                    attempt, {field.name for field in fields(BatchProbeEvidence)}, "batch attempt"
                )
                attempts.append(
                    load_record(
                        BatchProbeEvidence,
                        attempt,
                        result=load_record(ModelProbeEvidence, attempt["result"]),
                    )
                )
            search = BatchSearchEvidence(
                item["workload"], item["budget_bytes"], tuple(attempts), item["selected_maximum"]
            )
            # Canonical JSON comparison distinguishes booleans from integers.
            if json.dumps(item["by_batch"], sort_keys=True) != json.dumps(
                search.to_primitive()["by_batch"], sort_keys=True
            ):
                raise ValueError("by_batch must exactly match the ordered attempts")
            searches.append(search)
        comparisons = []
        for item in data["model_comparisons"]:
            record_fields(
                item, {field.name for field in fields(ModelComparisonEvidence)}, "model comparison"
            )
            comparisons.append(
                load_record(
                    ModelComparisonEvidence,
                    item,
                    reference=load_record(ModelProbeEvidence, item["reference"]),
                    candidate=load_record(ModelProbeEvidence, item["candidate"]),
                )
            )
        return cls(
            EnvironmentEvidence(data["environment"]),
            CampaignEvidence(data["campaign"]),
            tuple(searches),
            tuple(load_record(Measurement, item) for item in data["kernel_measurements"]),
            tuple(comparisons),
            data["execution"],
        )
