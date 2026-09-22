"""Performance/memory comparison of effective policies, not numerical validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from math import isfinite

from ..optimization.descriptors import ExecutionContext
from ..optimization.policies import PolicyRegistry
from ..optimization.policy import OptimizationPolicy, policy_to_primitive
from .evidence import ModelComparisonEvidence, ModelProbeEvidence


def effective_policy_identity(policy: OptimizationPolicy, context: ExecutionContext):
    """Identify the actual ordered external/exact-SM/shared runtime layers."""
    resolver = PolicyRegistry(external=policy_to_primitive(policy)).resolver()
    return tuple(
        {
            "name": layer.name,
            "sha256": hashlib.sha256(
                json.dumps(
                    policy_to_primitive(layer), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        }
        for layer in resolver.context_layers(context)
    )


def model_probe_failure(side: str, result: ModelProbeEvidence) -> str | None:
    if result.status != "ok":
        return f"{side}: status {result.status}"
    for name in ("step_ms", "peak_bytes"):
        value = getattr(result, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
        ):
            return f"{side}: invalid {name}"
    return None


def compare_model_results(
    objective: str,
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    workload: Mapping[str, object],
    batch: int,
    seed: int,
    effective_policy: tuple[Mapping[str, object], ...] = (),
) -> ModelComparisonEvidence:
    if objective not in ("balanced", "throughput", "memory"):
        raise ValueError(f"unknown profiling objective {objective!r}")
    native = ModelProbeEvidence.from_result(reference, seed=seed)
    proposed = ModelProbeEvidence.from_result(candidate, seed=seed)
    failure = model_probe_failure("reference", native) or model_probe_failure("candidate", proposed)
    if native.seed != seed or proposed.seed != seed:
        failure = failure or "reference/candidate seed mismatch"
    performance = memory = False
    if failure is None:
        if objective == "memory":
            performance = proposed.step_ms <= native.step_ms * 1.10
            memory = proposed.peak_bytes < native.peak_bytes
        else:
            performance = proposed.step_ms < native.step_ms
            memory = proposed.peak_bytes <= native.peak_bytes * (
                1.25 if objective == "throughput" else 1.15
            )
    reason = failure or (
        "accepted"
        if performance and memory
        else "performance rejected"
        if not performance
        else "memory rejected"
    )
    return ModelComparisonEvidence(
        workload,
        batch,
        objective,
        native,
        proposed,
        effective_policy,
        performance,
        memory,
        performance and memory,
        reason,
    )
