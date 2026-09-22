"""Performance/memory comparison of effective policies, not numerical validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from ..optimization.descriptors import ExecutionContext
from ..optimization.policies import PolicyRegistry
from ..optimization.policy import OptimizationPolicy, policy_to_primitive
from .evidence import ModelComparisonEvidence, ModelProbeEvidence, model_acceptance


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
    native = ModelProbeEvidence.from_result(reference, seed=seed)
    proposed = ModelProbeEvidence.from_result(candidate, seed=seed)
    performance, memory, reason = model_acceptance(objective, native, proposed, seed)
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
        seed=seed,
    )
