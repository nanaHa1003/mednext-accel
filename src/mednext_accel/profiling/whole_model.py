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
    """Identify ordered runtime layers without their descriptive evidence links.

    Excluding provenance avoids a circular hash when the published policy links
    back to the evidence containing this comparison. Every runtime field remains
    covered, so provisional and published selections have the same identity.
    """
    resolver = PolicyRegistry(external=policy_to_primitive(policy)).resolver()
    return tuple(
        {
            "name": layer.name,
            "sha256": hashlib.sha256(
                json.dumps(
                    {
                        key: value
                        for key, value in policy_to_primitive(layer).items()
                        if key != "evidence"
                    },
                    sort_keys=True,
                    separators=(",", ":"),
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
