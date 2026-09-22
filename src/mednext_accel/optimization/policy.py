"""Strict, immutable policy-v2 records and their portable representation.

Evidence is descriptive provenance only. Rules have no implicit defaults: an
omitted phase falls through, whereas an explicit reference selection terminates
resolution. Arithmetic belongs in :mod:`parameters`, never in documents.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal

from .implementations import ImplementationRegistry, default_implementation_registry
from .parameters import has_parameter_recipe, validate_parameters

Confidence = Literal[
    "measured-exact-context",
    "interpolated-bounded",
    "inferred-same-sm",
    "validated-cross-sm",
    "extrapolated",
    "guard",
]
_CONFIDENCES = {
    "measured-exact-context",
    "interpolated-bounded",
    "inferred-same-sm",
    "validated-cross-sm",
    "extrapolated",
    "guard",
}
_PHASES = {
    "training",
    "inference",
    "export",
    "forward",
    "backward_input",
    "backward_weight",
    "backward_bias",
}
_STRINGS = {"family", "direction", "role", "dtype", "model_family", "variant", "checkpointing"}
_TUPLES = {"kernel_size": 3, "stride": 3, "spatial_shape": 3, "channels": 2}
_NUMBERS = {"batch", "spatial_volume", "work", "reduction_work", "total_vram_gib"}


@dataclass(frozen=True, slots=True)
class NumericRange:
    minimum: int | float | None = None
    maximum: int | float | None = None

    def contains(self, value: int | float) -> bool:
        return (self.minimum is None or value >= self.minimum) and (
            self.maximum is None or value <= self.maximum
        )


Condition = str | tuple[int, ...] | NumericRange
Parameters = Mapping[str, str | int | float | bool | None]


@dataclass(frozen=True, slots=True)
class PolicySelection:
    implementation: str
    parameters: Parameters | Literal["auto"] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.parameters, Mapping):
            object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class PolicyRule:
    identifier: str
    when: Mapping[str, Condition]
    use: Mapping[str, PolicySelection]
    confidence: Confidence
    declared_when: Mapping[str, Condition]

    def __post_init__(self) -> None:
        for field in ("when", "use", "declared_when"):
            object.__setattr__(self, field, MappingProxyType(dict(getattr(self, field))))


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    identifier: str
    sha256: str
    file: str | None = None


@dataclass(frozen=True, slots=True)
class OptimizationPolicy:
    name: str
    vendor: str
    target_sm: tuple[int, int] | None
    scope: Mapping[str, Condition]
    rules: tuple[PolicyRule, ...]
    evidence: tuple[EvidenceReference, ...] = ()
    version: int = 2
    kind: str = "mednext-accel-policy"

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", MappingProxyType(dict(self.scope)))
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if self.target_sm is not None:
            object.__setattr__(self, "target_sm", tuple(self.target_sm))


def _mapping(value: object, path: str, allowed: set[str] | None = None) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} keys must be strings")
    if allowed is not None and (unknown := set(value) - allowed):
        raise ValueError(f"{path}: unknown fields {', '.join(sorted(unknown))}")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _integers(value: object, length: int, path: str, minimum: int = 1) -> tuple[int, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(type(item) is not int or item < minimum for item in value)
    ):
        raise ValueError(f"{path} must contain {length} integers >= {minimum}")
    return tuple(value)


def _range(value: object, path: str, *, integer: bool) -> NumericRange:
    if isinstance(value, Mapping):
        data = _mapping(value, path, {"min", "max"})
        if not data:
            raise ValueError(f"{path} range must supply min or max")
    else:
        data = {"min": value, "max": value}
    for bound, item in data.items():
        numeric = type(item) is int if integer else type(item) in (int, float)
        if not numeric or not math.isfinite(item) or item < (1 if integer else 0):
            kind = "positive integer" if integer else "nonnegative number"
            raise ValueError(f"{path}.{bound} must be a finite {kind}")
    low, high = data.get("min"), data.get("max")
    if low is not None and high is not None and low > high:
        raise ValueError(f"{path}: min must not exceed max")
    return NumericRange(low, high)


def _conditions(value: object, path: str) -> dict[str, Condition]:
    data = _mapping(value, path, _STRINGS | _TUPLES.keys() | _NUMBERS)
    result = {}
    for name, item in data.items():
        location = f"{path}.{name}"
        if name in _STRINGS:
            result[name] = _string(item, location)
        elif name in _TUPLES:
            result[name] = _integers(item, _TUPLES[name], location)
        else:
            result[name] = _range(item, location, integer=name != "total_vram_gib")
    return result


def _normalize(scope: Mapping[str, Condition], when: Mapping[str, Condition], path: str) -> dict:
    result = dict(scope)
    for name, value in when.items():
        inherited = scope.get(name)
        if isinstance(inherited, NumericRange) and isinstance(value, NumericRange):
            if (
                inherited.minimum is not None
                and value.minimum is not None
                and value.minimum < inherited.minimum
            ) or (
                inherited.maximum is not None
                and value.maximum is not None
                and value.maximum > inherited.maximum
            ):
                raise ValueError(f"{path}.{name} cannot widen scope")
            low = inherited.minimum if value.minimum is None else value.minimum
            high = inherited.maximum if value.maximum is None else value.maximum
            if low is not None and high is not None and low > high:
                raise ValueError(f"{path}.{name} contradicts scope")
            value = NumericRange(low, high)
        elif inherited is not None and inherited != value:
            raise ValueError(f"{path}.{name} contradicts scope")
        result[name] = value
    return result


def _selection(value: object, path: str, registry: ImplementationRegistry) -> PolicySelection:
    data = _mapping(value, path, {"implementation", "parameters"})
    implementation = _string(data.get("implementation"), f"{path}.implementation")
    try:
        registry.resolve_metadata(implementation)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    parameters = data.get("parameters")
    if "parameters" in data:
        if parameters == "auto":
            if not has_parameter_recipe(implementation):
                raise ValueError(f"{path}: {implementation!r} has no parameter recipe for auto")
        else:
            parameters = _mapping(parameters, f"{path}.parameters")
            try:
                validate_parameters(implementation, parameters)
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from error
    return PolicySelection(implementation, parameters)


def _sequence(value: object, path: str) -> list | tuple:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a sequence")
    return value


def _evidence(value: object, path: str) -> EvidenceReference:
    data = _mapping(value, path, {"id", "sha256", "file"})
    identifier = _string(data.get("id"), f"{path}.id")
    digest = data.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{path}.sha256 must be a lowercase SHA-256 hex digest")
    filename = data.get("file")
    if "file" in data:
        filename = _string(filename, f"{path}.file")
        if PurePosixPath(filename).name != filename or filename in (".", "..") or "\\" in filename:
            raise ValueError(f"{path}.file must be a relative evidence filename")
    return EvidenceReference(identifier, digest, filename)


def parse_policy(
    value: Mapping[str, object], *, registry: ImplementationRegistry | None = None
) -> OptimizationPolicy:
    """Normalize and validate a v2 document without importing CUDA backends."""
    value = _mapping(value, "policy document")
    if value.get("kind") == "mednext-accel-evidence":
        raise ValueError("profiling evidence is not a runtime policy; supply its policy YAML")
    if "schema_version" in value or (type(value.get("version")) is int and value["version"] == 1):
        raise ValueError("schema-v1 profiles are unsupported; migrate to policy v2")
    _mapping(value, "policy", {"version", "kind", "name", "target", "scope", "evidence", "rules"})
    if type(value.get("version")) is not int or value["version"] != 2:
        raise ValueError("policy.version must be 2")
    if value.get("kind") != "mednext-accel-policy":
        raise ValueError("policy.kind must be mednext-accel-policy")
    name = _string(value.get("name"), "policy.name")
    target = _mapping(value.get("target"), "target", {"vendor", "sm"})
    vendor = _string(target.get("vendor"), "target.vendor")
    sm = _integers(target["sm"], 2, "target.sm", minimum=0) if "sm" in target else None
    scope = _conditions(value.get("scope", {}), "scope")
    registry = registry or default_implementation_registry()
    rules = []
    for index, raw in enumerate(_sequence(value.get("rules"), "rules")):
        path = f"rules[{index}]"
        data = _mapping(raw, path, {"id", "when", "use", "confidence"})
        identifier = _string(data.get("id"), f"{path}.id")
        declared = _conditions(data.get("when", {}), f"{path}.when")
        when = _normalize(scope, declared, f"{path}.when")
        uses = _mapping(data.get("use"), f"{path}.use", _PHASES)
        if not uses:
            raise ValueError(f"{path}.use must contain a phase selection")
        selections = {
            phase: _selection(item, f"{path}.use.{phase}", registry) for phase, item in uses.items()
        }
        confidence = data.get("confidence")
        if not isinstance(confidence, str) or confidence not in _CONFIDENCES:
            raise ValueError(f"{path}.confidence must be one of {sorted(_CONFIDENCES)}")
        rules.append(PolicyRule(identifier, when, selections, confidence, declared))
    if len({rule.identifier for rule in rules}) != len(rules):
        raise ValueError("duplicate rule id")
    evidence = tuple(
        _evidence(item, f"evidence[{index}]")
        for index, item in enumerate(_sequence(value.get("evidence", []), "evidence"))
    )
    if len({item.identifier for item in evidence}) != len(evidence):
        raise ValueError("duplicate evidence id")
    return OptimizationPolicy(name, vendor, sm, scope, tuple(rules), evidence)


def policy_to_primitive(policy: OptimizationPolicy) -> dict[str, object]:
    """Return portable data, preserving omission versus explicit parameters."""

    def conditions(items):
        result = {}
        for name, item in items.items():
            if isinstance(item, NumericRange):
                result[name] = (
                    item.minimum
                    if item.minimum == item.maximum
                    else {
                        key: bound
                        for key, bound in (("min", item.minimum), ("max", item.maximum))
                        if bound is not None
                    }
                )
            else:
                result[name] = list(item) if isinstance(item, tuple) else item
        return result

    def selection(item):
        result = {"implementation": item.implementation}
        if item.parameters is not None:
            result["parameters"] = (
                dict(item.parameters) if isinstance(item.parameters, Mapping) else item.parameters
            )
        return result

    target = {"vendor": policy.vendor}
    if policy.target_sm is not None:
        target["sm"] = list(policy.target_sm)
    result = {"version": 2, "kind": "mednext-accel-policy", "name": policy.name, "target": target}
    if policy.scope:
        result["scope"] = conditions(policy.scope)
    if policy.evidence:
        result["evidence"] = [
            {
                "id": item.identifier,
                "sha256": item.sha256,
                **({"file": item.file} if item.file is not None else {}),
            }
            for item in policy.evidence
        ]
    result["rules"] = [
        {
            "id": rule.identifier,
            "when": conditions(rule.declared_when),
            "use": {phase: selection(item) for phase, item in rule.use.items()},
            "confidence": rule.confidence,
        }
        for rule in policy.rules
    ]
    return result
