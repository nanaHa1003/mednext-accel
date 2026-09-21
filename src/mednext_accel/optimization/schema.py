"""Versioned, immutable optimization-profile schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .implementations import ImplementationRegistry, default_implementation_registry

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
FrozenJsonMapping = Mapping[str, JsonValue]


def _freeze(value: Any) -> JsonValue:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"profile contains non-JSON value {value!r}")


@dataclass(frozen=True, slots=True)
class NumericRange:
    minimum: int | float | None = None
    maximum: int | float | None = None

    def __post_init__(self) -> None:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("range min must not exceed max")

    def contains(self, value: int | float) -> bool:
        return (self.minimum is None or value >= self.minimum) and (
            self.maximum is None or value <= self.maximum
        )


@dataclass(frozen=True, slots=True)
class ProfileSelection:
    implementation: str
    parameters: FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class ProfileRule:
    identifier: str
    family: str
    direction: str | None
    phases: Mapping[str, ProfileSelection]
    batch: NumericRange
    spatial_volume: NumericRange
    in_channels: int | None
    out_channels: int | None
    dtype: str | None
    total_vram_gib: NumericRange
    confidence: str
    spatial_shape: tuple[int, int, int] | None = None
    model_family: str | None = None
    variant: str | None = None
    checkpointing: str | None = None


@dataclass(frozen=True, slots=True)
class ProfileOverride(ProfileRule):
    pass


@dataclass(frozen=True, slots=True)
class OptimizationProfile:
    schema_version: int
    name: str
    vendor: str
    target_sm: tuple[int, int] | None
    provenance: FrozenJsonMapping
    defaults: Mapping[str, Mapping[str, ProfileSelection]]
    rules: tuple[ProfileRule, ...]
    overrides: tuple[ProfileOverride, ...]
    measurements: JsonValue = None


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _selection(
    value: object, path: str, registry: ImplementationRegistry
) -> ProfileSelection:
    data = _mapping(value, path)
    implementation = data.get("implementation")
    if not isinstance(implementation, str):
        raise ValueError(f"{path}.implementation must be a string")
    try:
        registry.resolve_metadata(implementation)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    parameters = _mapping(data.get("parameters", {}), f"{path}.parameters")
    return ProfileSelection(implementation, _freeze(parameters))  # type: ignore[arg-type]


def _range(value: object, path: str) -> NumericRange:
    if value is None:
        return NumericRange()
    data = _mapping(value, path)
    minimum, maximum = data.get("min"), data.get("max")
    for name, item in (("min", minimum), ("max", maximum)):
        if item is not None and not isinstance(item, (int, float)):
            raise ValueError(f"{path}.{name} must be numeric")
    try:
        return NumericRange(minimum, maximum)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _rule(
    value: object,
    path: str,
    registry: ImplementationRegistry,
    *,
    override: bool,
) -> ProfileRule:
    data = _mapping(value, path)
    identifier = data.get("id")
    family = data.get("family")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"{path}.id must be a non-empty string")
    if not isinstance(family, str) or not family:
        raise ValueError(f"{path}.family must be a non-empty string")
    match = _mapping(data.get("match", {}), f"{path}.match")
    raw_phases = _mapping(data.get("phases"), f"{path}.phases")
    phases = MappingProxyType({
        phase: _selection(selection, f"{path}.phases.{phase}", registry)
        for phase, selection in raw_phases.items()
    })
    shape = match.get("spatial_shape")
    spatial_shape = None
    if shape is not None:
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            raise ValueError(f"{path}.match.spatial_shape must contain three integers")
        spatial_shape = tuple(int(item) for item in shape)
    cls = ProfileOverride if override else ProfileRule
    return cls(
        identifier=identifier,
        family=family,
        direction=data.get("direction"),
        phases=phases,
        batch=_range(match.get("batch"), f"{path}.match.batch"),
        spatial_volume=_range(match.get("spatial_volume"), f"{path}.match.spatial_volume"),
        in_channels=match.get("in_channels"),
        out_channels=match.get("out_channels"),
        dtype=match.get("dtype"),
        total_vram_gib=_range(match.get("total_vram_gib"), f"{path}.match.total_vram_gib"),
        confidence=str(data.get("confidence", "measured" if override else "interpolated")),
        spatial_shape=spatial_shape,
        model_family=match.get("model_family"),
        variant=match.get("variant"),
        checkpointing=match.get("checkpointing"),
    )


def parse_profile(
    value: Mapping[str, object],
    *,
    registry: ImplementationRegistry | None = None,
) -> OptimizationProfile:
    """Parse a JSON-compatible mapping into an immutable profile."""

    registry = registry or default_implementation_registry()
    if value.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    header = _mapping(value.get("profile"), "profile")
    name = header.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("profile.name must be a non-empty string")
    target = _mapping(header.get("target"), "profile.target")
    vendor = target.get("vendor")
    if not isinstance(vendor, str):
        raise ValueError("profile.target.vendor must be a string")
    raw_sm = target.get("sm")
    target_sm = None
    if raw_sm is not None:
        if not isinstance(raw_sm, (list, tuple)) or len(raw_sm) != 2:
            raise ValueError("profile.target.sm must contain two integers")
        target_sm = (int(raw_sm[0]), int(raw_sm[1]))

    raw_defaults = _mapping(value.get("defaults", {}), "defaults")
    defaults: dict[str, Mapping[str, ProfileSelection]] = {}
    for family, raw_phases in raw_defaults.items():
        phase_mapping = _mapping(raw_phases, f"defaults.{family}")
        defaults[family] = MappingProxyType({
            phase: _selection(selection, f"defaults.{family}.{phase}", registry)
            for phase, selection in phase_mapping.items()
        })

    rules = tuple(
        _rule(item, f"rules[{index}]", registry, override=False)
        for index, item in enumerate(value.get("rules", ()))
    )
    overrides = tuple(
        _rule(item, f"overrides[{index}]", registry, override=True)
        for index, item in enumerate(value.get("overrides", ()))
    )
    identifiers = [rule.identifier for rule in (*rules, *overrides)]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate rule or override id")
    provenance = _mapping(header.get("provenance", {}), "profile.provenance")
    return OptimizationProfile(
        schema_version=1,
        name=name,
        vendor=vendor,
        target_sm=target_sm,
        provenance=_freeze(provenance),  # type: ignore[arg-type]
        defaults=MappingProxyType(defaults),
        rules=rules,
        overrides=overrides,
        measurements=_freeze(value.get("measurements")),
    )
