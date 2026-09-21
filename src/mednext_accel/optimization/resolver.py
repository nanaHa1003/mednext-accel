"""Pure deterministic selection of registered operator implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from .descriptors import ExecutionContext, OperatorDescriptor
from .implementations import ImplementationRegistry, default_implementation_registry
from .report import Decision
from .schema import OptimizationProfile, ProfileRule, ProfileSelection


def scale_work(
    anchor_value: int,
    anchor_work: int,
    context_work: int,
    *,
    minimum: int,
    maximum: int,
    multiple: int,
) -> int:
    if anchor_work <= 0 or multiple <= 0:
        raise ValueError("anchor_work and multiple must be positive")
    raw = round(anchor_value * context_work / anchor_work)
    clamped = min(max(raw, minimum), maximum)
    return max(multiple, round(clamped / multiple) * multiple)


def _matches(rule: ProfileRule, descriptor: OperatorDescriptor, context: ExecutionContext) -> bool:
    return (
        rule.family == descriptor.family
        and (rule.direction is None or rule.direction == descriptor.direction)
        and rule.batch.contains(context.batch_size)
        and rule.spatial_volume.contains(context.spatial_volume)
        and rule.total_vram_gib.contains(context.total_vram_gib)
        and (rule.in_channels is None or rule.in_channels == descriptor.in_channels)
        and (rule.out_channels is None or rule.out_channels == descriptor.out_channels)
        and (rule.dtype is None or rule.dtype == context.dtype)
        and (rule.spatial_shape is None or rule.spatial_shape == context.spatial_shape)
        and (rule.model_family is None or rule.model_family == context.model_family)
        and (rule.variant is None or rule.variant == context.variant)
        and (rule.checkpointing is None or rule.checkpointing == context.checkpointing)
    )


def _evaluate(value: Any, context: ExecutionContext) -> Any:
    if isinstance(value, Mapping):
        if value.get("formula") == "scale_work":
            context_work = context.batch_size * context.spatial_volume
            return scale_work(
                int(value["anchor_value"]),
                int(value["anchor_work"]),
                context_work,
                minimum=int(value["minimum"]),
                maximum=int(value["maximum"]),
                multiple=int(value["multiple"]),
            )
        return MappingProxyType({key: _evaluate(item, context) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_evaluate(item, context) for item in value)
    return value


class OptimizationResolver:
    """Resolve ordered external, exact-SM, and generic profile layers."""

    def __init__(
        self,
        profiles: Sequence[OptimizationProfile],
        *,
        registry: ImplementationRegistry | None = None,
        warning: str | None = None,
        target_agnostic_profiles: int = 0,
    ) -> None:
        if not profiles:
            raise ValueError("at least one optimization profile is required")
        self.profiles = tuple(profiles)
        self.registry = registry or default_implementation_registry()
        self.warning = warning
        self.target_agnostic_profiles = target_agnostic_profiles

    def resolve(
        self,
        descriptor: OperatorDescriptor,
        context: ExecutionContext,
        phase: str,
    ) -> Decision:
        for profile_index, profile in enumerate(self.profiles):
            if (
                profile_index >= self.target_agnostic_profiles
                and profile.target_sm is not None
                and profile.target_sm != context.sm
            ):
                continue
            for rules in (profile.overrides, profile.rules):
                for rule in rules:
                    selection = rule.phases.get(phase)
                    if selection is not None and _matches(rule, descriptor, context):
                        return self._decision(
                            profile,
                            rule.identifier,
                            rule.confidence,
                            descriptor,
                            context,
                            phase,
                            selection,
                        )
            phases = profile.defaults.get(descriptor.family)
            if phases is not None and phase in phases:
                return self._decision(
                    profile,
                    f"default:{descriptor.family}",
                    "default",
                    descriptor,
                    context,
                    phase,
                    phases[phase],
                )
        return Decision(
            descriptor=descriptor,
            phase=phase,
            implementation="reference",
            parameters={},
            profile=self.profiles[-1].name,
            rule="correctness-guard",
            confidence="guard",
            warning=self.warning,
        )

    def _decision(
        self,
        profile: OptimizationProfile,
        rule: str,
        confidence: str,
        descriptor: OperatorDescriptor,
        context: ExecutionContext,
        phase: str,
        selection: ProfileSelection,
    ) -> Decision:
        spec = self.registry.resolve_metadata(selection.implementation)
        family_allowed = "*" in spec.families or descriptor.family in spec.families
        phase_allowed = phase in spec.phases
        allowed = (
            family_allowed
            and phase_allowed
            and spec.is_allowed(
                allow_approximate=context.allow_approximate,
                export=context.export or phase == "export",
            )
        )
        if not allowed:
            return Decision(
                descriptor=descriptor,
                phase=phase,
                implementation="reference",
                parameters={},
                profile=profile.name,
                rule=f"guard:{rule}",
                confidence="guard",
                warning=self.warning,
            )
        parameters = {key: _evaluate(value, context) for key, value in selection.parameters.items()}
        return Decision(
            descriptor=descriptor,
            phase=phase,
            implementation=selection.implementation,
            parameters=parameters,
            profile=profile.name,
            rule=rule,
            confidence=confidence,  # type: ignore[arg-type]
            warning=self.warning,
        )

    def resolve_all(
        self,
        descriptors: Sequence[OperatorDescriptor],
        context: ExecutionContext,
        phase: str,
    ) -> tuple[Decision, ...]:
        return tuple(self.resolve(descriptor, context, phase) for descriptor in descriptors)
