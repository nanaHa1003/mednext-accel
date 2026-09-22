"""Resolve compact runtime policies against each execution context."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from types import MappingProxyType

from .descriptors import ExecutionContext, OperatorDescriptor
from .implementations import ImplementationRegistry, default_implementation_registry
from .parameters import resolve_parameters
from .policy import Confidence, NumericRange, OptimizationPolicy, PolicyRule, PolicySelection


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    descriptor: OperatorDescriptor
    phase: str
    implementation: str
    parameters: Mapping[str, object]
    policy: str
    rule: str
    confidence: Confidence
    warning: str | None = None
    guard_reason: str | None = None
    execution_guards: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "execution_guards", tuple(self.execution_guards))


def _matches(rule: PolicyRule, descriptor: OperatorDescriptor, context: ExecutionContext) -> bool:
    facts = {
        "family": descriptor.family,
        "direction": descriptor.direction,
        "role": descriptor.role,
        "kernel_size": descriptor.kernel_size,
        "stride": descriptor.stride,
        "channels": (descriptor.in_channels, descriptor.out_channels),
        "batch": context.batch_size,
        "spatial_shape": context.spatial_shape,
        "spatial_volume": context.spatial_volume,
        "work": context.batch_size * context.spatial_volume * descriptor.in_channels,
        "reduction_work": context.batch_size * context.spatial_volume,
        "total_vram_gib": context.total_vram_gib,
        "dtype": context.dtype,
        "model_family": context.model_family,
        "variant": context.variant,
        "checkpointing": context.checkpointing,
    }
    return all(
        expected.contains(facts[name])
        if isinstance(expected, NumericRange)
        else expected == facts[name]
        for name, expected in rule.when.items()
    )


def _geometry_allowed(implementation: str, descriptor: OperatorDescriptor) -> bool:
    if implementation == "pointwise_gemm_per_sample":
        return (
            descriptor.direction == "regular"
            and descriptor.kernel_size == (1, 1, 1)
            and descriptor.stride == (1, 1, 1)
            and descriptor.padding == (0, 0, 0)
            and descriptor.dilation == (1, 1, 1)
            and descriptor.groups == 1
        )
    if not implementation.startswith("triton_"):
        return True
    kernel = descriptor.kernel_size[0]
    direction = (
        "transpose"
        if implementation == "triton_transpose_split_dw"
        else "downsample"
        if implementation == "triton_downsample_dx"
        else "regular"
    )
    stride = 1 if direction == "regular" else 2
    return (
        descriptor.direction == direction
        and descriptor.kernel_size == (kernel,) * 3
        and kernel % 2 == 1
        and descriptor.stride == (stride,) * 3
        and descriptor.padding == (kernel // 2,) * 3
        and descriptor.dilation == (1, 1, 1)
        and descriptor.groups == descriptor.in_channels == descriptor.out_channels
    )


class PolicyResolver:
    """External → exact execution-SM → shared → built-in reference.

    Bundled layers are loaded by the caller once. Their applicable ordering is
    recomputed from each execution context, never frozen to a construction GPU.
    Tensor properties unavailable here remain explicit execution-time checks.
    """

    def __init__(
        self,
        bundled: Sequence[OptimizationPolicy] = (),
        *,
        external: OptimizationPolicy | None = None,
        registry: ImplementationRegistry | None = None,
        triton_available: bool | None = None,
    ) -> None:
        self.bundled = tuple(bundled)
        self.external = external
        self.registry = registry or default_implementation_registry()
        self.triton_available = (
            find_spec("triton") is not None if triton_available is None else triton_available
        )
        self._warned: set[str] = set()

    def context_layers(self, context: ExecutionContext) -> tuple[OptimizationPolicy, ...]:
        if context.device_type != "cuda" or context.sm is None:
            return ()
        applicable = tuple(policy for policy in self.bundled if policy.vendor == "nvidia")
        external = (
            (self.external,)
            if self.external is not None and self.external.vendor == "nvidia"
            else ()
        )
        exact = tuple(policy for policy in applicable if policy.target_sm == context.sm)
        shared = tuple(policy for policy in applicable if policy.target_sm is None)
        return external + exact + shared

    def _warning(self, context: ExecutionContext) -> str | None:
        if (
            self.external is None
            or self.external.vendor != "nvidia"
            or self.external.target_sm is None
            or self.external.target_sm == context.sm
        ):
            return None
        message = (
            f"policy {self.external.name!r} targets SM {self.external.target_sm}, "
            f"but the current device is SM {context.sm}; applying it as requested"
        )
        if message not in self._warned:
            warnings.warn(message, UserWarning, stacklevel=3)
            self._warned.add(message)
        return message

    def resolve(
        self, descriptor: OperatorDescriptor, context: ExecutionContext, phase: str
    ) -> PolicyDecision:
        if context.device_type != "cuda" or context.sm is None:
            return PolicyDecision(
                descriptor,
                phase,
                "reference",
                {},
                "reference",
                "context-guard",
                "guard",
                guard_reason="requires CUDA with a known NVIDIA SM",
            )
        warning = self._warning(context)
        for policy in self.context_layers(context):
            for rule in policy.rules:
                selection = rule.use.get(phase)
                if selection is not None and _matches(rule, descriptor, context):
                    return self._decision(
                        policy, rule, selection, descriptor, context, phase, warning
                    )
        return PolicyDecision(
            descriptor, phase, "reference", {}, "reference", "fallback", "guard", warning
        )

    def _guard_reason(
        self,
        selection: PolicySelection,
        descriptor: OperatorDescriptor,
        context: ExecutionContext,
        phase: str,
    ) -> str | None:
        spec = self.registry.resolve_metadata(selection.implementation)
        if "*" not in spec.families and descriptor.family not in spec.families:
            return "implementation does not support this operator family"
        if phase not in spec.phases:
            return "implementation does not support this phase"
        if (
            context.export or context.phase == "export" or phase == "export"
        ) and not spec.export_safe:
            return "implementation is not export safe"
        if spec.equivalence == "approximate" and not context.allow_approximate:
            return "approximate implementation requires allow_approximate"
        triton = selection.implementation.startswith("triton_")
        pointwise = selection.implementation == "pointwise_gemm_per_sample"
        if triton or pointwise:
            if context.phase != "training":
                return "implementation requires training"
            if context.dtype not in ("bfloat16", "float16", "float32"):
                return "implementation does not support this dtype"
            if not _geometry_allowed(selection.implementation, descriptor):
                return "implementation does not support this geometry"
        if triton:
            if not self.triton_available:
                return "Triton backend is unavailable"
            if len(set(context.spatial_shape)) != 1:
                return "implementation requires a cubic spatial shape"
        return None

    def _decision(
        self,
        policy: OptimizationPolicy,
        rule: PolicyRule,
        selection: PolicySelection,
        descriptor: OperatorDescriptor,
        context: ExecutionContext,
        phase: str,
        warning: str | None,
    ) -> PolicyDecision:
        reason = self._guard_reason(selection, descriptor, context, phase)
        parameters = {}
        if reason is None:
            try:
                parameters = resolve_parameters(selection, descriptor, context)
            except ValueError as error:
                reason = f"invalid launch parameters: {error}"
        if reason is not None:
            return PolicyDecision(
                descriptor,
                phase,
                "reference",
                {},
                policy.name,
                rule.identifier,
                "guard",
                warning,
                guard_reason=reason,
            )
        execution_guards = ()
        if selection.implementation.startswith("triton_"):
            execution_guards = ("grad_enabled", "rank_5", "contiguous", "zero_output_padding")
        elif selection.implementation == "pointwise_gemm_per_sample":
            execution_guards = ("grad_enabled",)
        return PolicyDecision(
            descriptor,
            phase,
            selection.implementation,
            parameters,
            policy.name,
            rule.identifier,
            rule.confidence,
            warning,
            execution_guards=execution_guards,
        )

    def resolve_all(
        self, descriptors: Sequence[OperatorDescriptor], context: ExecutionContext, phase: str
    ) -> tuple[PolicyDecision, ...]:
        return tuple(self.resolve(descriptor, context, phase) for descriptor in descriptors)
