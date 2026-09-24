"""Metadata registry for selectable operator implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .descriptors import ExecutionContext, OperatorDescriptor

SemanticEquivalence = Literal["exact", "numerical", "approximate"]
EligibilityGuard = Callable[["OperatorDescriptor", "ExecutionContext", str], str | None]


def _grn_eligibility(
    descriptor: OperatorDescriptor,
    context: ExecutionContext,
    phase: str,
    *,
    triton_available: bool,
) -> str | None:
    if context.phase != "training" or phase != "training":
        return "implementation requires training"
    if context.device_type != "cuda":
        return "implementation requires CUDA"
    if context.dtype not in ("float16", "bfloat16"):
        return "implementation does not support this dtype"
    if not triton_available:
        return "Triton backend is unavailable"
    return None


def _pointwise_eligibility(
    descriptor: OperatorDescriptor, context: ExecutionContext, phase: str
) -> str | None:
    if context.phase != "training":
        return "implementation requires training"
    if context.dtype not in ("bfloat16", "float16", "float32"):
        return "implementation does not support this dtype"
    if (
        descriptor.direction != "regular"
        or descriptor.kernel_size != (1, 1, 1)
        or descriptor.stride != (1, 1, 1)
        or descriptor.padding != (0, 0, 0)
        or descriptor.dilation != (1, 1, 1)
        or descriptor.groups != 1
    ):
        return "implementation does not support this geometry"
    return None


def _depthwise_eligibility(
    descriptor: OperatorDescriptor,
    context: ExecutionContext,
    phase: str,
    *,
    direction: str,
    triton_available: bool,
) -> str | None:
    if context.phase != "training":
        return "implementation requires training"
    if context.dtype not in ("bfloat16", "float16", "float32"):
        return "implementation does not support this dtype"
    kernel_size = descriptor.kernel_size
    stride = 1 if direction == "regular" else 2
    if (
        kernel_size is None
        or descriptor.direction != direction
        or kernel_size != (kernel_size[0],) * 3
        or kernel_size[0] % 2 != 1
        or descriptor.stride != (stride,) * 3
        or descriptor.padding != (kernel_size[0] // 2,) * 3
        or descriptor.dilation != (1, 1, 1)
        or descriptor.groups != descriptor.in_channels == descriptor.out_channels
    ):
        return "implementation does not support this geometry"
    if not triton_available:
        return "Triton backend is unavailable"
    if len(set(context.spatial_shape)) != 1:
        return "implementation requires a cubic spatial shape"
    return None


@dataclass(frozen=True, slots=True)
class ImplementationSpec:
    identifier: str
    version: int
    families: tuple[str, ...]
    phases: tuple[str, ...]
    equivalence: SemanticEquivalence
    export_safe: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "families", tuple(self.families))
        object.__setattr__(self, "phases", tuple(self.phases))
        if not self.identifier or self.version <= 0:
            raise ValueError("implementation identifier and version must be valid")
        if not self.families or not self.phases:
            raise ValueError("implementation families and phases must not be empty")
        if self.equivalence not in ("exact", "numerical", "approximate"):
            raise ValueError(f"unknown equivalence {self.equivalence!r}")

    def is_allowed(self, *, allow_approximate: bool, export: bool = False) -> bool:
        return (self.equivalence != "approximate" or allow_approximate) and (
            not export or self.export_safe
        )


class ImplementationRegistry:
    """A backend-independent registry keyed by stable public identifiers."""

    def __init__(self) -> None:
        self._specs: dict[str, ImplementationSpec] = {}
        self._eligibility: dict[str, EligibilityGuard | None] = {}
        self._execution_guards: dict[str, tuple[str, ...]] = {}

    def register(
        self,
        spec: ImplementationSpec,
        *,
        eligibility: EligibilityGuard | None = None,
        execution_guards: tuple[str, ...] = (),
    ) -> None:
        if spec.identifier in self._specs:
            raise ValueError(f"implementation {spec.identifier!r} is already registered")
        self._specs[spec.identifier] = spec
        self._eligibility[spec.identifier] = eligibility
        self._execution_guards[spec.identifier] = tuple(execution_guards)

    def resolve_metadata(self, identifier: str) -> ImplementationSpec:
        try:
            return self._specs[identifier]
        except KeyError as error:
            raise ValueError(f"unknown implementation {identifier!r}") from error

    def guard(
        self,
        identifier: str,
        descriptor: OperatorDescriptor,
        context: ExecutionContext,
        phase: str,
    ) -> str | None:
        self.resolve_metadata(identifier)
        guard = self._eligibility[identifier]
        return None if guard is None else guard(descriptor, context, phase)

    def execution_guards(self, identifier: str) -> tuple[str, ...]:
        self.resolve_metadata(identifier)
        return self._execution_guards[identifier]

    def __contains__(self, identifier: object) -> bool:
        return identifier in self._specs

    def __iter__(self):
        return iter(self._specs.values())


def default_implementation_registry(
    *, triton_available: bool | None = None
) -> ImplementationRegistry:
    registry = ImplementationRegistry()
    available = find_spec("triton") is not None if triton_available is None else triton_available
    definitions = (
        (
            "reference",
            ("*",),
            (
                "training",
                "inference",
                "export",
                "forward",
                "backward_input",
                "backward_weight",
                "backward_bias",
            ),
            "exact",
            True,
            None,
            (),
        ),
        (
            "pointwise_gemm_per_sample",
            ("pointwise_conv3d",),
            ("training",),
            "numerical",
            False,
            _pointwise_eligibility,
            ("grad_enabled", "rank_5", "contiguous"),
        ),
        (
            "triton_depthwise_dx",
            ("depthwise_conv3d",),
            ("backward_input",),
            "numerical",
            False,
            partial(_depthwise_eligibility, direction="regular", triton_available=available),
            ("grad_enabled", "rank_5", "contiguous", "zero_output_padding"),
        ),
        (
            "triton_split_dw",
            ("depthwise_conv3d",),
            ("backward_weight",),
            "numerical",
            False,
            partial(_depthwise_eligibility, direction="regular", triton_available=available),
            ("grad_enabled", "rank_5", "contiguous", "zero_output_padding"),
        ),
        (
            "triton_transpose_split_dw",
            ("depthwise_conv_transpose3d",),
            ("backward_weight",),
            "numerical",
            False,
            partial(_depthwise_eligibility, direction="transpose", triton_available=available),
            ("grad_enabled", "rank_5", "contiguous", "zero_output_padding"),
        ),
        (
            "triton_downsample_dx",
            ("depthwise_conv3d",),
            ("backward_input",),
            "numerical",
            False,
            partial(_depthwise_eligibility, direction="downsample", triton_available=available),
            ("grad_enabled", "rank_5", "contiguous", "zero_output_padding"),
        ),
        (
            "triton_fused_grn",
            ("global_response_norm3d",),
            ("training",),
            "numerical",
            False,
            partial(_grn_eligibility, triton_available=available),
            ("grad_enabled", "rank_5", "contiguous"),
        ),
        ("gelu_tanh", ("gelu",), ("inference",), "approximate", True, None, ()),
    )
    for (
        identifier,
        families,
        phases,
        equivalence,
        export_safe,
        eligibility,
        execution_guards,
    ) in definitions:
        registry.register(
            ImplementationSpec(
                identifier=identifier,
                version=1,
                families=families,
                phases=phases,
                equivalence=equivalence,
                export_safe=export_safe,
            ),
            eligibility=eligibility,
            execution_guards=execution_guards,
        )
    return registry
