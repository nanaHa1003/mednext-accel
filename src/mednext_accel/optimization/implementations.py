"""Metadata registry for selectable operator implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SemanticEquivalence = Literal["exact", "numerical", "approximate"]


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

    def register(self, spec: ImplementationSpec) -> None:
        if spec.identifier in self._specs:
            raise ValueError(f"implementation {spec.identifier!r} is already registered")
        self._specs[spec.identifier] = spec

    def resolve_metadata(self, identifier: str) -> ImplementationSpec:
        try:
            return self._specs[identifier]
        except KeyError as error:
            raise ValueError(f"unknown implementation {identifier!r}") from error

    def __contains__(self, identifier: object) -> bool:
        return identifier in self._specs

    def __iter__(self):
        return iter(self._specs.values())


def default_implementation_registry() -> ImplementationRegistry:
    registry = ImplementationRegistry()
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
        ),
        ("pointwise_gemm_per_sample", ("pointwise_conv3d",), ("training",), "numerical", False),
        ("triton_depthwise_dx", ("depthwise_conv3d",), ("backward_input",), "numerical", False),
        ("triton_split_dw", ("depthwise_conv3d",), ("backward_weight",), "numerical", False),
        (
            "triton_transpose_split_dw",
            ("depthwise_conv_transpose3d",),
            ("backward_weight",),
            "numerical",
            False,
        ),
        ("triton_downsample_dx", ("depthwise_conv3d",), ("backward_input",), "numerical", False),
        ("gelu_tanh", ("gelu",), ("inference",), "approximate", True),
    )
    for identifier, families, phases, equivalence, export_safe in definitions:
        registry.register(ImplementationSpec(
            identifier=identifier, version=1, families=families, phases=phases,
            equivalence=equivalence, export_safe=export_safe,
        ))
    return registry
