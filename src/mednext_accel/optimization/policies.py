"""Load all compact bundled policies independently of the construction device."""

from functools import cache
from importlib.resources import files

import yaml

from .policy import OptimizationPolicy, parse_policy
from .policy_io import PolicySource, load_policy
from .policy_resolver import PolicyResolver

_BUNDLED = ("shared-nvidia", "sm86", "sm89", "sm120")


@cache
def load_bundled_policy(name: str) -> OptimizationPolicy:
    if name not in _BUNDLED:
        raise ValueError(f"unknown bundled policy {name!r}")
    resource = files("mednext_accel.policies").joinpath(f"{name}.yaml")
    return parse_policy(yaml.safe_load(resource.read_text(encoding="utf-8")))


class PolicyRegistry:
    def __init__(self, external: PolicySource | None = None) -> None:
        self.external = None if external is None else load_policy(external)
        self.bundled = tuple(load_bundled_policy(name) for name in _BUNDLED)

    def resolver(self, *, triton_available: bool | None = None) -> PolicyResolver:
        return PolicyResolver(
            self.bundled, external=self.external, triton_available=triton_available
        )
