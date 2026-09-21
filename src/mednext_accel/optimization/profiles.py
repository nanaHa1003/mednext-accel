"""Load optimization profiles without importing execution backends."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from importlib.resources import files
from os import PathLike
from pathlib import Path

import yaml

from .implementations import ImplementationRegistry
from .schema import OptimizationProfile, parse_profile

ProfileSource = str | PathLike[str] | Mapping[str, object]


def load_profile(
    source: ProfileSource, *, registry: ImplementationRegistry | None = None
) -> OptimizationProfile:
    if isinstance(source, Mapping):
        return parse_profile(source, registry=registry)
    if isinstance(source, str) and source in ("auto", "reference"):
        raise ValueError(f"{source!r} is a factory mode, not a profile document")
    path = Path(source)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
    elif path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text())
    else:
        raise ValueError("profile path suffix must be .json, .yaml, or .yml")
    return parse_profile(_mapping_document(data), registry=registry)


def _mapping_document(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("profile document must contain a mapping")
    return value


def load_bundled_profile(name: str) -> OptimizationProfile:
    resource = files("mednext_accel.profiles").joinpath(f"{name}.json")
    return parse_profile(_mapping_document(json.loads(resource.read_text())))


class ProfileRegistry:
    """Select bundled profiles and optional user layers for an execution target."""

    def __init__(self, external: ProfileSource | None = None) -> None:
        self.external = None if external is None else load_profile(external)
        self._warned: set[str] = set()

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            warnings.warn(message, UserWarning, stacklevel=3)
            self._warned.add(key)

    def resolver_for(self, *, sm: tuple[int, int] | None):
        from .resolver import OptimizationResolver

        layers: list[OptimizationProfile] = []
        warning = None
        if self.external is not None:
            layers.append(self.external)
            if self.external.target_sm is not None and self.external.target_sm != sm:
                warning = (
                    f"profile {self.external.name!r} targets SM {self.external.target_sm}, "
                    f"but the current device is SM {sm}; applying it as requested"
                )
                self._warn_once(f"external:{self.external.name}:{sm}", warning)
        bundled_name = "sm120" if sm == (12, 0) else "generic-nvidia"
        if bundled_name == "generic-nvidia" and sm is not None:
            warning = warning or (
                f"no bundled profile for SM {sm}; using generic-nvidia defaults "
                "derived from RTX 5090 measurements"
            )
            self._warn_once(f"generic:{sm}", warning)
        layers.append(load_bundled_profile(bundled_name))
        if bundled_name != "generic-nvidia":
            layers.append(load_bundled_profile("generic-nvidia"))
        return OptimizationResolver(layers, warning=warning)
