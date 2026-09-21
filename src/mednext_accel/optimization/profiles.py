"""Load optimization profiles without importing execution backends."""

from __future__ import annotations

import json
from collections.abc import Mapping
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
