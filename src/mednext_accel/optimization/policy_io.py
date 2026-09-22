"""Load runtime policy v2 without reading provenance artifacts or CUDA state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import yaml

from .implementations import ImplementationRegistry
from .policy import OptimizationPolicy, parse_policy

PolicySource = str | PathLike[str] | Mapping[str, object]


def load_policy(
    source: PolicySource, *, registry: ImplementationRegistry | None = None
) -> OptimizationPolicy:
    if isinstance(source, Mapping):
        return parse_policy(source, registry=registry)
    if isinstance(source, str) and source in ("auto", "reference"):
        raise ValueError(f"{source!r} is a factory mode, not a policy document")
    path = Path(source).expanduser()
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("policy path suffix must be .json, .yaml, or .yml")
    return parse_policy(data, registry=registry)
