"""YAML-compatible profiling campaign configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

import yaml

from .batch_search import BatchSearch

_VARIANTS = ("small", "base", "medium", "large")
_CHECKPOINT_CONTEXTS = ("none", "all-expansion", "whole-block")


@dataclass(frozen=True, slots=True)
class Workload:
    model_family: str
    variant: str
    spatial: tuple[int, int, int]
    dtypes: tuple[str, ...] = ("bfloat16",)
    phases: tuple[str, ...] = ("training", "inference")
    checkpointing: str = "none"
    in_channels: int = 1
    out_channels: int = 3


@dataclass(frozen=True, slots=True)
class Campaign:
    preset: str
    workloads: tuple[Workload, ...]
    batch_search: BatchSearch
    objective: Literal["balanced", "throughput", "memory"] = "balanced"
    compile_mode: str = "max-autotune-no-cudagraphs"


CampaignSource = str | PathLike[str] | Mapping[str, object] | Campaign


def _default_workloads() -> tuple[Workload, ...]:
    return tuple(Workload("mednext_v1", variant, (128, 128, 128)) for variant in _VARIANTS)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _load_mapping(source: str | PathLike[str] | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(source, Mapping):
        return source
    value = yaml.safe_load(Path(source).read_text())
    return _mapping(value, "campaign")


def load_campaign(source: CampaignSource | None) -> Campaign:
    if isinstance(source, Campaign):
        return source
    if source is None:
        return Campaign("all", _default_workloads(), BatchSearch())
    data = _load_mapping(source)
    preset = str(data.get("preset", "mednext-v1"))
    if preset not in ("mednext-v1", "all"):
        raise ValueError(f"preset {preset!r} is unavailable; installed model families: mednext-v1")
    raw_search = _mapping(data.get("batch_search", {}), "batch_search")
    unknown_search_fields = sorted(set(raw_search) - {"memory_fraction", "maximum"})
    if unknown_search_fields:
        raise ValueError(f"unknown batch_search field(s): {', '.join(unknown_search_fields)}")
    batch_search = BatchSearch(
        memory_fraction=float(raw_search.get("memory_fraction", 0.90)),
        maximum=(None if raw_search.get("maximum") is None else int(raw_search["maximum"])),
    )
    raw_workloads = data.get("workloads")
    if raw_workloads is None:
        workloads = _default_workloads()
    elif not isinstance(raw_workloads, list):
        raise ValueError("workloads must be a list")
    else:
        expanded: list[Workload] = []
        for index, raw in enumerate(raw_workloads):
            item = _mapping(raw, f"workloads[{index}]")
            variant = str(item.get("variant", "base"))
            if variant not in _VARIANTS:
                raise ValueError(f"unknown MedNeXt v1 variant {variant!r}")
            spatial_value = item.get("spatial", (128, 128, 128))
            if not isinstance(spatial_value, (list, tuple)) or len(spatial_value) != 3:
                raise ValueError(f"workloads[{index}].spatial must contain three integers")
            spatial = tuple(int(size) for size in spatial_value)
            checkpoint = str(item.get("checkpointing", "none"))
            contexts = _CHECKPOINT_CONTEXTS if checkpoint == "auto" else (checkpoint,)
            for context in contexts:
                if context not in _CHECKPOINT_CONTEXTS:
                    raise ValueError(f"unknown checkpoint context {context!r}")
                expanded.append(
                    Workload(
                        model_family="mednext_v1",
                        variant=variant,
                        spatial=spatial,  # type: ignore[arg-type]
                        dtypes=tuple(item.get("dtypes", ("bfloat16",))),
                        phases=tuple(item.get("phases", ("training", "inference"))),
                        checkpointing=context,
                        in_channels=int(item.get("in_channels", 1)),
                        out_channels=int(item.get("out_channels", 3)),
                    )
                )
        workloads = tuple(expanded)
    return Campaign(
        preset=preset,
        workloads=workloads,
        batch_search=batch_search,
        objective=str(data.get("objective", "balanced")),  # type: ignore[arg-type]
        compile_mode=str(data.get("compile_mode", "max-autotune-no-cudagraphs")),
    )
