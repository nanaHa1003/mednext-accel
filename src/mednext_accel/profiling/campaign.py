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
_FAMILY_VARIANTS = {"mednext_v1": _VARIANTS, "mednext_v2": ("base", "wide")}
_CHECKPOINT_CONTEXTS = ("none", "all-expansion", "whole-block")


@dataclass(frozen=True, slots=True)
class Workload:
    model_family: Literal["mednext_v1", "mednext_v2"] = "mednext_v1"
    variant: str = "base"
    spatial: tuple[int, int, int] = (128, 128, 128)
    dtypes: tuple[str, ...] = ("bfloat16",)
    checkpointing: str = "none"
    in_channels: int = 1
    out_channels: int = 3

    @property
    def family(self) -> Literal["mednext_v1", "mednext_v2"]:
        """Campaign spelling; evidence retains the canonical model_family key."""
        return self.model_family

    def __post_init__(self) -> None:
        if self.model_family not in _FAMILY_VARIANTS:
            raise ValueError(f"unknown model family {self.model_family!r}")
        if self.variant not in _FAMILY_VARIANTS[self.model_family]:
            raise ValueError(f"unknown {self.model_family} variant {self.variant!r}")


@dataclass(frozen=True, slots=True)
class Campaign:
    preset: str
    workloads: tuple[Workload, ...]
    batch_search: BatchSearch
    objective: Literal["balanced", "throughput", "memory"] = "balanced"
    compile_mode: str = "max-autotune-no-cudagraphs"
    seed: int = 0

    def __post_init__(self):
        if self.objective not in ("balanced", "throughput", "memory"):
            raise ValueError(f"unknown profiling objective {self.objective!r}")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")


CampaignSource = str | PathLike[str] | Mapping[str, object] | Campaign


def campaign_to_primitive(campaign: Campaign) -> dict[str, object]:
    """Return the complete, JSON-compatible identity of a profiling campaign."""

    return {
        "phase": "training",
        "seed": campaign.seed,
        "preset": campaign.preset,
        "objective": campaign.objective,
        "compile_mode": campaign.compile_mode,
        "batch_search": {
            "memory_fraction": campaign.batch_search.memory_fraction,
            "maximum": campaign.batch_search.maximum,
        },
        "workloads": [
            {
                "model_family": workload.model_family,
                "variant": workload.variant,
                "spatial": list(workload.spatial),
                "dtypes": list(workload.dtypes),
                "checkpointing": workload.checkpointing,
                "in_channels": workload.in_channels,
                "out_channels": workload.out_channels,
            }
            for workload in campaign.workloads
        ],
    }


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
    if "phases" in data:
        raise ValueError("campaign phases was removed; profiling measures training only")
    preset = str(data.get("preset", "mednext-v1"))
    if preset not in ("mednext-v1", "all"):
        raise ValueError(f"preset {preset!r} is unavailable; available presets: mednext-v1, all")
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
            if "phases" in item:
                raise ValueError("workload phases was removed; profiling measures training only")
            unknown_fields = sorted(
                set(item)
                - {
                    "family",
                    "model_family",
                    "variant",
                    "spatial",
                    "dtypes",
                    "checkpointing",
                    "in_channels",
                    "out_channels",
                }
            )
            if unknown_fields:
                raise ValueError(
                    f"unknown workloads[{index}] field(s): {', '.join(unknown_fields)}"
                )
            if (
                "family" in item
                and "model_family" in item
                and item["family"] != item["model_family"]
            ):
                raise ValueError(f"workloads[{index}].family and model_family disagree")
            family = str(item.get("family", item.get("model_family", "mednext_v1")))
            variant = str(item.get("variant", "base"))
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
                        model_family=family,
                        variant=variant,
                        spatial=spatial,  # type: ignore[arg-type]
                        dtypes=tuple(item.get("dtypes", ("bfloat16",))),
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
        seed=data.get("seed", 0),
    )
