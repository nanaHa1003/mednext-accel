"""Public API for generating one reusable local optimization profile."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import torch

from ..optimization.schema import OptimizationProfile, profile_to_primitive
from .campaign import Campaign, CampaignSource, load_campaign
from .synthesize import Measurement, synthesize_profile


def run_campaign(campaign: Campaign) -> tuple[Measurement, ...]:
    """Execute a campaign; imported lazily to keep CLI discovery GPU-free."""

    from .runner import run_campaign as execute

    return execute(campaign)


def synthesize_campaign(
    campaign: Campaign, measurements: tuple[Measurement, ...]
) -> OptimizationProfile:
    if not torch.cuda.is_available():
        raise RuntimeError("profiling requires an NVIDIA CUDA device")
    sm = torch.cuda.get_device_capability()
    return synthesize_profile(
        measurements,
        name=f"sm{sm[0]}{sm[1]}-local",
        sm=sm,
        objective=campaign.objective,
    )


def default_profile_path(profile: OptimizationProfile) -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    cache = Path(root) if root else Path.home() / ".cache"
    return cache / "mednext_accel" / "profiles" / f"{profile.name}.json"


def write_profile_atomic(path: str | PathLike[str], profile: OptimizationProfile) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(profile_to_primitive(profile), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w") as temporary:
            temporary.write(document)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


@dataclass(frozen=True, slots=True)
class ProfilingResult:
    profile: OptimizationProfile
    measurements: tuple[Measurement, ...]
    output_path: Path

    def save(self, path: str | PathLike[str] | None = None) -> Path:
        return write_profile_atomic(self.output_path if path is None else path, self.profile)


def profile(source: CampaignSource | None = None) -> ProfilingResult:
    campaign = load_campaign(source)
    measurements = run_campaign(campaign)
    generated = synthesize_campaign(campaign, measurements)
    result = ProfilingResult(generated, measurements, default_profile_path(generated))
    result.save()
    return result
