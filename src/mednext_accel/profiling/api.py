"""Public API for generating one reusable local optimization profile."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import torch

from ..optimization.schema import OptimizationProfile, profile_to_primitive
from .campaign import Campaign, CampaignSource, load_campaign
from .environment import collect_environment
from .progress import ProgressEvent, ProgressReporter
from .synthesize import Measurement, synthesize_profile


def run_campaign(
    campaign: Campaign, progress: ProgressReporter | None = None
) -> tuple[Measurement, ...]:
    """Execute a campaign; imported lazily to keep CLI discovery GPU-free."""

    from .runner import run_campaign as execute

    return execute(campaign, progress=progress)


def synthesize_campaign(
    campaign: Campaign,
    measurements: tuple[Measurement, ...],
    environment: dict[str, object] | None = None,
) -> OptimizationProfile:
    if not torch.cuda.is_available():
        raise RuntimeError("profiling requires an NVIDIA CUDA device")
    sm = torch.cuda.get_device_capability()
    return synthesize_profile(
        measurements,
        name=f"sm{sm[0]}{sm[1]}-local",
        sm=sm,
        objective=campaign.objective,
        environment=environment,
        compile_mode=campaign.compile_mode,
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
    environment: dict[str, object] = field(default_factory=dict)

    def save(self, path: str | PathLike[str] | None = None) -> Path:
        return write_profile_atomic(self.output_path if path is None else path, self.profile)


def profile(
    source: CampaignSource | None = None,
    *,
    progress: ProgressReporter | None = None,
) -> ProfilingResult:
    campaign = load_campaign(source)
    environment = collect_environment()
    gpu = environment["gpu"]
    software = environment["software"]
    assert isinstance(gpu, dict) and isinstance(software, dict)
    sm = tuple(int(item) for item in gpu["sm"])
    planned_profile = synthesize_profile(
        (), name=f"sm{sm[0]}{sm[1]}-local", sm=sm, objective=campaign.objective
    )
    planned_output = default_profile_path(planned_profile)
    if progress is not None:
        progress.emit(
            ProgressEvent(
                "environment",
                "environment",
                message=(
                    f"{gpu['name']} · SM{gpu['sm'][0]}{gpu['sm'][1]} · "
                    f"{int(gpu['total_memory_bytes']) / 1024**3:.1f} GiB · "
                    f"driver {environment['driver']} · Python {environment['python']} · "
                    f"mednext-accel {software['mednext_accel']} · "
                    f"PyTorch {software['torch']} · CUDA {software['cuda']} · "
                    f"cuDNN {software['cudnn']} · Triton {software['triton']} · "
                    f"objective {campaign.objective} · compile {campaign.compile_mode}"
                ),
            )
        )
        progress.emit(
            ProgressEvent(
                "status", "output", message=f"profile will be written to {planned_output}"
            )
        )
    measurements = run_campaign(campaign, progress=progress)
    generated = synthesize_campaign(campaign, measurements, environment=environment)
    result = ProfilingResult(generated, measurements, planned_output, environment)
    result.save()
    if progress is not None:
        progress.emit(
            ProgressEvent(
                "complete",
                "campaign",
                message=f"saved {len(measurements)} measurements to {result.output_path}",
            )
        )
    return result
