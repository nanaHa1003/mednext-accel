"""Profile the local GPU and publish separate evidence and runtime policy artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ..optimization.policy import EvidenceReference, OptimizationPolicy, policy_to_primitive
from .campaign import Campaign, CampaignSource, campaign_to_primitive, load_campaign
from .environment import collect_environment
from .evidence import (
    CampaignEvidence,
    EnvironmentEvidence,
    PolicyPublicationEvidence,
    ProfilingEvidence,
    freeze,
)
from .execution import validate_campaign_dtypes
from .progress import ProgressEvent, ProgressReporter
from .synthesize import Measurement, synthesize_profile

if TYPE_CHECKING:
    from .runner import CampaignRun


def execute_campaign(campaign: Campaign, progress: ProgressReporter | None = None) -> CampaignRun:
    """Execute a campaign; imported lazily to keep CLI discovery GPU-free."""
    from .runner import execute_campaign as execute

    return execute(campaign, progress=progress)


def run_campaign(
    campaign: Campaign, progress: ProgressReporter | None = None
) -> tuple[Measurement, ...]:
    """Return all measured observations without publishing files."""
    return execute_campaign(campaign, progress=progress).measurements


def default_policy_path(policy: OptimizationPolicy) -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    cache = Path(root) if root else Path.home() / ".cache"
    return cache / "mednext_accel" / "profiles" / f"{policy.name}.policy.yaml"


@dataclass(frozen=True, slots=True)
class ProfilingArtifacts:
    policy_path: Path
    evidence_path: Path


def _write_temporary(directory: Path, prefix: str, data: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, dir=directory)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_profiling_artifacts(
    policy: OptimizationPolicy,
    evidence: ProfilingEvidence,
    *,
    policy_path: Path,
    evidence_directory: Path | None = None,
) -> tuple[ProfilingArtifacts, OptimizationPolicy]:
    """Publish immutable evidence first, then atomically replace its policy.

    A failed second write leaves the previous policy intact and may leave an
    unreferenced evidence file. No published evidence file is ever overwritten.
    """
    destination = Path(policy_path).expanduser()
    if destination.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("runtime policy output must use a .yaml or .yml path")
    directory = (
        destination.parent if evidence_directory is None else Path(evidence_directory).expanduser()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            evidence.to_primitive(), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    stem = destination.name.removesuffix(destination.suffix).removesuffix(".policy")
    temporary = _write_temporary(directory, f".{stem}.evidence.", raw)
    try:
        while True:
            evidence_path = directory / f"{stem}.{time.time_ns()}-{digest[:12]}.evidence.json"
            try:
                # An atomic no-clobber publication of already complete bytes.
                os.link(temporary, evidence_path)
                break
            except FileExistsError:
                continue
        _sync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    reference = EvidenceReference("profiling-evidence", digest, evidence_path.name)
    referenced = replace(
        policy,
        evidence=tuple(item for item in policy.evidence if item.identifier != reference.identifier)
        + (reference,),
    )
    document = yaml.safe_dump(
        policy_to_primitive(referenced), sort_keys=False, allow_unicode=True
    ).encode("utf-8")
    temporary = _write_temporary(destination.parent, f".{destination.name}.", document)
    try:
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return ProfilingArtifacts(destination, evidence_path), referenced


def write_profiling_artifacts(
    policy: OptimizationPolicy,
    evidence: ProfilingEvidence,
    *,
    policy_path: Path,
    evidence_directory: Path | None = None,
) -> ProfilingArtifacts:
    """Write both artifacts and return their actual paths."""
    artifacts, _ = _publish_profiling_artifacts(
        policy, evidence, policy_path=policy_path, evidence_directory=evidence_directory
    )
    return artifacts


@dataclass(frozen=True, slots=True)
class ProfilingResult:
    policy: OptimizationPolicy
    evidence: ProfilingEvidence
    artifacts: ProfilingArtifacts
    environment: Mapping[str, object]

    def __post_init__(self):
        object.__setattr__(self, "environment", freeze(self.environment))

    def save(
        self,
        *,
        policy_path: Path | None = None,
        evidence_directory: Path | None = None,
    ) -> ProfilingArtifacts:
        return write_profiling_artifacts(
            self.policy,
            self.evidence,
            policy_path=self.artifacts.policy_path if policy_path is None else policy_path,
            evidence_directory=evidence_directory,
        )


def profile(
    source: CampaignSource | None = None,
    *,
    progress: ProgressReporter | None = None,
) -> ProfilingResult:
    campaign = load_campaign(source)
    validate_campaign_dtypes(campaign)
    environment = collect_environment()
    gpu = environment["gpu"]
    software = environment["software"]
    assert isinstance(gpu, dict) and isinstance(software, dict)
    sm = tuple(int(item) for item in gpu["sm"])
    name = f"sm{sm[0]}{sm[1]}-local"
    planned_policy = synthesize_profile((), name=name, sm=sm, objective=campaign.objective)
    planned_output = default_policy_path(planned_policy)
    if progress is not None:
        progress.emit(
            ProgressEvent(
                "environment",
                "environment",
                message=(
                    f"{gpu['name']} · SM{sm[0]}{sm[1]} · "
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
                "status",
                "output",
                message=(
                    f"policy will be written to {planned_output}; "
                    f"immutable evidence JSON will be written alongside it"
                ),
            )
        )
    run = execute_campaign(campaign, progress=progress)
    statistics = run.statistics
    execution = {
        "kernel_case_count": statistics.kernel_case_count,
        "kernel_group_count": statistics.kernel_group_count,
        "whole_model_validation_count": statistics.whole_model_validation_count,
        "deduplicated_reference_count": statistics.requested_case_count
        - statistics.kernel_case_count,
    }
    publication = PolicyPublicationEvidence.from_comparisons(run.model_comparisons)
    generated = synthesize_profile(
        run.measurements,
        name=name,
        sm=sm,
        objective=campaign.objective,
        include_winners=publication.policy_accepted is True,
    )
    evidence = ProfilingEvidence(
        EnvironmentEvidence(environment),
        CampaignEvidence(campaign_to_primitive(campaign)),
        run.batch_searches,
        run.measurements,
        run.model_comparisons,
        execution,
        publication,
    )
    artifacts, referenced = _publish_profiling_artifacts(
        generated, evidence, policy_path=planned_output
    )
    result = ProfilingResult(referenced, evidence, artifacts, environment)
    if progress is not None:
        progress.emit(
            ProgressEvent(
                "complete",
                "campaign",
                message=(
                    f"policy {publication.reason}; "
                    f"saved {len(run.measurements)} kernel observations · "
                    f"policy: {artifacts.policy_path} · evidence: {artifacts.evidence_path}"
                ),
            )
        )
    return result
