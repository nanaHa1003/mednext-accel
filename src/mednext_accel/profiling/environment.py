"""Collect reproducibility metadata without recording user or host identity."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    return line or None


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment(
    *,
    torch_module: Any | None = None,
    package_version: str | None = None,
    triton_version: str | None = None,
    driver_version: str | None = None,
    timestamp: str | None = None,
    python_version: str | None = None,
    platform_name: str | None = None,
) -> dict[str, object]:
    """Return JSON-compatible hardware and software details for a campaign."""

    if torch_module is None:
        import torch

        torch_module = torch
    if not torch_module.cuda.is_available():
        raise RuntimeError("profiling requires an NVIDIA CUDA device")
    device = int(torch_module.cuda.current_device())
    properties = torch_module.cuda.get_device_properties(device)
    capability = torch_module.cuda.get_device_capability(device)
    if triton_version is None:
        triton_version = _installed_version("triton")
    return {
        "collected_at": (
            timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        ),
        "platform": platform_name or platform.platform(),
        "python": python_version or platform.python_version(),
        "driver": driver_version if driver_version is not None else _driver_version(),
        "gpu": {
            "index": device,
            "name": str(properties.name),
            "sm": [int(capability[0]), int(capability[1])],
            "total_memory_bytes": int(properties.total_memory),
        },
        "software": {
            "mednext_accel": package_version or _installed_version("mednext-accel") or "unknown",
            "torch": str(torch_module.__version__),
            "cuda": torch_module.version.cuda,
            "cudnn": torch_module.backends.cudnn.version(),
            "triton": triton_version,
        },
    }
