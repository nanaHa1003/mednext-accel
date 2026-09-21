"""Subprocess isolation and common benchmark result records."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class SubprocessResult:
    status: str
    payload: dict[str, Any]
    message: str = ""


def run_json_subprocess(
    command: Sequence[str], *, timeout: float, input: str | None = None
) -> SubprocessResult:
    """Run one isolated probe and decode its final JSON output line."""

    try:
        completed = subprocess.run(
            tuple(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            input=input,
        )
    except subprocess.TimeoutExpired:
        return SubprocessResult("timeout", {}, f"probe exceeded {timeout:g} seconds")
    if completed.returncode != 0:
        return SubprocessResult(
            "infrastructure_error", {}, completed.stderr.strip() or completed.stdout.strip()
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return SubprocessResult("infrastructure_error", {}, "probe emitted no JSON")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        return SubprocessResult("infrastructure_error", {}, f"invalid probe JSON: {error}")
    if not isinstance(payload, dict):
        return SubprocessResult("infrastructure_error", {}, "probe JSON must be an object")
    status = str(payload.get("status", "ok"))
    return SubprocessResult(
        status,
        dict(MappingProxyType(payload)),
        str(payload.get("message", "")),
    )
