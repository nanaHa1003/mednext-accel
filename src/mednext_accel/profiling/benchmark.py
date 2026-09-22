"""Subprocess isolation and common benchmark result records."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def normalize_probe_message(status: str, message: str | None) -> str | None:
    """Give unsuccessful child results a diagnostic without coercing malformed types."""
    if not isinstance(status, str) or not status:
        raise ValueError("probe status must be a nonempty string")
    if message is not None and not isinstance(message, str):
        raise ValueError("probe message must be a string or null")
    if message is not None and message.strip():
        return message
    if status != "ok":
        return f"probe reported {status} without a diagnostic"
    return None


@dataclass(frozen=True, slots=True)
class SubprocessResult:
    status: str
    payload: dict[str, Any]
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "message", normalize_probe_message(self.status, self.message))


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
    except OSError as error:
        diagnostic = str(error).strip() or "could not start probe"
        return SubprocessResult("infrastructure_error", {}, f"{type(error).__name__}: {diagnostic}")
    if completed.returncode != 0:
        return SubprocessResult(
            "infrastructure_error",
            {},
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"probe exited with status {completed.returncode} without output",
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
    status = payload.get("status", "ok")
    try:
        message = normalize_probe_message(status, payload.get("message"))
    except ValueError as error:
        return SubprocessResult("infrastructure_error", {}, f"invalid probe JSON: {error}")
    return SubprocessResult(status, dict(MappingProxyType(payload)), message)
