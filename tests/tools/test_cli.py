from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "script",
    [
        "benchmarks/autotune.py",
        "benchmarks/depthwise.py",
        "benchmarks/pointwise.py",
        "tools/profile_train_step.py",
        "tools/profile_matrix.py",
        "tools/summarize_profiles.py",
    ],
)
def test_cli_help_does_not_require_eager_triton_import(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
