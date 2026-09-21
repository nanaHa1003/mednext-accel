from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mednext_accel.profiling import cli

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "script",
    [
        "benchmarks/depthwise.py",
        "benchmarks/implementations.py",
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


@pytest.mark.parametrize("script", ["benchmarks/depthwise.py", "benchmarks/pointwise.py"])
def test_kernel_benchmark_accepts_batch_size(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--batch-size" in result.stdout


@pytest.mark.parametrize("arguments", [["--help"], ["profile", "--help"]])
def test_package_cli_help_does_not_import_cuda(arguments, capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments)
    assert caught.value.code == 0
    assert "usage:" in capsys.readouterr().out.lower()


def test_zero_argument_profile_uses_default_campaign(monkeypatch, capsys) -> None:
    seen = {}

    class Reporter:
        def close(self) -> None:
            pass

    reporter = Reporter()

    class Result:
        output_path = "generated.json"
        measurements = ()

    monkeypatch.setattr(
        cli,
        "profile",
        lambda source=None, progress=None: (
            seen.update(source=source, progress=progress) or Result()
        ),
    )
    monkeypatch.setattr(cli, "create_progress_reporter", lambda mode: reporter)
    assert cli.main(["profile", "--progress", "plain"]) == 0
    assert seen == {"source": None, "progress": reporter}
    assert "generated.json" in capsys.readouterr().out
