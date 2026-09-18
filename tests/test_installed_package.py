from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest


@pytest.mark.release
def test_wheel_installs_and_runs_outside_source_tree(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    wheel_dir = tmp_path / "wheelhouse"
    environment = tmp_path / "venv"
    outside_repository = tmp_path / "consumer"
    outside_repository.mkdir()

    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_dir)],
        cwd=repository,
        check=True,
    )
    wheel = next(wheel_dir.glob("mednext_accel-*.whl"))

    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        check=True,
    )
    program = """
import torch
import mednext_accel

model = mednext_accel.mednext_small(in_channels=1, out_channels=3).eval()
with torch.no_grad():
    output = model(torch.randn(1, 1, 32, 32, 32))
assert output.shape == (1, 3, 32, 32, 32)
assert 'site-packages' in mednext_accel.__file__
"""
    subprocess.run([str(python), "-c", program], cwd=outside_repository, check=True)
