from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from pathlib import Path

import pytest


@pytest.mark.release
def test_distributions_exclude_local_review_scratch_and_cache_files(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE", ".gitignore"):
        shutil.copy2(repository / name, project / name)
    shutil.copytree(
        repository / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    artifacts = (
        ".superpowers/sdd/review.md",
        "dist/final-fix-scratch/probe.json",
        ".pytest_cache/results.json",
        ".ruff_cache/checks.json",
        "src/mednext_accel/__pycache__/local.pyc",
    )
    for name in artifacts:
        artifact = project / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("local artifact must not ship")
    (project / ".superpowers/sdd/.gitignore").write_text("*\n")
    output = tmp_path / "distributions"
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output)], cwd=project, check=True
    )

    with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
        sdist_names = archive.getnames()
    with zipfile.ZipFile(next(output.glob("*.whl"))) as archive:
        wheel_names = archive.namelist()
    for names in (sdist_names, wheel_names):
        assert any(name.endswith("mednext_accel/__init__.py") for name in names)
        assert not any(
            part
            in {".superpowers", "final-fix-scratch", ".pytest_cache", ".ruff_cache", "__pycache__"}
            for name in names
            for part in Path(name).parts
        ), names


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
report = model.explain_optimization(
    input_shape=(3, 1, 128, 128, 128),
    dtype="bfloat16",
    device="cpu",
)
assert report.decisions
assert 'site-packages' in mednext_accel.__file__
"""
    subprocess.run([str(python), "-c", program], cwd=outside_repository, check=True)
