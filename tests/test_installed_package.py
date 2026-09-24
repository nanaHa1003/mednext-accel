from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest


@pytest.mark.release
@pytest.mark.parametrize("checkout", ["copy", "worktree"])
def test_distributions_exclude_local_review_scratch_and_cache_files(
    tmp_path: Path, checkout: str
) -> None:
    repository = Path(__file__).parents[1]
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE", ".gitignore"):
        shutil.copy2(repository / name, project / name)
    shutil.copytree(
        repository / "src", project / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    if checkout == "worktree":
        subprocess.run(["git", "init", "--quiet", str(project)], check=True)
        subprocess.run(["git", "add", "."], cwd=project, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Package Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "-m",
                "Package fixture",
            ],
            cwd=project,
            check=True,
        )
        subprocess.run(["git", "tag", "v0.2.0"], cwd=project, check=True)
        worktree = project / ".worktrees" / "archive"
        subprocess.run(
            ["git", "worktree", "add", "--quiet", "--detach", str(worktree)],
            cwd=project,
            check=True,
        )
        project = worktree
    artifacts = (
        ".superpowers/sdd/review.md",
        "artifacts/integrated-grid/probe.json",
        "dist/final-fix-scratch/probe.json",
        ".pytest_cache/results.json",
        ".ruff_cache/checks.json",
        "src/mednext_accel/__pycache__/local.pyc",
    )
    local_artifacts = tuple(
        f"{directory}{name}"
        for directory in ("", "nested/", "src/mednext_accel/", "src/mednext_accel/nested/")
        for name in (
            "sm89-local.json",
            "sm89-local.123-abc.evidence.json",
            "sm89-local.policy.yaml",
            "sm89-local.policy.yml",
            "sm89-local.policy.YAML",
            "sm89-local.policy.YML",
            "sm89-local.policy.yAmL",
            "sm89-local.policy.yMl",
        )
    )
    for name in artifacts + local_artifacts:
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
        metadata_name = next(name for name in wheel_names if name.endswith(".dist-info/METADATA"))
        built_version = BytesParser().parsebytes(archive.read(metadata_name))["Version"]
    assert built_version == ("0.2.0" if checkout == "worktree" else "0+unknown")
    for names in (sdist_names, wheel_names):
        assert any(name.endswith("mednext_accel/__init__.py") for name in names)
        for profile in ("shared-nvidia", "sm86", "sm89", "sm120"):
            assert any(name.endswith(f"mednext_accel/policies/{profile}.yaml") for name in names)
        assert not any(
            name.lower().endswith(
                ("-local.json", ".evidence.json", "-local.policy.yaml", "-local.policy.yml")
            )
            or "/profiles/" in name
            for name in names
        )
        assert not any(
            part
            in {
                ".superpowers",
                "artifacts",
                "final-fix-scratch",
                ".pytest_cache",
                ".ruff_cache",
                "__pycache__",
            }
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
        env={**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": "0.2.0"},
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
from importlib.metadata import version
from mednext_accel.optimization.policies import load_bundled_policy

assert mednext_accel.__version__ == version("mednext-accel") == "0.2.0"
assert load_bundled_policy("sm89").target_sm == (8, 9)
assert load_bundled_policy("shared-nvidia").rules
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
with torch.device("meta"):
    v2_base = mednext_accel.mednext_v2_base(
        in_channels=1,
        out_channels=3,
        optimization="reference",
    )
    v2_wide = mednext_accel.mednext_v2_wide(in_channels=2, out_channels=8)
assert v2_base.config.variant == "base"
assert v2_base.config.base_channels == 32
assert v2_base.optimization_source == "reference"
assert v2_wide.config.variant == "wide"
assert v2_wide.config.base_channels == 64
assert v2_wide.optimization_source == "auto"
assert 'site-packages' in mednext_accel.__file__
"""
    subprocess.run([str(python), "-c", program], cwd=outside_repository, check=True)


@pytest.mark.parametrize("suffix", ["yaml", "yml", "YAML", "YML", "yAmL", "yMl"])
@pytest.mark.parametrize(
    "directory", ["", "nested/", "src/mednext_accel/", "src/mednext_accel/nested/"]
)
def test_git_ignores_local_publication_artifacts_at_every_supported_location(
    tmp_path, suffix, directory
):
    repository = Path(__file__).parents[1]
    shutil.copy2(repository / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    paths = [
        f"{directory}sm89-local.policy.{suffix}",
        f"{directory}sm89-local.123-abc.evidence.json",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--", *paths],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert set(result.stdout.splitlines()) == set(paths)
