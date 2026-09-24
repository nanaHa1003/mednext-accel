from __future__ import annotations

from pathlib import Path

import mednext_accel


def test_packaging_version_is_derived_from_vcs() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()

    assert 'dynamic = ["version"]' in pyproject
    assert '[tool.hatch.version]\nsource = "vcs"' in pyproject
    assert 'fallback-version = "0+unknown"' in pyproject
    assert 'version-file = "src/mednext_accel/_version.py"' in pyproject
    assert 'version = "0.1.0a0"' not in pyproject


def test_public_version_matches_installed_distribution_metadata() -> None:
    assert "__version__" in mednext_accel.__all__
    assert isinstance(mednext_accel.__version__, str)
    assert mednext_accel.__version__
