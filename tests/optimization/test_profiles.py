import json
from pathlib import Path

import pytest
import yaml

from mednext_accel.optimization.profiles import load_profile

PROFILE = {
    "schema_version": 1,
    "profile": {
        "name": "test-sm120",
        "target": {"vendor": "nvidia", "sm": [12, 0]},
        "provenance": {},
    },
    "defaults": {
        "pointwise_conv3d": {
            "training": {"implementation": "reference", "parameters": {}}
        }
    },
    "rules": [],
    "overrides": [],
}


def test_yaml_path_and_mapping_produce_the_same_profile(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(PROFILE))
    assert load_profile(path) == load_profile(PROFILE)


def test_json_path_loads(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(PROFILE))
    assert load_profile(path).name == "test-sm120"


def test_loader_rejects_modes_and_unknown_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="factory mode"):
        load_profile("auto")
    path = tmp_path / "profile.toml"
    path.write_text("")
    with pytest.raises(ValueError, match="suffix"):
        load_profile(path)


def test_loader_expands_home_paths(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(PROFILE))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert load_profile("~/profile.json").name == "test-sm120"
