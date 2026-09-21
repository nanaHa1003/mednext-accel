import pytest

from mednext_accel.optimization.schema import parse_profile

PROFILE = {
    "schema_version": 1,
    "profile": {
        "name": "test-sm120",
        "target": {"vendor": "nvidia", "sm": [12, 0]},
        "provenance": {"gpu": "test", "torch": "test", "cuda": "test"},
    },
    "defaults": {
        "pointwise_conv3d": {
            "training": {"implementation": "reference", "parameters": {}}
        }
    },
    "rules": [],
    "overrides": [],
}


def test_profile_parses_json_compatible_mapping() -> None:
    profile = parse_profile(PROFILE)
    assert profile.name == "test-sm120"
    assert profile.target_sm == (12, 0)
    assert profile.defaults["pointwise_conv3d"]["training"].implementation == "reference"


def test_unknown_schema_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        parse_profile({"schema_version": 2})


def test_unknown_implementation_is_rejected_with_path() -> None:
    mapping = {**PROFILE, "defaults": {
        "pointwise_conv3d": {"training": {"implementation": "missing", "parameters": {}}}
    }}
    with pytest.raises(ValueError, match="defaults.pointwise_conv3d.training.*missing"):
        parse_profile(mapping)


def test_invalid_range_and_duplicate_rule_are_rejected() -> None:
    rule = {
        "id": "r1", "family": "pointwise_conv3d",
        "match": {"batch": {"min": 4, "max": 2}},
        "phases": {"training": {"implementation": "reference", "parameters": {}}},
    }
    mapping = {**PROFILE, "rules": [rule, rule]}
    with pytest.raises(ValueError, match="min.*max|duplicate"):
        parse_profile(mapping)
