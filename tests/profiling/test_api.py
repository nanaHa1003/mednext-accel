import json

from mednext_accel.profiling import api
from mednext_accel.profiling.campaign import load_campaign
from mednext_accel.profiling.synthesize import synthesize_profile


def test_profile_saves_one_atomic_result(monkeypatch, tmp_path) -> None:
    generated = synthesize_profile([], name="sm120-local", sm=(12, 0), objective="balanced")
    monkeypatch.setattr(api, "run_campaign", lambda campaign: ())
    monkeypatch.setattr(api, "synthesize_campaign", lambda campaign, values: generated)
    monkeypatch.setattr(api, "default_profile_path", lambda profile: tmp_path / "profile.json")
    result = api.profile(load_campaign(None))
    assert result.output_path.exists()
    assert json.loads(result.output_path.read_text())["profile"]["name"] == "sm120-local"
