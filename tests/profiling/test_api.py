import json

from mednext_accel.optimization.schema import OptimizationProfile
from mednext_accel.profiling import api
from mednext_accel.profiling.campaign import load_campaign
from mednext_accel.profiling.synthesize import synthesize_profile


def test_profile_saves_one_atomic_result(monkeypatch, tmp_path) -> None:
    generated = synthesize_profile([], name="sm120-local", sm=(12, 0), objective="balanced")
    environment = {
        "gpu": {"name": "NVIDIA RTX 5090", "sm": [12, 0]},
        "software": {"torch": "2.9"},
    }
    monkeypatch.setattr(api, "collect_environment", lambda: environment)
    monkeypatch.setattr(api, "run_campaign", lambda campaign, progress=None: ())
    monkeypatch.setattr(
        api,
        "synthesize_campaign",
        lambda campaign, values, environment=None: generated,
    )
    monkeypatch.setattr(api, "default_profile_path", lambda profile: tmp_path / "profile.json")
    result = api.profile(load_campaign(None))
    assert result.output_path.exists()
    assert json.loads(result.output_path.read_text())["profile"]["name"] == "sm120-local"
    assert result.environment == environment


def test_profile_reports_the_complete_environment_summary(monkeypatch, tmp_path) -> None:
    generated = synthesize_profile([], name="sm89-local", sm=(8, 9), objective="balanced")
    environment = {
        "python": "3.11.13",
        "platform": "Linux-6.8-x86_64",
        "driver": "580.65.06",
        "gpu": {
            "name": "NVIDIA L40S",
            "sm": [8, 9],
            "total_memory_bytes": 48 * 1024**3,
        },
        "software": {
            "mednext_accel": "0.1.0a0",
            "torch": "2.9.0",
            "cuda": "12.8",
            "cudnn": 90100,
            "triton": "3.5.0",
        },
    }

    class Reporter:
        def __init__(self) -> None:
            self.events = []

        def emit(self, event) -> None:
            self.events.append(event)

        def close(self) -> None:
            pass

    reporter = Reporter()
    monkeypatch.setattr(api, "collect_environment", lambda: environment)
    monkeypatch.setattr(api, "run_campaign", lambda campaign, progress=None: ())
    monkeypatch.setattr(
        api,
        "synthesize_campaign",
        lambda campaign, values, environment=None: generated,
    )
    monkeypatch.setattr(api, "default_profile_path", lambda profile: tmp_path / "profile.json")

    api.profile(load_campaign(None), progress=reporter)

    summary = reporter.events[0].message
    for value in (
        "NVIDIA L40S",
        "SM89",
        "48.0 GiB",
        "driver 580.65.06",
        "Python 3.11.13",
        "mednext-accel 0.1.0a0",
        "PyTorch 2.9.0",
        "CUDA 12.8",
        "cuDNN 90100",
        "Triton 3.5.0",
        "objective balanced",
        "compile max-autotune-no-cudagraphs",
    ):
        assert value in summary
    assert reporter.events[1].stage == "output"
    assert str(tmp_path / "profile.json") in reporter.events[1].message


def test_synthesized_profile_embeds_environment_in_provenance() -> None:
    campaign = load_campaign(None)
    environment = {"gpu": {"name": "NVIDIA L40S", "sm": [8, 9]}}
    generated = api.synthesize_campaign(campaign, (), environment=environment)

    assert generated.provenance["environment"]["gpu"]["name"] == "NVIDIA L40S"
    assert generated.provenance["compile_mode"] == "max-autotune-no-cudagraphs"


def test_profiling_result_preserves_three_argument_constructor(tmp_path) -> None:
    generated: OptimizationProfile = synthesize_profile(
        [], name="test", sm=(8, 9), objective="balanced"
    )
    result = api.ProfilingResult(generated, (), tmp_path / "profile.json")

    assert result.environment == {}
