import json
from dataclasses import replace

import pytest

from mednext_accel.optimization.policy import OptimizationPolicy
from mednext_accel.optimization.policy_io import load_policy
from mednext_accel.profiling import api, cli, runner
from mednext_accel.profiling.campaign import load_campaign
from mednext_accel.profiling.runner import CampaignRun, ExecutionStatistics
from mednext_accel.profiling.synthesize import Measurement, synthesize_profile


def measured() -> Measurement:
    return Measurement(
        family="pointwise_conv3d",
        direction="regular",
        phase="training",
        implementation="pointwise_gemm_per_sample",
        batch=1,
        spatial_shape=(32, 32, 32),
        in_channels=4,
        out_channels=8,
        dtype="bfloat16",
        checkpointing="none",
        reference_ms=2.0,
        candidate_ms=1.0,
        reference_peak_bytes=200,
        candidate_peak_bytes=100,
        valid=True,
    )


def test_profile_saves_runtime_policy_atomically_and_keeps_measurements_on_result(
    monkeypatch, tmp_path
) -> None:
    environment = {
        "gpu": {"name": "NVIDIA RTX 5090", "sm": [12, 0]},
        "software": {"torch": "2.9"},
    }
    campaign_run = CampaignRun(
        measurements=(measured(),),
        statistics=ExecutionStatistics(
            requested_case_count=576,
            kernel_case_count=192,
            kernel_group_count=20,
            whole_model_validation_count=2,
        ),
    )
    monkeypatch.setattr(api, "collect_environment", lambda: environment)
    monkeypatch.setattr(api, "execute_campaign", lambda campaign, progress=None: campaign_run)
    monkeypatch.setattr(api.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(api.torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(api, "default_profile_path", lambda profile: tmp_path / "profile.json")

    result = api.profile(load_campaign(None))

    assert result.output_path.exists()
    saved = json.loads(result.output_path.read_text())
    assert saved["name"] == "sm120-local"
    assert saved["version"] == 2
    assert "measurements" not in saved and "defaults" not in saved
    assert load_policy(result.output_path) == result.profile
    assert result.measurements == campaign_run.measurements
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
    monkeypatch.setattr(
        api,
        "execute_campaign",
        lambda campaign, progress=None: CampaignRun((), ExecutionStatistics(0, 0, 0, 0)),
    )
    monkeypatch.setattr(
        api,
        "synthesize_campaign",
        lambda campaign, values, environment=None, execution=None: generated,
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


def test_campaign_synthesis_keeps_adjacent_workload_maxima_as_exact_rules(monkeypatch) -> None:
    campaign = load_campaign(
        {
            "workloads": [
                {
                    "variant": "base",
                    "spatial": [128, 128, 128],
                    "checkpointing": "none",
                    "out_channels": 3,
                },
                {
                    "variant": "base",
                    "spatial": [128, 128, 128],
                    "checkpointing": "none",
                    "out_channels": 8,
                },
            ]
        }
    )
    first = replace(
        measured(),
        batch=2,
        spatial_shape=(128, 128, 128),
        checkpointing="none",
    )
    second = replace(first, batch=3)
    monkeypatch.setattr(api.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(api.torch.cuda, "get_device_capability", lambda: (8, 9))

    generated = api.synthesize_campaign(campaign, (first, second))

    assert [
        (rule.when["batch"].minimum, rule.when["batch"].maximum) for rule in generated.rules
    ] == [
        (2, 2),
        (3, 3),
    ]
    assert [rule.confidence for rule in generated.rules] == [
        "measured-exact-context",
        "measured-exact-context",
    ]


def test_generated_policy_round_trips_through_json(monkeypatch, tmp_path) -> None:
    campaign = load_campaign(None)
    monkeypatch.setattr(api.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(api.torch.cuda, "get_device_capability", lambda: (8, 9))
    generated = api.synthesize_campaign(campaign, (measured(),))
    destination = api.write_profile_atomic(tmp_path / "policy.json", generated)
    assert load_policy(destination) == generated


def test_profiling_result_preserves_three_argument_constructor(tmp_path) -> None:
    generated: OptimizationPolicy = synthesize_profile(
        [], name="test", sm=(8, 9), objective="balanced"
    )
    result = api.ProfilingResult(generated, (), tmp_path / "profile.json")

    assert result.environment == {}


def test_public_run_campaign_remains_tuple_compatible(monkeypatch) -> None:
    campaign = load_campaign(None)
    campaign_run = CampaignRun(
        (measured(),),
        ExecutionStatistics(
            requested_case_count=3,
            kernel_case_count=2,
            kernel_group_count=1,
            whole_model_validation_count=0,
        ),
    )
    monkeypatch.setattr(api, "execute_campaign", lambda value, progress=None: campaign_run)

    measurements = api.run_campaign(campaign)

    assert measurements == campaign_run.measurements
    assert isinstance(measurements, tuple)


def test_api_rejects_obsolete_batch_search_before_environment(monkeypatch) -> None:
    def fail_side_effect(*args, **kwargs):
        pytest.fail("profiling side effect before campaign validation")

    monkeypatch.setattr(api, "collect_environment", fail_side_effect)
    with pytest.raises(ValueError, match=r"unknown batch_search field.*dense_until"):
        api.profile({"batch_search": {"dense_until": 8}})


@pytest.mark.parametrize("entrypoint", ["api", "cli"])
@pytest.mark.parametrize("dtypes", [["float32"], ["bfloat16", "float32"]])
def test_public_profile_rejects_dtypes_before_any_profiling_side_effect(
    monkeypatch, tmp_path, entrypoint, dtypes
) -> None:
    source = {"workloads": [{"variant": "base", "dtypes": dtypes}]}
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(json.dumps(source))

    def forbidden(*args, **kwargs):
        pytest.fail("profiling side effect before dtype validation")

    monkeypatch.setattr(api, "collect_environment", forbidden)
    monkeypatch.setattr(api, "default_profile_path", forbidden)
    monkeypatch.setattr(api, "execute_campaign", forbidden)
    monkeypatch.setattr(api.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(runner, "discover_workload_shapes", forbidden)
    monkeypatch.setattr(runner, "_invoke", forbidden)

    with pytest.raises(ValueError, match=r"unsupported.*dtype.*float32"):
        if entrypoint == "api":
            api.profile(source)
        else:
            cli.main(["profile", str(campaign_path), "--progress", "quiet"])
