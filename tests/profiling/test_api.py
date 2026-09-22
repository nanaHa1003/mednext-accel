import json

import pytest
import torch

from mednext_accel.profiling import api, cli, runner
from mednext_accel.profiling.campaign import load_campaign
from mednext_accel.profiling.runner import CampaignRun, ExecutionStatistics
from mednext_accel.profiling.synthesize import Measurement


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
        kernel_valid=True,
    )


def test_profile_reports_the_complete_environment_summary(monkeypatch, tmp_path) -> None:
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
        api, "default_policy_path", lambda policy: tmp_path / "sm89-local.policy.yaml"
    )

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
    assert str(tmp_path / "sm89-local.policy.yaml") in reporter.events[1].message


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
    monkeypatch.setattr(api, "default_policy_path", forbidden)
    monkeypatch.setattr(api, "execute_campaign", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(runner, "discover_workload_shapes", forbidden)
    monkeypatch.setattr(runner, "_invoke", forbidden)

    with pytest.raises(ValueError, match=r"unsupported.*dtype.*float32"):
        if entrypoint == "api":
            api.profile(source)
        else:
            cli.main(["profile", str(campaign_path), "--progress", "quiet"])


@pytest.mark.parametrize("entrypoint", ["api", "cli"])
def test_phases_rejected_before_environment_and_cuda(monkeypatch, tmp_path, entrypoint):
    source = {"workloads": [{"phases": ["training"]}]}
    path = tmp_path / "obsolete.yaml"
    path.write_text(json.dumps(source))

    def forbidden(*args, **kwargs):
        pytest.fail("side effects before rejecting phases")

    monkeypatch.setattr(api, "collect_environment", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    with pytest.raises(ValueError, match="phases.*removed.*training"):
        if entrypoint == "api":
            api.profile(source)
        else:
            cli.main(["profile", str(path), "--progress", "quiet"])
