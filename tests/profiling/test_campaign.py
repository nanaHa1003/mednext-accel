from pathlib import Path

from mednext_accel.profiling.campaign import load_campaign


def test_default_campaign_covers_installed_models() -> None:
    campaign = load_campaign(None)
    assert campaign.preset == "all"
    assert {workload.variant for workload in campaign.workloads} == {
        "small", "base", "medium", "large"
    }
    assert campaign.batch_search.strategy == "auto"
    assert campaign.batch_search.memory_fraction == 0.90
    assert campaign.batch_search.dense_until == 8


def test_minimal_yaml_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "workload.yaml"
    path.write_text(
        "preset: mednext-v1\nworkloads:\n  - variant: base\n"
        "    spatial: [128, 128, 128]\n"
    )
    campaign = load_campaign(path)
    assert len(campaign.workloads) == 1
    assert campaign.workloads[0].dtypes == ("bfloat16",)
    assert campaign.workloads[0].phases == ("training", "inference")


def test_checkpoint_auto_expands_to_three_user_visible_contexts(tmp_path: Path) -> None:
    path = tmp_path / "workload.yaml"
    path.write_text(
        "workloads:\n  - variant: base\n    spatial: [96, 96, 96]\n"
        "    checkpointing: auto\n"
    )
    contexts = {item.checkpointing for item in load_campaign(path).workloads}
    assert contexts == {"none", "all-expansion", "whole-block"}
