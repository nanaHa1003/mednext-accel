from pathlib import Path

import pytest

from mednext_accel.profiling.batch_search import BatchSearch
from mednext_accel.profiling.campaign import load_campaign


def test_default_campaign_covers_installed_models() -> None:
    campaign = load_campaign(None)
    assert campaign.preset == "all"
    assert {workload.variant for workload in campaign.workloads} == {
        "small",
        "base",
        "medium",
        "large",
    }
    assert campaign.batch_search == BatchSearch()


def test_minimal_yaml_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "workload.yaml"
    path.write_text(
        "preset: mednext-v1\nworkloads:\n  - variant: base\n    spatial: [128, 128, 128]\n"
    )
    campaign = load_campaign(path)
    assert len(campaign.workloads) == 1
    assert campaign.workloads[0].dtypes == ("bfloat16",)
    assert campaign.workloads[0].phases == ("training", "inference")


def test_checkpoint_auto_expands_to_three_user_visible_contexts(tmp_path: Path) -> None:
    path = tmp_path / "workload.yaml"
    path.write_text(
        "workloads:\n  - variant: base\n    spatial: [96, 96, 96]\n    checkpointing: auto\n"
    )
    contexts = {item.checkpointing for item in load_campaign(path).workloads}
    assert contexts == {"none", "all-expansion", "whole-block"}


@pytest.mark.parametrize("obsolete", ["strategy", "dense_until"])
def test_obsolete_batch_search_keys_are_rejected(obsolete: str) -> None:
    with pytest.raises(ValueError, match=rf"unknown batch_search field.*{obsolete}"):
        load_campaign({"batch_search": {obsolete: "unused"}})


def test_batch_search_accepts_only_memory_fraction_and_maximum() -> None:
    campaign = load_campaign({"batch_search": {"memory_fraction": 0.8, "maximum": 12}})
    assert campaign.batch_search == BatchSearch(memory_fraction=0.8, maximum=12)
