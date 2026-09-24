from pathlib import Path

import pytest

from mednext_accel.profiling.batch_search import BatchSearch
from mednext_accel.profiling.campaign import campaign_to_primitive, load_campaign


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
    assert not hasattr(campaign.workloads[0], "phases")


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


def test_campaign_to_primitive_records_every_expanded_workload() -> None:
    campaign = load_campaign(
        {
            "preset": "mednext-v1",
            "objective": "throughput",
            "compile_mode": "fullgraph",
            "batch_search": {"memory_fraction": 0.8, "maximum": 6},
            "workloads": [
                {
                    "variant": "base",
                    "spatial": [96, 128, 160],
                    "dtypes": ["bfloat16"],
                    "checkpointing": "auto",
                    "in_channels": 2,
                    "out_channels": 8,
                }
            ],
        }
    )

    serialized = campaign_to_primitive(campaign)

    assert serialized == {
        "phase": "training",
        "seed": 0,
        "preset": "mednext-v1",
        "objective": "throughput",
        "compile_mode": "fullgraph",
        "batch_search": {"memory_fraction": 0.8, "maximum": 6},
        "workloads": [
            {
                "model_family": "mednext_v1",
                "variant": "base",
                "spatial": [96, 128, 160],
                "dtypes": ["bfloat16"],
                "checkpointing": checkpointing,
                "in_channels": 2,
                "out_channels": 8,
            }
            for checkpointing in ("none", "all-expansion", "whole-block")
        ],
    }


@pytest.mark.parametrize(
    "source", [{"phases": ["training"]}, {"workloads": [{"phases": ["training"]}]}]
)
def test_obsolete_phases_are_rejected(source):
    with pytest.raises(ValueError, match="phases.*removed.*training"):
        load_campaign(source)


def test_unknown_workload_fields_are_rejected_in_sorted_order() -> None:
    with pytest.raises(
        ValueError, match=r"unknown workloads\[1\] field\(s\): base_channels, kernel_size"
    ):
        load_campaign({"workloads": [{}, {"kernel_size": 3, "base_channels": 32}]})


def test_family_alias_defaults_and_preserves_legacy_evidence(tmp_path):
    from mednext_accel.profiling.campaign import Workload

    assert Workload(variant="base", spatial=(32, 32, 32)).family == "mednext_v1"
    assert load_campaign({"workloads": [{}]}).workloads[0].family == "mednext_v1"
    path = tmp_path / "v2.yaml"
    path.write_text("workloads:\n  - family: mednext_v2\n    variant: wide\n")
    campaign = load_campaign(path)
    assert campaign.workloads[0].family == "mednext_v2"
    serialized = campaign_to_primitive(campaign)["workloads"][0]
    assert serialized["model_family"] == "mednext_v2"
    assert "family" not in serialized
    assert (
        load_campaign({"workloads": [{"model_family": "mednext_v2"}]}).workloads[0].variant
        == "base"
    )


def test_matching_family_aliases_are_allowed_and_conflicts_are_precise():
    item = {"family": "mednext_v2", "model_family": "mednext_v2"}
    assert load_campaign({"workloads": [item]}).workloads[0].family == "mednext_v2"
    with pytest.raises(ValueError, match="family.*model_family.*disagree"):
        load_campaign({"workloads": [{**item, "model_family": "mednext_v1"}]})


@pytest.mark.parametrize(
    "family,variant", [("unknown", "base"), ("mednext_v1", "wide"), ("mednext_v2", "small")]
)
def test_family_and_variant_must_be_supported(family, variant):
    with pytest.raises(ValueError, match="unknown.*(family|variant)"):
        load_campaign({"workloads": [{"family": family, "variant": variant}]})


@pytest.mark.parametrize("preset", ["unknown", "mednext-v2"])
def test_invalid_preset_lists_supported_presets_without_claiming_missing_v2(preset):
    with pytest.raises(ValueError) as error:
        load_campaign({"preset": preset})
    assert str(error.value) == (
        f"preset {preset!r} is unavailable; available presets: mednext-v1, all"
    )
    for available in ("mednext-v1", "all"):
        campaign = load_campaign(
            {
                "preset": available,
                "workloads": [{"family": "mednext_v2", "variant": "base"}],
            }
        )
        assert campaign.workloads[0].family == "mednext_v2"
