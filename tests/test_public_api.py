import pytest

from mednext_accel import CheckpointConfig
from mednext_accel.models.config import get_mednext_v1_config


def test_base_config_matches_official_v1() -> None:
    config = get_mednext_v1_config("base", in_channels=1, out_channels=3)

    assert config.block_counts == (2,) * 9
    assert config.expansion_ratios == (2, 3, 4, 4, 4, 4, 4, 3, 2)
    assert config.downsample_expansion_ratios == (3, 4, 4, 4)


def test_monai_base_config_records_its_downsample_difference() -> None:
    config = get_mednext_v1_config("base", in_channels=1, out_channels=3, compatibility="monai")

    assert config.downsample_expansion_ratios == (2, 3, 4, 4)


@pytest.mark.parametrize(
    "stages, message",
    [
        ((0, 0), "duplicates"),
        ((-1,), "between 0 and 4"),
        ((5,), "between 0 and 4"),
        ((1.0,), "integers"),
    ],
)
def test_checkpoint_stages_are_validated(stages: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CheckpointConfig(stages=stages)


def test_checkpoint_stages_are_canonicalized() -> None:
    config = CheckpointConfig(stages=(2, 0, 1))

    assert config.stages == (0, 1, 2)


def test_checkpoint_config_rejects_disabled_expansion_with_stages() -> None:
    with pytest.raises(ValueError, match="expansion"):
        CheckpointConfig(expansion=False, stages=(0,))
