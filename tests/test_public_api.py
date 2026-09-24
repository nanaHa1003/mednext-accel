import pytest
import torch

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


def test_whole_block_checkpointing_accepts_resolution_stages() -> None:
    config = CheckpointConfig(style="block", stages=(2, 0, 1))

    assert config.style == "block"
    assert config.stages == (0, 1, 2)
    assert config.checkpoints_blocks
    assert not config.checkpoints_expansion


def test_unknown_checkpoint_style_is_rejected() -> None:
    with pytest.raises(ValueError, match="style"):
        CheckpointConfig(style="whole")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name",
    [
        "MedNeXtV2",
        "MedNeXtV2Config",
        "GlobalResponseNorm3d",
        "get_mednext_v2_config",
        "mednext_v2_base",
        "mednext_v2_wide",
    ],
)
def test_v2_exports_are_available_in_both_namespaces(name):
    import mednext_accel
    import mednext_accel.models as models

    assert getattr(mednext_accel, name) is getattr(models, name)
    assert name in mednext_accel.__all__
    assert name in models.__all__


@pytest.mark.parametrize("variant,channels", [("base", 32), ("wide", 64)])
def test_v2_factories_have_fixed_architecture_and_auto_default(variant, channels):
    import mednext_accel

    factory = getattr(mednext_accel, f"mednext_v2_{variant}")
    with torch.device("meta"):
        model = factory(in_channels=2, out_channels=4)
    assert model.config.base_channels == channels
    assert model.config.kernel_size == 3
    assert model.optimization_source == "auto"
    for option in ["base_channels", "kernel_size", "approximate_gelu_eval", "spatial_dims"]:
        with pytest.raises(TypeError, match=option):
            factory(in_channels=1, out_channels=3, **{option: 2})
