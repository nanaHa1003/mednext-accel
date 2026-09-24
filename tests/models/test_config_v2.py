from __future__ import annotations

import pytest

from mednext_accel.models.config_v2 import MedNeXtV2Config, get_mednext_v2_config


@pytest.mark.parametrize("variant,channels", [("base", 32), ("wide", 64)])
def test_published_v2_configuration(variant: str, channels: int) -> None:
    config = get_mednext_v2_config(variant, in_channels=1, out_channels=3)

    assert config.base_channels == channels
    assert config.block_counts == (3, 4, 8, 8, 8, 8, 8, 4, 3)
    assert config.expansion_ratios == (3, 4, 8, 8, 8, 8, 8, 4, 3)
    assert config.downsample_expansion_ratios == (4, 8, 8, 8)
    assert config.upsample_expansion_ratios == (8, 8, 4, 3)
    assert config.kernel_size == 3


def test_v2_config_normalizes_yaml_lists_to_tuples() -> None:
    config = MedNeXtV2Config(
        variant="base",
        in_channels=1,
        out_channels=3,
        base_channels=32,
        kernel_size=3,
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        expansion_ratios=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        downsample_expansion_ratios=[4, 8, 8, 8],
        upsample_expansion_ratios=[8, 8, 4, 3],
    )

    assert config.block_counts == (3, 4, 8, 8, 8, 8, 8, 4, 3)
    assert config.expansion_ratios == (3, 4, 8, 8, 8, 8, 8, 4, 3)
    assert config.downsample_expansion_ratios == (4, 8, 8, 8)
    assert config.upsample_expansion_ratios == (8, 8, 4, 3)
    assert all(
        isinstance(values, tuple)
        for values in (
            config.block_counts,
            config.expansion_ratios,
            config.downsample_expansion_ratios,
            config.upsample_expansion_ratios,
        )
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_channels": 0},
        {"out_channels": 0},
        {"base_channels": 0},
        {"block_counts": (3,) * 8},
        {"expansion_ratios": (3,) * 8},
        {"downsample_expansion_ratios": (4,) * 3},
        {"upsample_expansion_ratios": (4,) * 3},
    ],
)
def test_v2_config_rejects_invalid_schema(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "variant": "base",
        "in_channels": 1,
        "out_channels": 3,
        "base_channels": 32,
        "kernel_size": 3,
        "block_counts": (3,) * 9,
        "expansion_ratios": (3,) * 9,
        "downsample_expansion_ratios": (4,) * 4,
        "upsample_expansion_ratios": (4,) * 4,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        MedNeXtV2Config(**values)  # type: ignore[arg-type]


def test_v2_config_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variant"):
        get_mednext_v2_config("large", in_channels=1, out_channels=3)
