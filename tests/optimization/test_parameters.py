import csv
from dataclasses import replace
from pathlib import Path

import pytest

from mednext_accel.optimization import parameters
from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor
from mednext_accel.optimization.parameters import parameter_defaults as defaults
from mednext_accel.optimization.policy import PolicySelection


def descriptor(*, transpose=False, channels=128):
    return OperatorDescriptor(
        family="depthwise_conv_transpose3d" if transpose else "depthwise_conv3d",
        direction="transpose" if transpose else "regular",
        in_channels=channels,
        out_channels=channels,
        groups=channels,
        kernel_size=(3, 3, 3),
        stride=(2, 2, 2) if transpose else (1, 1, 1),
        padding=(1, 1, 1),
        dilation=(1, 1, 1),
    )


def context(*, batch=3, shape=(32, 32, 32), sm=(8, 9)):
    return ExecutionContext(
        "training",
        "cuda",
        sm,
        48 * 2**30,
        "bfloat16",
        batch,
        shape,
        "mednext_v1",
        "base",
        "all-expansion",
    )


@pytest.mark.parametrize("sm", [(8, 6), (8, 9), (9, 0)])
@pytest.mark.parametrize(
    "implementation,batch,shape,want",
    [
        ("triton_split_dw", 3, (32, 32, 32), 3),
        ("triton_split_dw", 5, (16, 32, 32), 2),  # 2.5 rounds to the even integer.
        ("triton_split_dw", 7, (16, 32, 32), 4),
        ("triton_split_dw", 1, (8, 8, 8), 1),
        ("triton_split_dw", 9, (128, 128, 128), 512),
        ("triton_transpose_split_dw", 3, (16, 16, 16), 3),
        ("triton_transpose_split_dw", 1, (8, 8, 8), 1),
        ("triton_transpose_split_dw", 10, (64, 64, 64), 512),
    ],
)
def test_reduction_recipes_use_full_spatial_volume_and_clamped_even_rounding(
    sm, implementation, batch, shape, want
):
    result = defaults(
        implementation,
        descriptor(transpose="transpose" in implementation),
        context(batch=batch, shape=shape, sm=sm),
    )
    assert result == {"dw_splits": want, "dw_block": 512}


@pytest.mark.parametrize("implementation", ["triton_depthwise_dx", "triton_downsample_dx"])
def test_dx_recipe_uses_preserved_block_size(implementation):
    assert defaults(implementation, descriptor(), context()) == {"dx_block": 128}


with (Path(__file__).parent / "fixtures" / "legacy_dw_launches.csv").open() as stream:
    LAUNCHES = list(csv.DictReader(stream))


@pytest.mark.parametrize(
    "row", LAUNCHES, ids=lambda row: f"{row['profile']}-{row['rule']}-b{row['batch']}"
)
def test_recipes_and_only_real_exceptions_reproduce_legacy_launch_snapshot(row):
    implementation = row["implementation"]
    operator = descriptor(transpose="transpose" in implementation, channels=int(row["channels"]))
    execution = context(
        batch=int(row["batch"]),
        shape=(int(row["size"]),) * 3,
        sm=(8, 9) if row["profile"] == "sm89" else (12, 0),
    )
    expected = {"dw_splits": int(row["dw_splits"]), "dw_block": int(row["dw_block"])}
    result = defaults(implementation, operator, execution)
    exceptions = {"dw-split-b4-128-32": 4, "dw-split-b6-128-32": 8}
    if row["section"] == "overrides" and row["rule"] in exceptions:
        assert result != expected
        selection = PolicySelection(implementation, {"dw_splits": exceptions[row["rule"]]})
        result = parameters.resolve_parameters(selection, operator, execution)
    assert result == expected


def test_recipe_auto_explicit_override_and_omission_have_distinct_meanings():
    defaults("triton_split_dw", descriptor(), context())
    selections = [
        (PolicySelection("triton_split_dw", "auto"), {"dw_splits": 3, "dw_block": 512}),
        (PolicySelection("triton_split_dw", {}), {"dw_splits": 3, "dw_block": 512}),
        (PolicySelection("triton_split_dw", {"dw_splits": 2}), {"dw_splits": 2, "dw_block": 512}),
        (PolicySelection("triton_split_dw"), {}),
        (PolicySelection("reference", {"tag": "literal"}), {"tag": "literal"}),
    ]
    for selection, want in selections:
        assert parameters.resolve_parameters(selection, descriptor(), context()) == want


def test_unknown_recipe_and_invalid_auto_raise_clear_errors():
    defaults("triton_split_dw", descriptor(), context())
    with pytest.raises(ValueError, match="recipe"):
        parameters.parameter_defaults("reference", descriptor(), context())
    with pytest.raises(ValueError, match="recipe"):
        parameters.resolve_parameters(PolicySelection("reference", "auto"), descriptor(), context())


def test_sm120_recipe_is_shape_and_channel_specific_and_scales_above_measured_batches():
    execution = context(sm=(12, 0), shape=(32, 32, 32), batch=24)
    assert defaults("triton_split_dw", descriptor(), execution) == {
        "dw_splits": 48,
        "dw_block": 1024,
    }
    assert defaults("triton_split_dw", descriptor(channels=64), execution) == {
        "dw_splits": 24,
        "dw_block": 512,
    }
    rectangular = replace(execution, spatial_shape=(16, 32, 64))
    assert defaults("triton_split_dw", descriptor(), rectangular) == {
        "dw_splits": 24,
        "dw_block": 512,
    }
