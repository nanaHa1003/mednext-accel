from torch import nn

from mednext_accel import mednext_small
from mednext_accel.profiling.discovery import discover_operators


def test_discovery_deduplicates_and_retains_roles() -> None:
    model = mednext_small(in_channels=1, out_channels=3, base_channels=2, optimization="reference")
    result = discover_operators(model)
    families = {item.descriptor.family for item in result.operators}
    assert {
        "pointwise_conv3d",
        "depthwise_conv3d",
        "depthwise_conv_transpose3d",
        "group_norm",
        "gelu",
    }.issubset(families)
    assert sum(item.occurrences for item in result.operators) > len(result.operators)
    assert result.architecture_fingerprint


def test_group_norm_discovery_has_no_fake_convolution_geometry() -> None:
    result = discover_operators(nn.Sequential(nn.GroupNorm(4, 8)))
    item = next(
        operator for operator in result.operators if operator.descriptor.family == "group_norm"
    )
    assert item.descriptor.kernel_size is None
    assert item.descriptor.stride is None
    assert item.descriptor.padding is None
    assert item.descriptor.dilation is None
    assert item.descriptor.groups is None


def test_grn_discovery_deduplicates_channels_and_retains_roles_without_geometry():
    from mednext_accel.ops.grn import GlobalResponseNorm3d

    result = discover_operators(nn.Sequential(GlobalResponseNorm3d(8), GlobalResponseNorm3d(8)))
    assert len(result.operators) == 1
    item = result.operators[0]
    assert item.descriptor.family == "global_response_norm3d"
    assert item.descriptor.kernel_size is item.descriptor.stride is None
    assert item.descriptor.in_channels == item.descriptor.out_channels == 8
    assert item.occurrences == 2
    assert item.roles == ("0", "1")


def test_v2_base_and_wide_discover_unique_grn_channel_spatial_shapes():
    from mednext_accel.profiling.campaign import Workload
    from mednext_accel.profiling.runner import discover_workload_shapes

    base = discover_workload_shapes(Workload("mednext_v2", "base", (32, 32, 32)))
    wide = discover_workload_shapes(Workload("mednext_v2", "wide", (32, 32, 32)))
    assert base.grn
    assert len(base.grn) == len(set(base.grn))
    assert (96, (32, 32, 32)) in base.grn
    assert (4096, (2, 2, 2)) in base.grn
    assert wide.grn == tuple(sorted((2 * channels, spatial) for channels, spatial in base.grn))
    assert base.pointwise and base.depthwise


def test_v2_discovery_supports_single_voxel_bottleneck_workloads():
    from mednext_accel.profiling.campaign import Workload
    from mednext_accel.profiling.runner import discover_workload_shapes

    shapes = discover_workload_shapes(Workload("mednext_v2", "base", (16, 16, 16)))

    assert shapes.pointwise
    assert shapes.depthwise
    assert (4096, (1, 1, 1)) in shapes.grn


def test_v2_discovery_retains_all_repeated_block_grn_roles():
    from mednext_accel.profiling.campaign import Workload
    from mednext_accel.profiling.runner import _meta_model

    for variant in ("base", "wide"):
        model = _meta_model(Workload("mednext_v2", variant, (32, 32, 32)))
        operators = [
            item
            for item in discover_operators(model).operators
            if item.descriptor.family == "global_response_norm3d"
        ]
        assert sum(item.occurrences for item in operators) == 62
        assert all(item.occurrences == len(item.roles) for item in operators)
        assert any(item.occurrences > 1 for item in operators)
        assert len({role for item in operators for role in item.roles}) == 62
