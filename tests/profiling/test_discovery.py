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
