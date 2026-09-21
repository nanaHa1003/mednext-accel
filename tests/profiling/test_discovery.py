from mednext_accel import mednext_small
from mednext_accel.profiling.discovery import discover_operators


def test_discovery_deduplicates_and_retains_roles() -> None:
    model = mednext_small(
        in_channels=1, out_channels=3, base_channels=2, optimization="reference"
    )
    result = discover_operators(model)
    families = {item.descriptor.family for item in result.operators}
    assert {
        "pointwise_conv3d", "depthwise_conv3d", "depthwise_conv_transpose3d",
        "group_norm", "gelu",
    }.issubset(families)
    assert sum(item.occurrences for item in result.operators) > len(result.operators)
    assert result.architecture_fingerprint
