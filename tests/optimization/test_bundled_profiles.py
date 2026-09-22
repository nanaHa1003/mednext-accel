from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor
from mednext_accel.optimization.profiles import load_bundled_profile
from mednext_accel.optimization.resolver import OptimizationResolver


def pointwise_descriptor():
    return OperatorDescriptor(
        "pointwise_conv3d",
        "regular",
        32,
        64,
        (1, 1, 1),
        (1, 1, 1),
        (0, 0, 0),
        (1, 1, 1),
        1,
    )


def cuda_context(batch: int, spatial: int):
    return ExecutionContext(
        "training",
        "cuda",
        (12, 0),
        32 * 2**30,
        "bfloat16",
        batch,
        (spatial,) * 3,
        "mednext_v1",
        "base",
        "all-expansion",
    )


def test_bundled_profiles_parse_and_have_family_defaults() -> None:
    for name in ("generic-nvidia", "sm120", "sm89"):
        profile = load_bundled_profile(name)
        assert profile.name == name
        assert "pointwise_conv3d" in profile.defaults
        assert "depthwise_conv3d" in profile.defaults


def test_sm120_resolves_supported_batch_and_volume_grid() -> None:
    resolver = OptimizationResolver((load_bundled_profile("sm120"),))
    for batch in range(1, 9):
        for spatial in (96, 128, 160, 192):
            context = cuda_context(batch, spatial)
            decision = resolver.resolve(pointwise_descriptor(), context, "training")
            assert decision.implementation in ("reference", "pointwise_gemm_per_sample")
            assert decision.confidence in {
                "measured",
                "interpolated",
                "extrapolated",
                "default",
                "guard",
            }
