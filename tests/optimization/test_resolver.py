import warnings

from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor
from mednext_accel.optimization.profiles import ProfileRegistry, load_bundled_profile
from mednext_accel.optimization.resolver import OptimizationResolver, scale_work


def pointwise_descriptor(in_channels: int = 32, out_channels: int = 64):
    return OperatorDescriptor(
        family="pointwise_conv3d", direction="regular",
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
        dilation=(1, 1, 1), groups=1,
    )


def cuda_context(*, batch=3, sm=(12, 0), vram=32, approximate=False):
    return ExecutionContext(
        phase="training", device_type="cuda", sm=sm,
        total_vram_bytes=vram * 2**30, dtype="bfloat16", batch_size=batch,
        spatial_shape=(128, 128, 128), model_family="mednext_v1",
        variant="base", checkpointing="all-expansion",
        allow_approximate=approximate,
    )


def test_batch_three_uses_batch_two_to_four_rule() -> None:
    resolver = OptimizationResolver((load_bundled_profile("sm120"),))
    decision = resolver.resolve(pointwise_descriptor(), cuda_context(), phase="training")
    assert decision.implementation == "pointwise_gemm_per_sample"
    assert decision.confidence == "interpolated"


def test_unknown_sm_uses_generic_profile_and_warns_once() -> None:
    registry = ProfileRegistry()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolver = registry.resolver_for(sm=(9, 0))
        first = resolver.resolve(pointwise_descriptor(), cuda_context(sm=(9, 0)), "training")
        second = registry.resolver_for(sm=(9, 0))
        second.resolve(pointwise_descriptor(64, 32), cuda_context(sm=(9, 0)), "training")
    assert first.profile == "generic-nvidia"
    assert first.warning is not None
    assert len(caught) == 1


def test_every_descriptor_gets_a_default_decision() -> None:
    resolver = OptimizationResolver((load_bundled_profile("sm120"),))
    descriptor = pointwise_descriptor(7, 13)
    assert resolver.resolve(descriptor, cuda_context(batch=17), "training").implementation


def test_scale_work_clamps_and_rounds_to_multiple() -> None:
    assert scale_work(8, 100, 200, minimum=4, maximum=12, multiple=4) == 12


def test_non_export_safe_selection_is_guarded() -> None:
    resolver = OptimizationResolver((load_bundled_profile("sm120"),))
    context = cuda_context(batch=3)
    decision = resolver.resolve(pointwise_descriptor(), context, phase="export")
    assert decision.implementation == "reference"


def test_approximate_selection_requires_opt_in() -> None:
    descriptor = OperatorDescriptor(
        family="gelu", direction="forward", in_channels=32, out_channels=32,
        kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
        dilation=(1, 1, 1), groups=1,
    )
    resolver = OptimizationResolver((load_bundled_profile("sm120"),))
    assert resolver.resolve(descriptor, cuda_context(), "inference").implementation == "reference"
    assert resolver.resolve(
        descriptor, cuda_context(approximate=True), "inference"
    ).implementation == "gelu_tanh"
