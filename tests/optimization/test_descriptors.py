import pytest

from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor


def test_descriptor_has_a_stable_primitive_identity() -> None:
    descriptor = OperatorDescriptor(
        family="depthwise_conv3d",
        direction="regular",
        in_channels=32,
        out_channels=32,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        dilation=(1, 1, 1),
        groups=32,
    )
    assert descriptor.to_primitive()["family"] == "depthwise_conv3d"
    assert descriptor.signature == ("depthwise_conv3d", "regular", 32, 32, (3, 3, 3), (1, 1, 1))


def test_positional_convolution_descriptor_retains_legacy_primitive_shape() -> None:
    descriptor = OperatorDescriptor(
        "depthwise_conv3d",
        "regular",
        32,
        32,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        32,
    )
    assert descriptor.to_primitive() == {
        "family": "depthwise_conv3d",
        "direction": "regular",
        "in_channels": 32,
        "out_channels": 32,
        "kernel_size": [3, 3, 3],
        "stride": [1, 1, 1],
        "padding": [1, 1, 1],
        "dilation": [1, 1, 1],
        "groups": 32,
        "role": None,
    }


def test_normalization_descriptor_has_no_fake_convolution_geometry() -> None:
    item = OperatorDescriptor.normalization(
        family="global_response_norm3d", channels=96, role="encoder_stages.0.0.grn"
    )
    assert item.kernel_size is None
    assert item.groups is None
    assert item.to_primitive() == {
        "family": "global_response_norm3d",
        "direction": "regular",
        "in_channels": 96,
        "out_channels": 96,
        "role": "encoder_stages.0.0.grn",
    }


def test_convolution_descriptor_retains_geometry_validation() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        OperatorDescriptor.convolution(
            family="depthwise_conv3d",
            direction="regular",
            in_channels=4,
            out_channels=4,
            kernel_size=(0, 0, 0),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            dilation=(1, 1, 1),
            groups=4,
        )


def test_execution_context_records_runtime_resolution_inputs() -> None:
    context = ExecutionContext(
        phase="training",
        device_type="cuda",
        sm=(12, 0),
        total_vram_bytes=32 * 2**30,
        dtype="bfloat16",
        batch_size=3,
        spatial_shape=(128, 128, 128),
        model_family="mednext_v1",
        variant="base",
        checkpointing="all-expansion",
    )
    assert context.spatial_volume == 128**3
    assert context.total_vram_gib == 32


def test_descriptors_reject_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        ExecutionContext(
            phase="training",
            device_type="cpu",
            sm=None,
            total_vram_bytes=0,
            dtype="float32",
            batch_size=0,
            spatial_shape=(8, 8, 8),
            model_family="mednext_v1",
            variant="base",
            checkpointing="none",
        )
