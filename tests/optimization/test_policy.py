from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from mednext_accel.optimization import optimize, use_backend
from mednext_accel.optimization.policy import conservative_selections


def test_torch_policy_is_a_noop_for_reference_model() -> None:
    model = nn.Sequential(nn.Conv3d(2, 4, 1))
    keys = tuple(model.state_dict())

    report = optimize(
        model,
        input_shape=(1, 2, 8, 8, 8),
        dtype=torch.float32,
        policy="torch",
    )

    assert report.policy == "torch"
    assert report.compile_mode == "default"
    assert report.selections.is_empty
    assert report.replacements == 0
    assert tuple(model.state_dict()) == keys


def test_compile_mode_is_validated_and_recorded() -> None:
    model = nn.Sequential(nn.Conv3d(2, 4, 1))

    report = optimize(
        model,
        input_shape=(1, 2, 8, 8, 8),
        dtype=torch.float32,
        policy="torch",
        compile_mode="max-autotune-no-cudagraphs",
    )

    assert report.compile_mode == "max-autotune-no-cudagraphs"
    with pytest.raises(ValueError, match="torch.compile mode"):
        optimize(
            model,
            input_shape=(1, 2, 8, 8, 8),
            dtype=torch.float32,
            policy="torch",
            compile_mode="fastest",  # type: ignore[arg-type]
        )


def test_conservative_table_is_limited_to_validated_blackwell_case() -> None:
    selected = conservative_selections(
        device_type="cuda",
        capability=(12, 0),
        dtype=torch.bfloat16,
        input_shape=(1, 1, 128, 128, 128),
    )
    unsupported = conservative_selections(
        device_type="cuda",
        capability=(8, 9),
        dtype=torch.bfloat16,
        input_shape=(1, 1, 128, 128, 128),
    )

    assert selected.depthwise_regular == (
        (32, 128),
        (64, 64),
        (128, 32),
        (256, 16),
        (512, 8),
    )
    assert selected.depthwise_transpose == ((64, 64),)
    assert selected.pointwise_gemm == ()
    assert selected.depthwise_downsample == ()
    assert unsupported.is_empty


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_conservative_policy_preserves_parameters_and_can_restore_torch() -> None:
    model = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1, groups=32)).cuda()
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())

    report = optimize(
        model,
        input_shape=(1, 32, 128, 128, 128),
        dtype=torch.bfloat16,
        policy="conservative",
    )

    assert report.selections.depthwise_regular
    assert report.replacements == 1
    assert tuple(model.parameters()) == parameters
    assert tuple(model.state_dict()) == keys
    wrapper = model[0]
    selected_before = wrapper.selected_shapes
    with use_backend(model, "torch"):
        assert wrapper.selected_shapes == frozenset()
    assert wrapper.selected_shapes == selected_before


def test_backend_context_restores_policy_after_exception() -> None:
    from mednext_accel.ops.pointwise import GemmPointwise3d

    model = nn.Sequential(GemmPointwise3d(nn.Conv3d(2, 4, 1), selected_shapes={(2, 4, 8, 8, 8)}))
    original = copy.copy(model[0].selected_shapes)

    with pytest.raises(RuntimeError, match="stop"):
        with use_backend(model, "torch"):
            assert model[0].selected_shapes == frozenset()
            raise RuntimeError("stop")

    assert model[0].selected_shapes == original
