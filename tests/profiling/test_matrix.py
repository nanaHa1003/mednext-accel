from mednext_accel.profiling.campaign import Workload
from mednext_accel.profiling.matrix import build_workload_cases, deduplicate_cases, group_cases


def test_same_kernel_across_checkpoint_contexts_is_measured_once() -> None:
    none = Workload("mednext_v1", "base", (128, 128, 128), checkpointing="none")
    expansion = Workload("mednext_v1", "base", (128, 128, 128), checkpointing="all-expansion")
    shapes = ((32, 64, (128, 128, 128)),)
    first = build_workload_cases(none, (1,), pointwise_shapes=shapes, depthwise_shapes=())
    second = build_workload_cases(expansion, (1,), pointwise_shapes=shapes, depthwise_shapes=())

    unique, references = deduplicate_cases((first, second))

    assert len(unique) == 1
    assert references[0] == references[1]


def test_batch_phase_shape_and_launch_parameters_remain_in_identity() -> None:
    workload = Workload("mednext_v1", "base", (128, 128, 128))
    cases = build_workload_cases(
        workload,
        (1, 2),
        pointwise_shapes=((32, 64, (128, 128, 128)),),
        depthwise_shapes=(("regular", 32, 3, (128, 128, 128)),),
    )
    assert len({case.key for case in cases}) == 6


def test_cases_are_grouped_into_five_categories_per_batch() -> None:
    workload = Workload("mednext_v1", "base", (128, 128, 128))
    cases = build_workload_cases(
        workload,
        (1,),
        pointwise_shapes=((32, 64, (128, 128, 128)),),
        depthwise_shapes=(
            ("regular", 32, 3, (128, 128, 128)),
            ("downsample", 64, 3, (64, 64, 64)),
            ("transpose", 64, 3, (32, 32, 32)),
        ),
    )
    groups = group_cases(cases)

    assert {group.category for group in groups} == {
        "pointwise",
        "regular-dx",
        "regular-dw",
        "downsample-dx",
        "transpose-dw",
    }
    assert all(len({case.key.batch for case in group.cases}) == 1 for group in groups)


def test_profiling_launch_recipe_uses_all_spatial_dimensions():
    from mednext_accel.profiling.matrix import depthwise_parameters

    assert dict(depthwise_parameters("regular", "backward_weight", 5, (16, 32, 32))) == {
        "dw_splits": 2,
        "dw_block": 512,
    }


def test_execution_sm_selects_architecture_recipe_in_kernel_plan():
    from mednext_accel.profiling.matrix import build_workload_cases

    workload = Workload("mednext_v1", "base", (128, 128, 128))
    cases = build_workload_cases(
        workload,
        (3,),
        pointwise_shapes=(),
        depthwise_shapes=(("regular", 128, 3, (32, 32, 32)),),
        sm=(12, 0),
    )
    dw = next(case for case in cases if case.key.phase == "backward_weight")
    assert dict(dw.key.parameters) == {"dw_splits": 6, "dw_block": 1024}


def test_grn_cases_have_one_training_candidate_without_fake_kernel_or_parameters():
    workload = Workload("mednext_v2", "base", (32, 32, 32))
    cases = build_workload_cases(
        workload,
        (1, 2),
        pointwise_shapes=(),
        depthwise_shapes=(),
        grn_shapes=((96, (32, 32, 32)), (96, (32, 32, 32))),
    )
    assert len(cases) == 2
    assert {case.key.batch for case in cases} == {1, 2}
    assert all(case.key.family == "global_response_norm3d" for case in cases)
    assert all(case.key.phase == "training" for case in cases)
    assert all(case.key.kernel_size is None and not case.key.parameters for case in cases)
    assert all(case.key.implementation == "triton_fused_grn" for case in cases)
    assert {group.category for group in group_cases(cases)} == {"grn"}


def test_nullable_kernel_size_is_exclusive_to_grn():
    from dataclasses import replace

    import pytest

    from mednext_accel.profiling.matrix import KernelCaseKey

    grn = KernelCaseKey(
        "global_response_norm3d",
        "regular",
        "training",
        1,
        (3, 4, 5),
        8,
        8,
        None,
        "bfloat16",
        "triton_fused_grn",
        (),
    )
    for changes in (
        {"family": "pointwise_conv3d"},
        {"kernel_size": 1},
        {"phase": "backward_input"},
        {"parameters": (("block", 128),)},
    ):
        with pytest.raises(ValueError):
            replace(grn, **changes)
    for kernel in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="kernel_size"):
            replace(grn, family="pointwise_conv3d", kernel_size=kernel)
