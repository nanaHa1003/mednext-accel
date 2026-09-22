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
