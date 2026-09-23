"""Provisional profiler dispatch must use the same v2 factory as runtime."""

import yaml

from mednext_accel import mednext_small
from mednext_accel.optimization.policy import policy_to_primitive
from mednext_accel.optimization.policy_io import load_policy
from mednext_accel.optimization.policy_resolver import PolicyResolver
from mednext_accel.profiling.synthesize import Measurement, synthesize_profile


def test_generated_provisional_policy_roundtrips_and_is_factory_input(tmp_path):
    item = Measurement(
        "depthwise_conv3d",
        "regular",
        "backward_input",
        "triton_depthwise_dx",
        9,
        (32, 32, 32),
        128,
        128,
        "bfloat16",
        "all-expansion",
        10,
        5,
        100,
        100,
        True,
        (("dx_block", 128),),
        kernel_size=3,
        benchmark_kind="integrated_operator",
        gradient_mask=(True, True, True),
        memory_measured=True,
    )
    policy = synthesize_profile([item], name="provisional", sm=(12, 0), objective="balanced")
    path = tmp_path / "provisional.policy.yaml"
    path.write_text(yaml.safe_dump(policy_to_primitive(policy)))
    document = yaml.safe_load(path.read_text())
    assert document.get("version") == 2
    assert "defaults" not in document and "measurements" not in document
    assert load_policy(path) == policy
    model = mednext_small(in_channels=1, out_channels=3, optimization=document)
    assert isinstance(model.__dict__["_optimization_resolver"], PolicyResolver)
