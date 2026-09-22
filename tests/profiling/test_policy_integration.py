"""Provisional profiler dispatch must use the same v2 factory as runtime."""

import json

from mednext_accel import mednext_small
from mednext_accel.optimization.policy_io import load_policy
from mednext_accel.optimization.policy_resolver import PolicyResolver
from mednext_accel.profiling.api import write_profile_atomic
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
    )
    policy = synthesize_profile([item], name="provisional", sm=(12, 0), objective="balanced")
    path = write_profile_atomic(tmp_path / "provisional.json", policy)
    document = json.loads(path.read_text())
    assert document.get("version") == 2
    assert "defaults" not in document and "measurements" not in document
    assert load_policy(path) == policy
    model = mednext_small(in_channels=1, out_channels=3, optimization=document)
    assert isinstance(model.__dict__["_optimization_resolver"], PolicyResolver)
