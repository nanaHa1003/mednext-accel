from mednext_accel.optimization.descriptors import ExecutionContext
from mednext_accel.profiling.synthesize import synthesize_profile
from mednext_accel.profiling.whole_model import effective_policy_identity


def test_effective_policy_identity_hashes_ordered_runtime_layers():
    policy = synthesize_profile([], name="provisional", sm=(8, 6), objective="balanced")
    context = ExecutionContext(
        "training",
        "cuda",
        (8, 6),
        48 * 1024**3,
        "bfloat16",
        12,
        (128, 128, 128),
        "mednext_v1",
        "base",
        "none",
    )
    first = effective_policy_identity(policy, context)
    assert [item["name"] for item in first] == ["provisional", "sm86", "shared-nvidia"]
    assert all(len(item["sha256"]) == 64 for item in first)
    assert first == effective_policy_identity(policy, context)
