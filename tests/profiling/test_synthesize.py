from mednext_accel.profiling.synthesize import Measurement, candidate_wins, synthesize_profile


def measured(batch: int, reference_ms: float, candidate_ms: float) -> Measurement:
    return Measurement(
        family="pointwise_conv3d", direction="regular", phase="training",
        implementation="pointwise_gemm_per_sample", batch=batch,
        spatial_shape=(128, 128, 128), in_channels=32, out_channels=64,
        dtype="bfloat16", checkpointing="all-expansion",
        reference_ms=reference_ms, candidate_ms=candidate_ms,
        reference_peak_bytes=100, candidate_peak_bytes=100, valid=True,
    )


def test_adjacent_batches_with_same_winner_merge_into_interval() -> None:
    profile = synthesize_profile(
        [measured(2, 4.0, 3.0), measured(3, 6.0, 4.5), measured(4, 8.0, 6.2)],
        name="test", sm=(12, 0), objective="balanced",
    )
    assert len(profile.rules) == 1
    assert profile.rules[0].batch.minimum == 2
    assert profile.rules[0].batch.maximum == 4
    assert profile.rules[0].phases["training"].implementation == "pointwise_gemm_per_sample"


def test_invalid_candidate_and_memory_objective_choose_safely() -> None:
    invalid = measured(2, 4.0, 1.0)
    invalid = Measurement(**{**invalid.to_primitive(), "valid": False})
    memory = measured(3, 4.0, 4.1)
    memory = Measurement(**{
        **memory.to_primitive(), "reference_peak_bytes": 200,
        "candidate_peak_bytes": 100,
    })
    profile = synthesize_profile(
        [invalid, memory], name="test", sm=(12, 0), objective="memory"
    )
    assert profile.rules[0].batch.minimum == 3


def test_reported_winner_uses_the_same_balanced_threshold_as_synthesis() -> None:
    assert not candidate_wins(measured(2, 100.0, 99.0), "balanced")
    assert candidate_wins(measured(2, 100.0, 96.0), "balanced")
