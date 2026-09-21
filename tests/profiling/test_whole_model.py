from mednext_accel.profiling.whole_model import WholeModelResult, validate_profile_candidate


def test_whole_model_validation_reports_memory_regression() -> None:
    result = validate_profile_candidate(
        lambda mode: WholeModelResult(
            mode=mode, step_ms=10 if mode == "reference" else 8,
            peak_bytes=100 if mode == "reference" else 140, valid=True,
        ),
        memory_tolerance=0.10,
    )
    assert result.requires_memory_override
