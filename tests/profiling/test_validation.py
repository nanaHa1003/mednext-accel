from mednext_accel.profiling.synthesize import Measurement
from mednext_accel.profiling.validation import (
    DecisionSegment,
    apply_validation_results,
    decision_signature,
    segment_decisions,
    validation_batches,
)


def _measurement(
    batch: int,
    *,
    implementation: str = "pointwise_gemm_per_sample",
    candidate_ms: float = 1.0,
    valid: bool = True,
) -> Measurement:
    return Measurement(
        family="pointwise_conv3d",
        direction="regular",
        phase="training",
        implementation=implementation,
        batch=batch,
        spatial_shape=(16, 16, 16),
        in_channels=8,
        out_channels=16,
        dtype="bfloat16",
        checkpointing="none",
        reference_ms=2.0,
        candidate_ms=candidate_ms,
        reference_peak_bytes=1_000,
        candidate_peak_bytes=900,
        valid=valid,
    )


def test_equal_consecutive_signatures_form_one_segment() -> None:
    gemm = (
        (
            "pointwise_conv3d",
            "regular",
            "training",
            (16, 16, 16),
            8,
            16,
            "pointwise_gemm_per_sample",
            (),
        ),
    )
    decisions = {1: gemm, 2: gemm, 3: gemm}
    segments = segment_decisions(decisions)
    assert segments == (DecisionSegment((1, 2, 3), gemm),)
    assert validation_batches(segments) == (1, 3)


def test_sample_gap_and_decision_change_split_segments() -> None:
    native = (("pointwise_conv3d", "regular", "training", (16, 16, 16), 8, 16, "reference", ()),)
    gemm = (
        (
            "pointwise_conv3d",
            "regular",
            "training",
            (16, 16, 16),
            8,
            16,
            "pointwise_gemm_per_sample",
            (),
        ),
    )
    decisions = {1: native, 2: gemm, 4: gemm}
    segments = segment_decisions(decisions)
    assert tuple(segment.batches for segment in segments) == ((1,), (2,), (4,))
    assert validation_batches(segments) == (1, 2, 4)


def test_decision_signature_selects_reference_or_candidate_and_sorts_entries() -> None:
    measurements = (
        _measurement(2, implementation="z_candidate"),
        _measurement(2, implementation="a_candidate", candidate_ms=3.0),
    )
    assert decision_signature(measurements, "throughput") == (
        (
            "pointwise_conv3d",
            "regular",
            "training",
            (16, 16, 16),
            8,
            16,
            "reference",
            (),
        ),
        (
            "pointwise_conv3d",
            "regular",
            "training",
            (16, 16, 16),
            8,
            16,
            "z_candidate",
            (),
        ),
    )


def test_failed_endpoint_invalidates_every_measurement_in_segment() -> None:
    measurements = tuple(_measurement(batch) for batch in (1, 2, 3))
    signature = decision_signature((measurements[0],), "throughput")
    segment = DecisionSegment((1, 2, 3), signature)
    updated = apply_validation_results(
        measurements,
        (segment,),
        accepted={1: True, 3: False},
    )
    assert {item.valid for item in updated} == {False}


def test_missing_endpoint_rejects_segment_and_unrelated_measurements_keep_validity() -> None:
    measurements = (_measurement(1), _measurement(2), _measurement(5))
    signature = decision_signature((measurements[0],), "throughput")
    segments = (DecisionSegment((1, 2), signature),)
    updated = apply_validation_results(measurements, segments, accepted={1: True})
    assert tuple(item.valid for item in updated) == (False, False, True)


def test_validation_batches_deduplicates_one_batch_segments_in_ascending_order() -> None:
    signature = (("family", "direction", "phase", (1, 1, 1), 1, 1, "reference", ()),)
    segments = (
        DecisionSegment((4,), signature),
        DecisionSegment((2, 3), signature),
        DecisionSegment((1,), signature),
    )
    assert validation_batches(segments) == (1, 2, 3, 4)
