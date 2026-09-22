import pytest

from mednext_accel.profiling.batch_search import BatchSearch, ProbeResult, search_batches


def result_for_boundary(boundary: int, calls: list[int]):
    def probe(batch: int) -> ProbeResult:
        calls.append(batch)
        return ProbeResult(batch, batch <= boundary, batch * 100)

    return probe


def test_feasible_maximum_is_the_only_probe() -> None:
    calls: list[int] = []
    result = search_batches(
        BatchSearch(maximum=16),
        total_vram_bytes=10_000,
        probe=result_for_boundary(16, calls),
    )
    assert calls == [16]
    assert result.probed_batches == (16,)
    assert result.maximum_feasible == 16


def test_infeasible_maximum_is_refined_by_binary_search() -> None:
    calls: list[int] = []
    result = search_batches(
        BatchSearch(maximum=16),
        total_vram_bytes=10_000,
        probe=result_for_boundary(13, calls),
    )
    assert calls == [16, 1, 8, 12, 14, 13]
    assert result.probed_batches == (1, 8, 12, 13, 14, 16)
    assert result.maximum_feasible == 13


def test_unbounded_search_uses_exponential_growth_then_binary_search() -> None:
    calls: list[int] = []
    result = search_batches(
        BatchSearch(),
        total_vram_bytes=10_000,
        probe=result_for_boundary(13, calls),
    )
    assert calls == [1, 2, 4, 8, 16, 12, 14, 13]
    assert result.probed_batches == (1, 2, 4, 8, 12, 13, 14, 16)
    assert result.maximum_feasible == 13


def test_batch_one_oom_is_a_complete_result() -> None:
    result = search_batches(
        BatchSearch(),
        total_vram_bytes=100,
        probe=lambda batch: ProbeResult(batch, False, 200, "oom"),
    )
    assert result.maximum_feasible == 0
    assert result.probed_batches == (1,)


def test_memory_headroom_can_reject_an_otherwise_successful_probe() -> None:
    result = search_batches(
        BatchSearch(maximum=3, memory_fraction=0.9),
        total_vram_bytes=1000,
        probe=lambda batch: ProbeResult(batch, True, batch * 400),
    )
    assert result.maximum_feasible == 2


@pytest.mark.parametrize(
    ("feasible", "expected_maximum"),
    [(True, 1), (False, 0)],
)
def test_maximum_one_requires_one_probe(feasible: bool, expected_maximum: int) -> None:
    calls: list[int] = []

    def probe(batch: int) -> ProbeResult:
        calls.append(batch)
        return ProbeResult(batch, feasible, 100)

    result = search_batches(BatchSearch(maximum=1), total_vram_bytes=1000, probe=probe)

    assert calls == [1]
    assert result.probed_batches == (1,)
    assert result.maximum_feasible == expected_maximum


def test_mismatched_probe_result_is_rejected() -> None:
    with pytest.raises(ValueError, match="different batch"):
        search_batches(
            BatchSearch(maximum=2),
            total_vram_bytes=1000,
            probe=lambda batch: ProbeResult(batch + 1, True, 100),
        )
