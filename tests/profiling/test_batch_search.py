from mednext_accel.profiling.batch_search import BatchSearch, ProbeResult, search_batches


def test_auto_search_is_dense_then_refines_to_vram_limit() -> None:
    def probe(batch: int) -> ProbeResult:
        return ProbeResult(batch=batch, feasible=batch <= 13, peak_bytes=batch * 100)

    result = search_batches(
        BatchSearch(strategy="auto", memory_fraction=0.90, dense_until=8),
        total_vram_bytes=1500,
        probe=probe,
    )
    assert set(range(1, 9)).issubset(result.probed_batches)
    assert result.maximum_feasible == 13
    assert 13 in result.probed_batches
    assert 14 in result.probed_batches


def test_batch_one_oom_is_a_complete_result() -> None:
    result = search_batches(
        BatchSearch(), total_vram_bytes=100,
        probe=lambda batch: ProbeResult(batch, False, 200, "oom"),
    )
    assert result.maximum_feasible == 0
    assert result.probed_batches == (1,)


def test_memory_headroom_can_reject_an_otherwise_successful_probe() -> None:
    result = search_batches(
        BatchSearch(maximum=3, memory_fraction=0.9), total_vram_bytes=1000,
        probe=lambda batch: ProbeResult(batch, True, batch * 400),
    )
    assert result.maximum_feasible == 2
