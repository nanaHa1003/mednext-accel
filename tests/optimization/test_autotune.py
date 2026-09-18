from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from mednext_accel.ops.pointwise import replace_pointwise_convs
from mednext_accel.optimization.autotune import autotune_selections


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_autotune_benchmarks_each_unique_shape_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mednext_accel.optimization.autotune as tuning

    calls: list[tuple[int, int, int, tuple[int, ...]]] = []

    def fake_benchmark(
        batch_size: int,
        in_channels: int,
        out_channels: int,
        spatial: tuple[int, ...],
        dtype: torch.dtype,
        warmup: int,
        repetitions: int,
    ) -> tuple[float, float]:
        calls.append((batch_size, in_channels, out_channels, spatial))
        return 2.0, 1.0

    monkeypatch.setattr(tuning, "_benchmark_pointwise", fake_benchmark)
    model = nn.Sequential(nn.Conv3d(2, 2, 1), nn.Conv3d(2, 2, 1)).cuda()

    result = autotune_selections(
        model,
        input_shape=(3, 2, 8, 8, 8),
        dtype=torch.bfloat16,
        warmup=1,
        repetitions=1,
        cache_path=None,
    )

    assert calls == [(3, 2, 2, (8, 8, 8))]
    assert result.selections.pointwise_gemm == ((2, 2, 8, 8, 8),)
    assert len(result.measurements) == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cache_hit_and_corruption_fallback(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mednext_accel.optimization.autotune as tuning

    calls = 0

    def fake_benchmark(*args, **kwargs) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return 2.0, 1.0

    monkeypatch.setattr(tuning, "_benchmark_pointwise", fake_benchmark)
    model = nn.Sequential(nn.Conv3d(2, 2, 1)).cuda()
    path = tmp_path / "nested" / "cache.json"
    kwargs = {
        "input_shape": (1, 2, 8, 8, 8),
        "dtype": torch.bfloat16,
        "warmup": 1,
        "repetitions": 1,
        "cache_path": path,
    }

    first = autotune_selections(model, **kwargs)
    second = autotune_selections(model, **kwargs)
    assert calls == 1
    assert not first.cache_hit
    assert second.cache_hit

    path.write_text("not json")
    third = autotune_selections(model, **kwargs)
    assert calls == 2
    assert not third.cache_hit
    assert json.loads(path.read_text())["version"] == 2


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_autotune_can_retune_an_existing_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mednext_accel.optimization.autotune as tuning

    calls = 0

    def fake_benchmark(*args, **kwargs) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return 2.0, 1.0

    monkeypatch.setattr(tuning, "_benchmark_pointwise", fake_benchmark)
    model = nn.Sequential(nn.Conv3d(2, 2, 1)).cuda()
    replace_pointwise_convs(model, selected_shapes=())

    result = autotune_selections(
        model,
        input_shape=(1, 2, 8, 8, 8),
        dtype=torch.bfloat16,
        warmup=1,
        repetitions=1,
        cache_path=None,
    )

    assert calls == 1
    assert result.selections.pointwise_gemm == ((2, 2, 8, 8, 8),)
