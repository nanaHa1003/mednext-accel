from __future__ import annotations

import io

import pytest
import torch

from mednext_accel import mednext_small
from mednext_accel.export import trace


def test_direct_trace_save_load_matches_eager_without_custom_nodes() -> None:
    torch.manual_seed(1)
    model = mednext_small(in_channels=1, out_channels=3, base_channels=2).eval()
    example = torch.randn(1, 1, 32, 32, 32)

    traced = torch.jit.trace(model, example)
    buffer = io.BytesIO()
    torch.jit.save(traced, buffer)
    buffer.seek(0)
    restored = torch.jit.load(buffer)

    with torch.no_grad():
        torch.testing.assert_close(restored(example), model(example), rtol=0, atol=0)
    assert "mednext_accel::" not in str(restored.inlined_graph)


def test_trace_helper_requires_eval_mode() -> None:
    model = mednext_small(in_channels=1, out_channels=3, base_channels=2).train()

    with pytest.raises(ValueError, match="eval"):
        trace(model, torch.randn(1, 1, 32, 32, 32))


def test_trace_helper_returns_loadable_module() -> None:
    model = mednext_small(in_channels=1, out_channels=3, base_channels=2).eval()
    example = torch.randn(1, 1, 32, 32, 32)

    traced = trace(model, example)

    torch.testing.assert_close(traced(example), model(example), rtol=0, atol=0)


def test_v2_direct_trace_save_load_matches_eager_without_grn_backend(
    no_v2_grn_backend, reduced_v2_export_model, v2_export_example, tmp_path
) -> None:
    traced = torch.jit.trace(reduced_v2_export_model, v2_export_example)
    path = tmp_path / "mednext-v2.pt"
    torch.jit.save(traced, path)
    restored = torch.jit.load(path)

    with torch.no_grad():
        eager = reduced_v2_export_model(v2_export_example)
        traced_output = traced(v2_export_example)
        restored_output = restored(v2_export_example)
    torch.testing.assert_close(traced_output, eager, rtol=0, atol=0)
    torch.testing.assert_close(restored_output, eager, rtol=0, atol=0)
