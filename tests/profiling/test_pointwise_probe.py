import weakref

import pytest
import torch

from mednext_accel.profiling import runner


def test_pointwise_probe_uses_autocast_with_fp32_parameters_and_releases_validation(monkeypatch):
    # Keep the real convolution, GEMM, autograd and validation. Only redirect the
    # CUDA boundary so dtype and tensor-lifetime regressions run in CPU CI.
    randn, autocast, grad = torch.randn, torch.autocast, torch.autograd.grad
    monkeypatch.setattr(
        torch, "randn", lambda *args, **kwargs: randn(*args, **{**kwargs, "device": "cpu"})
    )
    monkeypatch.setattr(torch, "autocast", lambda device, **kwargs: autocast("cpu", **kwargs))
    comparison_tensors = []

    def checked_grad(output, inputs, upstream):
        assert [item.dtype for item in inputs] == [torch.bfloat16, torch.float32, torch.float32]
        assert output.dtype == upstream.dtype == torch.bfloat16
        values = grad(output, inputs, upstream)
        comparison_tensors.extend(weakref.ref(item) for item in (output, *values))
        return values

    def timed(function):
        assert all(reference() is None for reference in comparison_tensors)
        function()
        return 1.0, 64

    monkeypatch.setattr(torch.autograd, "grad", checked_grad)
    monkeypatch.setattr(runner, "_timed", timed)
    torch.manual_seed(13)
    result = runner._pointwise_probe(
        {"batch": 2, "in_channels": 8, "out_channels": 16, "spatial_shape": (4, 4, 4)}
    )

    assert result["valid"]
    assert set(result["validation_metrics"]) == {"output", "dX", "dW", "dB"}
    assert result["validator"] == "component-relative-l2-v1"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", [1, 2, 3])
def test_pointwise_probe_bf16_cuda_forward_and_backward(batch):
    torch.manual_seed(41)
    result = runner._pointwise_probe(
        {"batch": batch, "in_channels": 32, "out_channels": 64, "spatial_shape": (32, 32, 32)}
    )
    assert result["valid"], result
    assert result["rejection_reason"] is None
    assert set(result["validation_metrics"]) == {"output", "dX", "dW", "dB"}
