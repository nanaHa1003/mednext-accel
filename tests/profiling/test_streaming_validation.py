import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mednext_accel.profiling import runner, validation


@pytest.fixture
def cpu_probe(monkeypatch):
    # CUDA allocation/timing are the external boundaries. Keep actual autocast,
    # convolution, GEMM, autograd, and validation calculations.
    randn, autocast = torch.randn, torch.autocast
    monkeypatch.setattr(
        torch, "randn", lambda *args, **kwargs: randn(*args, **{**kwargs, "device": "cpu"})
    )
    monkeypatch.setattr(torch, "autocast", lambda device, **kwargs: autocast("cpu", **kwargs))
    monkeypatch.setattr(runner, "_timed", lambda function, **kwargs: (1.0, 64))
    return {"batch": 2, "in_channels": 8, "out_channels": 16, "spatial_shape": (4, 4, 4)}


def test_pointwise_components_release_pairs_and_graphs_before_next_component(
    cpu_probe, monkeypatch
):
    prior = []
    order = []
    compare = runner.validate_components
    grad = torch.autograd.grad
    gradient_arities = []

    def record_grad(output, inputs, gradient):
        gradient_arities.append(len(inputs))
        return grad(output, inputs, gradient)

    def checked_compare(actual, expected):
        assert len(actual) == len(expected) == 1, "retained multiple component pairs"
        assert all(item() is None for item in prior), "previous pair remains live"
        assert all(item.grad_fn is None for item in (*actual.values(), *expected.values()))
        order.extend(actual)
        prior.extend(weakref.ref(item) for item in (*actual.values(), *expected.values()))
        return compare(actual, expected)

    monkeypatch.setattr(torch.autograd, "grad", record_grad)
    monkeypatch.setattr(runner, "validate_components", checked_compare)
    result = runner._pointwise_probe(cpu_probe)
    assert result["status"] == "ok", result
    assert result["valid"]
    assert order == ["output", "dX", "dW", "dB"]
    assert gradient_arities == [1] * 6
    assert all(item() is None for item in prior)


class FloatingAllocations(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.largest = 0
        self.references = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        outputs = result if isinstance(result, (tuple, list)) else (result,)
        for item in outputs:
            if isinstance(item, torch.Tensor) and item.dtype in (torch.float32, torch.float64):
                self.largest = max(self.largest, item.numel())
                self.references.append(weakref.ref(item))
        return result


def test_comparison_error_scratch_is_bounded_for_large_activations(monkeypatch):
    # Scale the same 127^3 channel-expanded activation structure down for CPU CI.
    # Full FP32 casts/difference/abs would each exceed the 1024-element budget.
    monkeypatch.setattr(validation, "_CHUNK_ELEMENTS", 1024, raising=False)
    expected = torch.randn(2, 16, 31, 31, 31, dtype=torch.bfloat16)
    actual = expected * 1.0078125
    allocations = FloatingAllocations()
    with allocations:
        result = validation.validate_components({"output": actual}, {"output": expected})
    assert result["valid"]
    assert allocations.largest <= 1024
    assert all(item() is None for item in allocations.references)
    delta = actual.float() - expected.float()
    assert result["validation_metrics"]["output"]["relative_l2"] == pytest.approx(
        (delta.double().norm() / expected.double().norm()).item(), rel=1e-5
    )


@pytest.mark.parametrize(
    "component_index,component", tuple(enumerate(("output", "dX", "dW", "dB")))
)
@pytest.mark.parametrize("side", ["reference", "candidate", "metrics"])
@pytest.mark.parametrize("exception", [torch.cuda.OutOfMemoryError, RuntimeError])
def test_pointwise_failure_stage_retains_completed_components(
    cpu_probe, monkeypatch, component_index, component, side, exception
):
    if side == "reference":
        owner, name = torch.nn.functional, "conv3d"
        interval = 1
    elif side == "candidate":
        owner, name = torch, "addmm"
        interval = cpu_probe["batch"]
    else:
        owner, name = runner, "validate_components"
        interval = 1
    original = getattr(owner, name)
    calls = 0

    def failing(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == component_index * interval + 1:
            raise exception("injected component failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, failing)
    result = runner._pointwise_probe(cpu_probe)
    assert result["status"] == ("oom" if exception is torch.cuda.OutOfMemoryError else "error")
    assert result["failure_stage"] == f"validation.{component}.{side}"
    assert (
        list(result.get("validation_metrics", {})) == ["output", "dX", "dW", "dB"][:component_index]
    )
    assert "valid" not in result, "incomplete validation must not become a passing verdict"
    assert "injected component failure" in result["message"]


@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_timing_failure_keeps_complete_validation_and_prior_measurement(
    cpu_probe, monkeypatch, side
):
    count = 0

    def timed(function, **kwargs):
        nonlocal count
        count += 1
        if count == (1 if side == "reference" else 2):
            raise torch.cuda.OutOfMemoryError("injected timing failure")
        return 1.0, 64

    monkeypatch.setattr(runner, "_timed", timed)
    result = runner._pointwise_probe(cpu_probe)
    assert result["status"] == "oom"
    assert result["failure_stage"] == f"timing.{side}"
    assert result["valid"]
    assert len(result["validation_metrics"]) == 4
    if side == "candidate":
        assert result["reference_ms"] == 1.0 and result["reference_peak_bytes"] == 64


@pytest.mark.parametrize(
    "stage", ["warmup", "synchronize", "memory_reset", "measurement", "memory_read"]
)
def test_pointwise_timing_substage_is_preserved(cpu_probe, monkeypatch, stage):
    def fail(function, *, on_stage):
        on_stage(stage)
        raise RuntimeError("timing operation failed")

    monkeypatch.setattr(runner, "_timed", fail)
    result = runner._pointwise_probe(cpu_probe)
    assert result["failure_stage"] == f"timing.reference.{stage}"
    assert result["valid"] is True


def test_depthwise_validation_failure_retains_exact_phase(cpu_probe, monkeypatch):
    from types import SimpleNamespace

    from mednext_accel.ops import _triton

    def fail(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("depthwise candidate failed")

    monkeypatch.setattr(
        _triton, "depthwise", SimpleNamespace(depthwise_weight_grad=fail), raising=False
    )
    payload = {
        **cpu_probe,
        "kernel_size": 3,
        "direction": "regular",
        "phase": "backward_weight",
        "parameters": (("dw_splits", 1), ("dw_block", 128)),
    }
    result = runner._depthwise_probe(payload)
    assert result["status"] == "oom"
    assert result["failure_stage"] == "validation.dW.candidate"
    assert result["implementation"] == "triton_split_dw"
    assert "valid" not in result


@pytest.mark.parametrize("stage", ["initialization", "forward", "backward", "optimizer"])
def test_model_failure_stages_are_preserved(cpu_probe, monkeypatch, stage):
    class Tiny(torch.nn.Conv3d):
        def cuda(self):
            return self

        def compile(self, **kwargs):
            pass

        def forward(self, sample):
            if stage == "forward":
                raise RuntimeError("forward failed")
            return super().forward(sample)

    def factory(**kwargs):
        if stage == "initialization":
            raise RuntimeError("initialization failed")
        return Tiny(1, 2, 1)

    def timed(function, *, on_stage, **kwargs):
        on_stage("measurement")
        function()
        return 1.0, 64

    def fail(*args, **kwargs):
        raise RuntimeError(f"{stage} failed")

    monkeypatch.setattr(runner, "_factory", lambda variant: factory)
    monkeypatch.setattr(runner, "_timed", timed)
    if stage == "backward":
        monkeypatch.setattr(torch.Tensor, "backward", fail)
    if stage == "optimizer":
        monkeypatch.setattr(torch.optim.AdamW, "step", fail)
    result = runner._model_probe(
        {
            "batch": 1,
            "workload": {
                "variant": "small",
                "checkpointing": "none",
                "in_channels": 1,
                "out_channels": 2,
                "spatial": (4, 4, 4),
            },
        }
    )
    assert result["status"] == "error"
    expected = (
        "model.initialization" if stage == "initialization" else f"model.timing.measurement.{stage}"
    )
    assert result["failure_stage"] == expected


def test_child_failure_payload_survives_subprocess_ingestion(monkeypatch):
    import json
    import sys

    from mednext_accel.profiling.benchmark import run_json_subprocess

    raw = {
        "status": "oom",
        "seed": 91,
        "failure_stage": "model.timing.measurement.backward",
        "message": "allocation failed",
        "step_ms": 12.0,
        "diagnostics": {"loss": 0.5},
    }
    encoded = json.dumps(raw)
    monkeypatch.setattr(
        runner,
        "run_json_subprocess",
        lambda *args, **kwargs: run_json_subprocess(
            [sys.executable, "-c", f"print({encoded!r})"], timeout=5
        ),
    )
    assert runner._invoke({"kind": "model"}) == raw


_COMPONENTS = ("output", "dX", "dW", "dB")


@pytest.mark.parametrize(
    "invalid_component,failure_component",
    [
        (bad, later)
        for index, bad in enumerate(_COMPONENTS)
        for later in (*_COMPONENTS[index + 1 :], "timing")
    ],
)
@pytest.mark.parametrize("exception", [torch.cuda.OutOfMemoryError, RuntimeError])
def test_numerical_failure_survives_later_failure_and_subprocess_ingestion(
    cpu_probe, monkeypatch, invalid_component, failure_component, exception
):
    import json
    import sys

    from mednext_accel.profiling.benchmark import run_json_subprocess
    from mednext_accel.profiling.matrix import KernelCase, KernelCaseKey
    from mednext_accel.profiling.synthesize import measurement_from_result, synthesize_profile

    validate_pair = runner._validate_pair

    def failure():
        raise exception("failure after a known numerical rejection")

    def inject(probe, name, native, candidate):
        if name == invalid_component:
            original_candidate = candidate

            def candidate():
                return torch.zeros_like(original_candidate())

        if name == failure_component:
            native = failure
        return validate_pair(probe, name, native, candidate)

    monkeypatch.setattr(runner, "_validate_pair", inject)
    if failure_component == "timing":
        monkeypatch.setattr(runner, "_timed", lambda *args, **kwargs: failure())
    raw = runner._pointwise_probe(cpu_probe)
    encoded = json.dumps(raw)
    monkeypatch.setattr(
        runner,
        "run_json_subprocess",
        lambda *args, **kwargs: run_json_subprocess(
            [sys.executable, "-c", f"print({encoded!r})"], timeout=5
        ),
    )
    ingested = runner._invoke({"kind": "pointwise"})
    case = KernelCase(
        KernelCaseKey(
            "pointwise_conv3d",
            "regular",
            "training",
            cpu_probe["batch"],
            cpu_probe["spatial_shape"],
            cpu_probe["in_channels"],
            cpu_probe["out_channels"],
            1,
            "bfloat16",
            "pointwise_gemm_per_sample",
            (),
        )
    )
    measurement = measurement_from_result(case, ingested, checkpointing="none")
    assert measurement.kernel_valid is False
    assert measurement.probe_status == (
        "oom" if exception is torch.cuda.OutOfMemoryError else "error"
    )
    assert measurement.failure_stage == (
        "timing.reference"
        if failure_component == "timing"
        else f"validation.{failure_component}.reference"
    )
    assert measurement.validation_metrics[invalid_component]["relative_l2"] == 1.0
    assert measurement.rejection_reason.startswith(f"{invalid_component}:")
    assert len(measurement.validation_metrics) == (
        4 if failure_component == "timing" else _COMPONENTS.index(failure_component)
    )
    policy = synthesize_profile(
        [measurement], name="known-invalid", sm=(8, 9), objective="balanced"
    )
    assert measurement.benchmark_kind == "raw_kernel_diagnostic"
    assert policy.rules == (), "raw numerical diagnostics must not select runtime rules"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("exception", [torch.cuda.OutOfMemoryError, RuntimeError])
def test_cuda_probe_retains_numerical_rejection_before_later_failure(monkeypatch, exception):
    from mednext_accel.profiling.matrix import KernelCase, KernelCaseKey
    from mednext_accel.profiling.synthesize import measurement_from_result, synthesize_profile

    validate_pair = runner._validate_pair

    def inject(probe, name, native, candidate):
        if name == "output":
            original = candidate

            def candidate():
                return torch.zeros_like(original())
        elif name == "dX":

            def native():
                raise exception("later CUDA component failure")

        return validate_pair(probe, name, native, candidate)

    monkeypatch.setattr(runner, "_validate_pair", inject)
    result = runner._pointwise_probe(
        {
            "batch": 1,
            "in_channels": 1,
            "out_channels": 32,
            "spatial_shape": (8, 8, 8),
            "seed": 17,
        }
    )
    case = KernelCase(
        KernelCaseKey(
            "pointwise_conv3d",
            "regular",
            "training",
            1,
            (8, 8, 8),
            1,
            32,
            1,
            "bfloat16",
            "pointwise_gemm_per_sample",
            (),
        )
    )
    record = measurement_from_result(case, result, checkpointing="none")
    assert record.kernel_valid is False
    assert record.failure_stage == "validation.dX.reference"
    assert record.validation_metrics["output"]["relative_l2"] == 1.0
    policy = synthesize_profile([record], name="invalid", sm=(8, 9), objective="balanced")
    assert record.benchmark_kind == "raw_kernel_diagnostic"
    assert policy.rules == (), "raw numerical diagnostics must not select runtime rules"
