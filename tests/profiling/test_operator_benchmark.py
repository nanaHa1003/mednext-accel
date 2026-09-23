"""Integrated dispatch evidence keeps campaign execution settings and raw diagnostics."""

from dataclasses import replace

import pytest
import torch

from mednext_accel.profiling import runner
from mednext_accel.profiling.grouped import case_payload, run_group_with_bisection
from mednext_accel.profiling.matrix import KernelCase, KernelCaseKey, KernelGroup
from mednext_accel.profiling.synthesize import Measurement, measurement_from_result


def make_case(direction="regular", phase="backward_input", *, pointwise=False):
    implementation = {
        ("regular", "backward_input"): "triton_depthwise_dx",
        ("regular", "backward_weight"): "triton_split_dw",
        ("downsample", "backward_input"): "triton_downsample_dx",
        ("transpose", "backward_weight"): "triton_transpose_split_dw",
    }
    return KernelCase(
        KernelCaseKey(
            family=(
                "pointwise_conv3d"
                if pointwise
                else "depthwise_conv_transpose3d"
                if direction == "transpose"
                else "depthwise_conv3d"
            ),
            direction=direction,
            phase="training" if pointwise else phase,
            batch=1,
            spatial_shape=(5, 5, 5),
            in_channels=2,
            out_channels=4 if pointwise else 2,
            kernel_size=1 if pointwise else 3,
            dtype="bfloat16",
            implementation="pointwise_gemm_per_sample"
            if pointwise
            else implementation[direction, phase],
            parameters=()
            if pointwise
            else (("dx_block", 128),)
            if phase == "backward_input"
            else (("dw_splits", 1), ("dw_block", 256)),
        )
    )


def test_integrated_payload_preserves_identity_seed_and_exact_gradient_mask():
    case = make_case()
    payload = case_payload(
        case, seed=31, compile_mode="reduce-overhead", gradient_mask=(False, True, False)
    )
    alternative = case_payload(
        case, seed=31, compile_mode="eager", gradient_mask=(True, False, True)
    )
    assert payload["benchmark_kind"] == "integrated_operator"
    assert payload["compile_mode"] == "reduce-overhead"
    assert payload["gradient_mask"] == (False, True, False)
    assert payload["case_id"] == alternative["case_id"] == case.identifier
    assert payload["seed"] == alternative["seed"]


def test_bisected_children_retain_integrated_execution_settings():
    case = make_case()
    cases = (case, KernelCase(replace(case.key, batch=2)))
    payloads = []

    def invoke(payload):
        payloads.extend(payload["cases"])
        if len(payload["cases"]) > 1:
            return {"status": "timeout"}
        return {"status": "ok", "results": [{"case_id": payload["cases"][0]["case_id"]}]}

    run_group_with_bisection(
        KernelGroup("regular-dx", 1, cases), invoke, compile_mode="max-autotune"
    )
    assert len(payloads) == 4
    assert all(p["compile_mode"] == "max-autotune" for p in payloads)
    assert all(p["gradient_mask"] == (True, True, True) for p in payloads)
    assert all(p["benchmark_kind"] == "integrated_operator" for p in payloads)


def test_dispatch_evidence_retains_raw_diagnostics_without_using_their_timings():
    raw = {"benchmark_kind": "raw_kernel_diagnostic", "status": "ok", "candidate_ms": 0.1}
    result = {
        "status": "ok",
        "valid": True,
        "reference_ms": 2.0,
        "candidate_ms": 3.0,
        "reference_peak_bytes": 64,
        "candidate_peak_bytes": 64,
        "benchmark_kind": "integrated_operator",
        "compile_mode": "default",
        "gradient_mask": [False, True, True],
        "raw_diagnostic": raw,
        "memory_measured": False,
    }
    evidence = measurement_from_result(make_case(), result, checkpointing="none")
    restored = Measurement(**evidence.to_primitive())
    assert restored.benchmark_kind == "integrated_operator"
    assert restored.compile_mode == "default"
    assert restored.gradient_mask == (False, True, True)
    assert restored.raw_diagnostic == raw
    assert restored.memory_measured is False
    assert restored.candidate_ms == 3.0
    assert restored.objective_winner is False
    with pytest.raises(TypeError):
        restored.raw_diagnostic["candidate_ms"] = 99


@pytest.mark.filterwarnings(
    "error:torch.compile with fullgraph=True found no compiled frames:UserWarning"
)
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "compile_mode,memory_measured",
    [
        ("eager", True),
        ("default", True),
        ("reduce-overhead", False),
        ("max-autotune", False),
        ("max-autotune-no-cudagraphs", True),
    ],
)
@pytest.mark.parametrize("mask", [(True, True, True), (False, True, False)])
@pytest.mark.parametrize(
    "case",
    [
        make_case(),
        make_case(phase="backward_weight"),
        make_case("downsample"),
        make_case("transpose", "backward_weight"),
        make_case(pointwise=True),
    ],
)
def test_integrated_benchmark_validates_and_times_actual_adaptive_modules(
    case, compile_mode, mask, memory_measured
):
    result = runner._operator_probe(
        case_payload(case, seed=13, compile_mode=compile_mode, gradient_mask=mask)
    )
    assert result["status"] == "ok", result
    assert result["valid"] is True, result
    assert result["benchmark_kind"] == "integrated_operator"
    assert result["compile_mode"] == compile_mode
    assert result["memory_measured"] is memory_measured
    evidence = measurement_from_result(case, result, checkpointing="none")
    assert evidence.memory_measured is memory_measured
    assert result["gradient_mask"] == mask
    assert set(result["validation_metrics"]) == (
        {"output", "dX", "dW", "dB"} if mask[0] else {"output", "dW"}
    )
    assert result["reference_ms"] > 0 and result["candidate_ms"] > 0
    assert result["reference_peak_bytes"] > 0 and result["candidate_peak_bytes"] > 0


def test_failed_integrated_setup_retains_execution_settings():
    case = KernelCase(replace(make_case().key, dtype="float32"))
    result = runner._operator_probe(
        case_payload(case, compile_mode="reduce-overhead", gradient_mask=(False, True, True))
    )
    assert result["status"] == "error"
    assert result["benchmark_kind"] == "integrated_operator"
    assert result["compile_mode"] == "reduce-overhead"
    assert result["gradient_mask"] == (False, True, True)
    assert "require bfloat16" in result["message"]


def test_subprocess_failure_retains_integrated_settings_for_gap_evidence():
    case = make_case()
    result = run_group_with_bisection(
        KernelGroup("regular-dx", 1, (case,)),
        lambda _: {"status": "timeout"},
        compile_mode="max-autotune",
    )[case.identifier]
    evidence = measurement_from_result(case, result, checkpointing="none")
    assert evidence.benchmark_kind == "integrated_operator"
    assert evidence.compile_mode == "max-autotune"
    assert evidence.gradient_mask == (True, True, True)
    assert evidence.kernel_valid is None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_subprocess_retains_raw_diagnostic_beside_integrated_result():
    case = make_case("downsample")
    result = runner._invoke({"kind": "kernel_group", "cases": [case_payload(case)]}, timeout=120)
    assert result["status"] == "ok", result
    measured = result["results"][0]
    assert measured["status"] == "ok", measured
    evidence = measurement_from_result(case, measured, checkpointing="none")
    assert evidence.benchmark_kind == "integrated_operator"
    assert evidence.compile_mode == "default"
    assert set(evidence.validation_metrics) == {"output", "dX", "dW", "dB"}
    assert evidence.raw_diagnostic["benchmark_kind"] == "raw_kernel_diagnostic"
    assert evidence.raw_diagnostic["status"] == "ok"
    assert set(evidence.raw_diagnostic["validation_metrics"]) == {"dX"}
    assert evidence.raw_diagnostic["candidate_ms"] > 0
    assert evidence.candidate_ms > 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_operator_callable_streams_detached_components_with_the_requested_mask(monkeypatch):
    import weakref

    from mednext_accel.profiling.operator_benchmark import benchmark_operator

    prior = []
    components = []
    arities = []
    compare, grad = runner.validate_components, torch.autograd.grad

    def checked_compare(actual, expected):
        assert all(item() is None for item in prior), "previous component pair remains live"
        assert len(actual) == len(expected) == 1
        assert all(item.grad_fn is None for item in (*actual.values(), *expected.values()))
        prior.extend(weakref.ref(item) for item in (*actual.values(), *expected.values()))
        components.extend(actual)
        return compare(actual, expected)

    def checked_grad(output, targets, upstream):
        arities.append(len(targets))
        return grad(output, targets, upstream)

    monkeypatch.setattr(runner, "validate_components", checked_compare)
    monkeypatch.setattr(torch.autograd, "grad", checked_grad)
    result = benchmark_operator(
        make_case(phase="backward_weight"),
        compile_mode="eager",
        seed=17,
        gradient_mask=(False, True, True),
    )
    assert result["status"] == "ok"
    assert result["valid"] is True
    assert result["seed"] == 17
    assert components == ["output", "dW", "dB"]
    assert arities and set(arities) == {2}
    assert all(item() is None for item in prior)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_fallback_cannot_be_recorded_as_the_requested_custom_candidate():
    # A 1->1 convolution is classified as depthwise by the actual wrapper.
    # A pointwise case must not silently time that native fallback as GEMM.
    case = KernelCase(replace(make_case(pointwise=True).key, in_channels=1, out_channels=1))
    result = runner._operator_probe(case_payload(case, compile_mode="eager"))
    assert result["status"] == "error"
    assert "did not select" in result["message"]
    assert "candidate_ms" not in result


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compile_mode", ["eager", "default", "max-autotune-no-cudagraphs"])
def test_non_graph_timing_accounts_for_live_working_allocations(compile_mode):
    torch.compiler.reset()
    sample = torch.randn(1 << 18, device="cuda")

    def operator(value):
        return value.sin()

    forward = (
        operator
        if compile_mode == "eager"
        else torch.compile(operator, mode=compile_mode, fullgraph=True)
    )
    forward(sample)  # Complete compilation before establishing the live baseline.
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    _, peak = runner._timed(lambda: forward(sample))
    assert peak >= baseline + sample.numel() * sample.element_size()
