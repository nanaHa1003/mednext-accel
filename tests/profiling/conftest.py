"""Integral GRN profiling cases and complete probe observations."""

import pytest

from mednext_accel.profiling.matrix import KernelCase, KernelCaseKey


@pytest.fixture
def grn_case():
    return KernelCase(
        KernelCaseKey(
            "global_response_norm3d",
            "regular",
            "training",
            1,
            (3, 4, 5),
            8,
            8,
            None,
            "bfloat16",
            "triton_fused_grn",
            (),
        )
    )


@pytest.fixture
def grn_result():
    return {
        "status": "ok",
        "valid": True,
        "reference_ms": 2.0,
        "candidate_ms": 1.0,
        "reference_peak_bytes": 200,
        "candidate_peak_bytes": 100,
        "benchmark_kind": "integrated_operator",
        "compile_mode": "eager",
        "gradient_mask": (True, True, True),
        "memory_measured": True,
        "validation_metrics": {
            name: {"relative_l2": 0.001, "max_absolute": 0.001, "finite": True}
            for name in ("output", "dX", "dGamma", "dBeta")
        },
        **{
            name: {"reference": 0.4, "candidate": 0.2}
            for name in ("forward_ms", "dx_ms", "dgamma_ms", "dbeta_ms")
        },
        "component_peak_bytes": {
            name: {"reference": 200, "candidate": 100}
            for name in ("forward", "dx", "dgamma", "dbeta")
        },
    }
