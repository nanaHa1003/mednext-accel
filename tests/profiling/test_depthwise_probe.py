from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mednext_accel.ops import _triton
from mednext_accel.profiling import runner
from mednext_accel.profiling.campaign import Workload
from mednext_accel.profiling.grouped import case_payload
from mednext_accel.profiling.matrix import build_workload_cases
from mednext_accel.profiling.synthesize import measurement_from_result


@pytest.mark.parametrize("supplied", [True, False])
@pytest.mark.parametrize(
    ("direction", "phase", "backend_name", "planned", "legacy"),
    [
        (
            "regular",
            "backward_input",
            "depthwise_input_grad",
            (("dx_block", 256),),
            {"block": 128},
        ),
        (
            "downsample",
            "backward_input",
            "depthwise_stride2_input_grad",
            (("dx_block", 256),),
            {"block": 128},
        ),
        (
            "regular",
            "backward_weight",
            "depthwise_weight_grad",
            (("dw_splits", 7), ("dw_block", 256)),
            {"splits": 1, "block": 512, "kernel_size": 3},
        ),
        (
            "transpose",
            "backward_weight",
            "depthwise_transpose_weight_grad",
            (("dw_splits", 7), ("dw_block", 256)),
            {"splits": 1, "block": 512, "kernel_size": 3},
        ),
    ],
)
def test_depthwise_probe_executes_planned_parameters_and_labels_evidence(
    monkeypatch, supplied, direction, phase, backend_name, planned, legacy
):
    # Real CPU tensor/reference operations; intercept only CUDA allocation/timing and Triton.
    randn = torch.randn
    monkeypatch.setattr(
        torch, "randn", lambda *args, **kwargs: randn(*args, **{**kwargs, "device": "cpu"})
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        runner, "_timed", lambda function, **kwargs: (1.0 if function() is not None else 0.0, 64)
    )
    calls = []

    def candidate(*args, **kwargs):
        calls.append(kwargs)
        shape = (1, 2, 4, 4, 4) if phase == "backward_input" else (2, 1, 3, 3, 3)
        return torch.zeros(shape, dtype=torch.bfloat16)

    monkeypatch.setattr(
        _triton, "depthwise", SimpleNamespace(**{backend_name: candidate}), raising=False
    )
    cases = build_workload_cases(
        Workload("mednext_v1", "base", (16, 16, 16)),
        (1,),
        pointwise_shapes=(),
        depthwise_shapes=((direction, 2, 3, (4, 4, 4)),),
    )
    case = next(item for item in cases if item.key.phase == phase)
    if supplied:
        case = replace(case, key=replace(case.key, parameters=planned))
    payload = case_payload(case)
    if supplied:
        result = runner._kernel_group_probe({"cases": [payload]})["results"][0]
    else:
        del payload["parameters"]
        result = runner._depthwise_probe(payload)

    expected = (
        (
            {"block": 256}
            if phase == "backward_input"
            else {"splits": 7, "block": 256, "kernel_size": 3}
        )
        if supplied
        else legacy
    )
    assert result["status"] == "ok"
    assert calls == [expected, expected]
    evidence = measurement_from_result(case, result, checkpointing="none")
    assert tuple(result["parameters"]) == evidence.parameters == case.key.parameters
    assert result["implementation"] == evidence.implementation == case.key.implementation
    # The existing all-zero stand-in is a 100% relative-L2 error. Preserve
    # the old rejection while retaining numerical diagnostics in the evidence.
    assert result["valid"] is False
    assert evidence.kernel_valid is False
    assert evidence.validator == "component-relative-l2-v1"
    component = "dX" if phase == "backward_input" else "dW"
    assert evidence.validation_metrics[component]["relative_l2"] == pytest.approx(1.0)
