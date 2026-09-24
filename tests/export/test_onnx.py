from __future__ import annotations

import numpy as np
import pytest
import torch

from mednext_accel import mednext_small
from mednext_accel.export import export_onnx

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


@pytest.mark.onnx
def test_onnx_checker_and_runtime_match_eager() -> None:
    torch.manual_seed(3)
    model = mednext_small(in_channels=1, out_channels=3, base_channels=2).eval()
    example = torch.randn(1, 1, 32, 32, 32)

    program = export_onnx(model, example)
    onnx.checker.check_model(program.model_proto)
    session = ort.InferenceSession(
        program.model_proto.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(None, {session.get_inputs()[0].name: example.numpy()})[0]

    with torch.no_grad():
        expected = model(example).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    assert all("mednext_accel" not in node.domain for node in program.model_proto.graph.node)


@pytest.mark.onnx
def test_v2_onnx_checker_and_runtime_match_eager_without_grn_backend(
    no_v2_grn_backend, reduced_v2_export_model, v2_export_example
) -> None:
    program = export_onnx(reduced_v2_export_model, v2_export_example)
    onnx.checker.check_model(program.model_proto)
    session = ort.InferenceSession(
        program.model_proto.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(None, {session.get_inputs()[0].name: v2_export_example.numpy()})[0]

    with torch.no_grad():
        expected = reduced_v2_export_model(v2_export_example).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
