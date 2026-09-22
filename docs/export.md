# Evaluation export

MedNeXt-Accel models use ordinary PyTorch operations during evaluation. Call
`model.eval()` before tracing or export. Deep supervision does not change the
evaluation signature: exported models always return one primary logits tensor.

## TorchScript trace

Direct tracing and the convenience helper are both supported:

```python
import torch
from mednext_accel import mednext_base
from mednext_accel.export import trace

model = mednext_base(in_channels=1, out_channels=3).eval()
example = torch.randn(1, 1, 128, 128, 128)
traced = trace(model, example)
torch.jit.save(traced, "mednext.pt")
```

`torch.jit.script` is not part of the supported interface. Tracing records the
control flow selected by evaluation mode, so do not switch the traced module
back to training mode.

## torch.export

The model can be passed directly to strict export:

```python
program = torch.export.export(model, (example,), strict=True)
```

The example fixes the spatial dimensions unless the caller supplies explicit
dynamic-shape constraints to `torch.export.export`.

## ONNX

Install the export dependencies with `pip install 'mednext-accel[export]'`, then
use the dynamo-based exporter:

```python
from mednext_accel.export import export_onnx

program = export_onnx(model, example)
program.save("mednext.onnx")
```

Spatial dimensions are static. Set `dynamic_batch=True` when one exported model
must accept multiple batch sizes. ONNX correctness is checked against eager
FP32 execution with `rtol=1e-4` and `atol=1e-5` because backend convolution
implementations need not be bit exact.

## Verification scope

Export regression tests use a Small model at 32³ with reduced base channels on
CPU: direct trace/save/load and strict `torch.export` match eager FP32 exactly;
ONNX checker and ONNX Runtime use the tolerances above. These checks exercise
the evaluation path of the default optimized factory and require no custom
MedNeXt operators in the exported graph. They do not establish training export,
`jit.script`, every dynamic-shape constraint, or equivalence across all ONNX
execution providers. The ONNX suite requires the optional export dependencies.
