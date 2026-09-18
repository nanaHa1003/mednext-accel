# MedNeXt-Accel

MedNeXt-Accel is a production-oriented implementation of MedNeXt model
architectures for PyTorch. It provides a pure-PyTorch reference model, lossless
checkpoint import, selective activation checkpointing, portable evaluation
export, and optional per-shape CUDA acceleration.

The package contains architecture code only. It has no trainer, dataset
pipeline, loss framework, preprocessing stack, or nnU-Net dependency.

## Install

Choose the PyTorch build for the target driver and GPU first. This project does
not pin a CUDA build or an exact PyTorch release.

```bash
python -m pip install torch
python -m pip install .
```

Install optional features as needed:

```bash
python -m pip install '.[accelerated]'  # Triton training kernels
python -m pip install '.[export]'       # ONNX validation/runtime
python -m pip install '.[monai]'        # compatibility tests; not needed to load weights
```

## Build a model

```python
from mednext_accel import CheckpointConfig, mednext_base

model = mednext_base(
    in_channels=1,
    out_channels=3,
    spatial_dims=3,
    kernel_size=3,
    deep_supervision=True,
    checkpointing=CheckpointConfig(stages=(0, 1)),
)
```

Factories are available for Small, Base, Medium, and Large. Stage zero is full
resolution and stage four is the bottleneck. `CheckpointConfig(stages=None)`
checkpoints every expansion branch. Checkpointing runs only during gradient
enabled training and does not change state-dict keys.

With deep supervision enabled, training returns a tuple by default. `list` and
upsampled `stacked` formats are optional. Evaluation always returns one primary
logits tensor.

## Load existing weights

```python
from mednext_accel.checkpoints import load_checkpoint

report = load_checkpoint(model, "model_final_checkpoint.model", source="auto")
```

Official MedNeXt v1 checkpoints are converted losslessly. MONAI Small is
structurally compatible after key conversion. MONAI Base, Medium, and Large
require the explicit factories in `mednext_accel.compat.monai` because their
downsampling expansion widths differ from the official architecture. See
[checkpoint compatibility](docs/checkpoints.md).

## Select acceleration

Optimization is explicit and runs after device transfer, before compilation,
optimizer construction, or DDP wrapping:

```python
import torch
from mednext_accel.optimization import optimize

model = model.cuda()
report = optimize(
    model,
    input_shape=(1, 1, 128, 128, 128),
    dtype=torch.bfloat16,
    policy="autotune",
    compile_mode="default",
)
model.compile(mode=report.compile_mode, fullgraph=True)
```

The `torch` policy uses only native operators. `conservative` applies the narrow
RTX 50-series BF16 configurations validated in this repository. `autotune`
benchmarks every unique eligible shape on the current GPU and caches the result.
Every unselected shape uses PyTorch. Optional backends preserve parameter
identity and state-dict paths. See [optimization and compilation](docs/optimization.md).

## Export evaluation models

Evaluation supports direct `torch.jit.trace` with save/load, strict
`torch.export`, and the dynamo-based ONNX exporter. Optimized models use standard
PyTorch convolution paths in evaluation, so portable graphs contain no
`mednext_accel::*` custom operators. See [evaluation export](docs/export.md).

## Reproduce measurements

Kernel benchmarks are under [`benchmarks/`](benchmarks/). The training-step
profiler under [`tools/`](tools/) records the environment, policy, compile mode,
timings, peak allocated/reserved memory, operator table, and Chrome trace.
Generated output belongs in `artifacts/` and is ignored by Git.

Published RTX 5090 measurements and their limits are retained under
[`docs/benchmarks/`](docs/benchmarks/). Results are hardware and software
specific; use the autotuner or collect a matched profile on L40S, RTX A6000,
RTX 8000, and other architectures.

CPU correctness, export, and wheel-install tests run in CI on Python 3.10–3.12.
CUDA kernel correctness and performance tests require a local NVIDIA GPU; run
the complete suite with `python -m pytest -q` after installing the
`accelerated`, `export`, and `monai` extras.

## Attribution

The architecture follows the official
[MedNeXt v1 implementation](https://github.com/MIC-DKFZ/MedNeXt) and the paper
*MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation*
(Roy et al.). MONAI compatibility refers to
[MONAI's MedNeXt implementation](https://docs.monai.io/en/stable/networks.html#mednext).
See [NOTICE](NOTICE) and [LICENSE](LICENSE) for project attribution and terms.
