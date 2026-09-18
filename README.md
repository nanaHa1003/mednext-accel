# MedNeXt-Accel

[![CI](https://github.com/nanaHa1003/mednext-accel/actions/workflows/ci.yml/badge.svg)](https://github.com/nanaHa1003/mednext-accel/actions/workflows/ci.yml)

MedNeXt-Accel is a production-oriented implementation of MedNeXt model
architectures for PyTorch. It provides a pure-PyTorch reference model, lossless
checkpoint import, selective activation checkpointing, portable evaluation
export, and optional per-shape CUDA acceleration.

The package contains architecture code only. It has no trainer, dataset
pipeline, loss framework, preprocessing stack, or nnU-Net dependency.

## Why MedNeXt-Accel?

The [official MedNeXt repository](https://github.com/MIC-DKFZ/MedNeXt) is the
authoritative research release, and [MONAI](https://github.com/Project-MONAI/MONAI)
provides a maintained implementation inside a broad medical-imaging framework.
MedNeXt-Accel focuses on a smaller target: a standalone model package with
checkpoint interoperability, measured training-memory controls, and optional
shape-specific CUDA acceleration.

| Capability | Official MedNeXt v1 | MONAI MedNeXt | MedNeXt-Accel |
|---|---|---|---|
| Published Small/Base/Medium/Large, 2D and 3D | Yes | Yes | Yes |
| Standalone model without nnU-Net | No; distributed in an nnU-Net v1 fork | Yes, as part of MONAI | **Yes; architecture-only package** |
| Load official v1 checkpoints | Native format | No conversion API; Base/Medium/Large also differ in down-block expansion ratios | **Automatic, tensor-preserving conversion** |
| Load MONAI checkpoints | No conversion API | Native format | **Automatic; explicit MONAI-compatible factories for Base/Medium/Large** |
| Activation checkpointing policy | Whole-block checkpointing in published Medium/Large factories | No model-level policy | **Expansion branch only, selectable per resolution stage** |
| Model-specific optimized training operators | No | No | **Triton depthwise backward and optional pointwise GEMM** |
| Hardware/shape-aware backend selection | No | No | **Conservative policy or persistent per-shape autotuning** |
| Explicit portable-eval validation | No model export API | No model export API | **`jit.trace`, `torch.export`, and ONNX Runtime tested** |

“No” means that the upstream model API does not provide that facility; it does
not imply that an external wrapper cannot add it. Optional optimized operators
keep the same parameters and state-dict paths as the PyTorch implementation.

### RTX 5090 training results

The table below isolates the execution-policy changes using the same
official-compatible MedNeXt Base architecture. Measurements use an RTX 5090,
PyTorch 2.12.0+cu132, CUDA 13.2, cuDNN 9.20, BF16 autocast, batch size one,
`128x128x128` input, three classes, deep supervision, AdamW, and
`torch.compile`. Each value is the median of three independent 50-step runs
after 20 warmup steps. Memory is peak PyTorch allocation.

| Execution policy | Checkpointed stages | Step time | Change vs. native | Peak allocated | Change vs. native |
|---|---:|---:|---:|---:|---:|
| Native PyTorch convolutions | None | 73.39 ms | baseline | 8,426 MiB | baseline |
| Custom operators | None | **59.46 ms** | **19.0% less time / 1.23x throughput** | 8,375 MiB | 0.6% lower |
| Custom operators + selective checkpointing | `(0,)` | **65.93 ms** | **10.2% less time** | **5,946 MiB** | **29.4% lower** |
| Custom operators + selective checkpointing | `(0, 1)` | **68.62 ms** | **6.5% less time** | **4,510 MiB** | **46.5% lower** |
| Custom operators + selective checkpointing | All expansion stages | **69.20 ms** | **5.7% less time** | **3,914 MiB** | **53.6% lower** |

These results compare optimized and native execution inside this package, rather
than timing the full nnU-Net or MONAI frameworks. The native row uses the
official-compatible architecture and ordinary PyTorch convolutions; the custom
rows change execution without changing learned parameters. Results are specific
to this GPU and software stack. The complete 3/8-class results, raw run medians,
methodology, and limitations are in the
[benchmark notes](docs/benchmarks/activation-memory.md).

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

Conversion only remaps parameter names: tensors are neither resized nor
approximated. The converted model therefore reuses existing learned weights
without retraining. Strict loading and exact-output regression tests cover the
supported mappings, including eager and `torch.compile` wrapper checkpoints.

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
