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
| Activation checkpointing policy | Whole-block checkpointing in published Medium/Large factories | No model-level policy | **Expansion branch or whole block, selectable per resolution stage** |
| Model-specific optimized training operators | No | No | **Triton depthwise backward and optional pointwise GEMM** |
| Hardware/shape-aware backend selection | No | No | **Conservative policy or persistent per-shape autotuning** |
| Explicit portable-eval validation | No model export API | No model export API | **`jit.trace`, `torch.export`, and ONNX Runtime tested** |

“No” means that the upstream model API does not provide that facility; it does
not imply that an external wrapper cannot add it. Optional optimized operators
keep the same parameters and state-dict paths as the PyTorch implementation.

### RTX 5090 training results

This is a direct model-level comparison against official MedNeXt commit
`0b78ed8` and MONAI 1.5.2. The workload is MedNeXt Base, an RTX 5090, PyTorch
2.12.0+cu132, CUDA 13.2, cuDNN 9.20, BF16 autocast, batch size one,
`128x128x128` input, three classes, five-head deep supervision, mean cross
entropy, and AdamW. Each value is the median of three independent 50-step runs
after 20 warmup steps. Compilation uses
`torch.compile(mode="default", fullgraph=True)`; memory is peak PyTorch
allocation.

The primary comparison below has no activation checkpointing. Eager results are
included because compilation is recommended but not required.

| Implementation and policy | Parameters | Eager step | Eager peak | Compiled step | Compiled peak |
|---|---:|---:|---:|---:|---:|
| Official MedNeXt v1 Base | 10,529,232¹ | 100.01 ms | 9,358 MiB | 73.98 ms | 8,339 MiB |
| MONAI MedNeXt Base | 10,513,775² | 99.78 ms | 9,318 MiB | 73.84 ms | 8,298 MiB |
| MedNeXt-Accel, PyTorch reference | 10,529,231 | 99.96 ms | 9,355 MiB | 74.06 ms | 8,339 MiB |
| **MedNeXt-Accel, optimized** | 10,529,231 | **85.66 ms** | 9,355 MiB | **59.46 ms** | 8,339 MiB |

Without compilation, the optimized policy takes **14.3% less step time than
official** and **14.2% less than MONAI**. Compilation reduces its step time by a
further 30.6%. The compiled optimized path takes **19.6% less step time than
official** and **19.5% less than MONAI**.

All checkpointing rows below use compilation so that they compare memory
policies under the recommended execution setting.

| Implementation and policy | Checkpointing | Step time | Peak allocated |
|---|---|---:|---:|
| Official MedNeXt v1 Base | None | 73.98 ms | 8,339 MiB |
| Official MedNeXt v1 Base | Whole block | 84.92 ms | 3,236 MiB |
| MONAI MedNeXt Base | None; no checkpoint API | 73.84 ms | 8,298 MiB |
| **MedNeXt-Accel, optimized** | None | **59.46 ms** | 8,339 MiB |
| **MedNeXt-Accel, optimized** | Expansion stage `(0,)` | **62.54 ms** | 5,795 MiB |
| **MedNeXt-Accel, optimized** | Expansion stages `(0, 1)` | **63.78 ms** | 4,358 MiB |
| **MedNeXt-Accel, optimized** | All expansion stages | **64.26 ms** | 3,762 MiB |
| **MedNeXt-Accel, optimized** | Whole block | 72.55 ms | **3,235 MiB** |

Checkpointing all expansion
stages remains 13.1% faster than uncheckpointed official MedNeXt while reducing
peak allocation by 54.9%. Official whole-block checkpointing reaches a lower
3,236 MiB, but costs 84.92 ms; the all-expansion policy is 24.3% faster at a
526 MiB memory cost. MedNeXt-Accel whole-block checkpointing reaches the same
memory point in 72.55 ms, 14.6% faster than the official policy. MONAI 1.5.2
exposes no activation-checkpointing option in its MedNeXt model API.

¹ The official model has one extra trainable dummy scalar used by its legacy
checkpoint implementation. ² MONAI Base is not structurally identical: its
first two down-block expansion ratios differ, accounting for the parameter-count
difference. The MedNeXt-Accel PyTorch-reference row matching official within
0.1% shows that the optimized result comes from its execution policy rather
than a smaller architecture. Results remain hardware and software specific.
Raw triplicate medians, reproduction instructions, and older 3/8-class policy
matrices are in the [benchmark notes](docs/benchmarks/implementation-comparison.md)
and [activation-memory study](docs/benchmarks/activation-memory.md).

## Install

Choose the PyTorch build for the target driver and GPU first. This project does
not pin a CUDA build or an exact PyTorch release.

```bash
python -m pip install torch
python -m pip install \
  'mednext-accel @ git+https://github.com/nanaHa1003/mednext-accel.git'
```

Install optional features from the same repository as needed:

```bash
python -m pip install \
  'mednext-accel[accelerated] @ git+https://github.com/nanaHa1003/mednext-accel.git'
python -m pip install \
  'mednext-accel[export] @ git+https://github.com/nanaHa1003/mednext-accel.git'
python -m pip install \
  'mednext-accel[monai] @ git+https://github.com/nanaHa1003/mednext-accel.git'
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

Use `CheckpointConfig(style="block")` to checkpoint every complete MedNeXt and
resampling block. `stages=(...)` can restrict either style to selected resolution
levels. Whole-block checkpointing saves more activation memory at a higher
recomputation cost.

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

Use `fullgraph=True` when the complete training graph is supported. On the RTX
5090, `mode="default"` is a good general setting, while
`mode="max-autotune-no-cudagraphs"` has worked well for long runs with a fixed
input shape and enough steps to amortize compilation.

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
