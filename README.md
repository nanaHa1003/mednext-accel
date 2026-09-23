# MedNeXt-Accel

[![CI](https://github.com/nanaHa1003/mednext-accel/actions/workflows/ci.yml/badge.svg)](https://github.com/nanaHa1003/mednext-accel/actions/workflows/ci.yml)

MedNeXt-Accel is a production-oriented implementation of MedNeXt model
architectures for PyTorch. It provides a pure-PyTorch reference model, lossless
checkpoint import, selective activation checkpointing, portable evaluation
export, and optional operator-based CUDA acceleration.

The package contains architecture code only. It has no trainer, dataset
pipeline, loss framework, preprocessing stack, or nnU-Net dependency.

## Why MedNeXt-Accel?

The [official MedNeXt repository](https://github.com/MIC-DKFZ/MedNeXt) is the
authoritative research release, and [MONAI](https://github.com/Project-MONAI/MONAI)
provides a maintained implementation inside a broad medical-imaging framework.
MedNeXt-Accel focuses on a smaller target: a standalone model package with
checkpoint interoperability, measured training-memory controls, and optional
shape-aware CUDA acceleration.

| Capability | Official MedNeXt v1 | MONAI MedNeXt | MedNeXt-Accel |
|---|---|---|---|
| Published Small/Base/Medium/Large, 2D and 3D | Yes | Yes | Yes |
| Standalone model without nnU-Net | No; distributed in an nnU-Net v1 fork | Yes, as part of MONAI | **Yes; architecture-only package** |
| Load official v1 checkpoints | Native format | No conversion API; Base/Medium/Large also differ in down-block expansion ratios | **Automatic, tensor-preserving conversion** |
| Load MONAI checkpoints | No conversion API | Native format | **Automatic; explicit MONAI-compatible factories for Base/Medium/Large** |
| Activation checkpointing policy | Whole-block checkpointing in published Medium/Large factories | No model-level policy | **Expansion branch or whole block, selectable per resolution stage** |
| Model-specific optimized training operators | No | No | **Triton depthwise backward and optional pointwise GEMM** |
| Hardware/shape-aware backend selection | No | No | **Shared NVIDIA and per-SM policies plus a standalone local profiler** |
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
checkpoints every expansion branch; `checkpointing=None` disables checkpointing.
For YAML configuration, construct `CheckpointConfig(**settings["checkpointing"])`
from the parsed mapping; `stages` accepts a list. Checkpointing runs only during
gradient-enabled training and does not change state-dict keys.

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

Factories use `optimization="auto"` by default. Construction loads compact
bundled policies and installs adaptive wrappers without benchmarking, writing
cache files, or changing parameters and state-dict keys:

```python
from mednext_accel import mednext_base

model = mednext_base(in_channels=1, out_channels=3)
```

Use `optimization="reference"` for standard PyTorch operators only. Supply a
local **policy YAML** from the profiler when you want measured overrides:

```python
model = mednext_base(
    in_channels=1,
    out_channels=3,
    optimization="~/.cache/mednext_accel/profiles/sm120-local.policy.yaml",
)
```

Resolution follows **external policy → exact-SM policy → shared NVIDIA policy →
reference**. A missing rule continues to the next layer; an explicit
`implementation: reference` stops that phase for a known counterexample.
Bundled policies cover SM86, SM89, and SM120. Generalized rules match operator
work and channel ranges, so an unmeasured batch/channel/spatial tuple can use an
accelerated path without an exact lookup entry. Unknown NVIDIA SMs receive the
shared policy. Selection uses the execution device, so moving a model between
GPUs selects the appropriate layer.
A user policy targeting another SM still applies with a warning. Correctness
guards remain active, and approximate implementations require explicit opt-in.

The validated generalized domain is CUDA BF16 training with contiguous 3D NCDHW
inputs, cubic k3 depthwise operators, and regular k1 pointwise operators.
Unsupported operators, 2D execution, and evaluation/export remain native.
A custom selection that fails an implementation guard falls through to lower
policy layers; an explicit reference tombstone stops resolution.

Checkpointing and batch size remain your choices. Bundled and newly generated
local rules match operators independently of model variant or checkpoint style;
evidence retains those campaign contexts. High-batch SM120 devices retain
structurally scaled depthwise kernels beyond the RTX 5090 measurements. More
VRAM alone does not establish
that every pointwise GEMM is faster. See the
[policy format and inference assumptions](docs/optimization.md).

Generate a local policy and its evidence with no workload flags:

```bash
mednext-accel profile
```

For a reproducible campaign, save the configuration as YAML and pass its path
to the same command. This example profiles MedNeXt Base at 128³ with one
checkpointing policy and an explicit batch upper bound:

```yaml
# mednext-base-128.yaml
preset: mednext-v1
objective: balanced
compile_mode: max-autotune-no-cudagraphs

batch_search:
  memory_fraction: 0.90
  maximum: 16

workloads:
  - variant: base
    spatial: [128, 128, 128]
    in_channels: 1
    out_channels: 3
    dtypes: [bfloat16]
    checkpointing: all-expansion
```

```bash
mednext-accel profile mednext-base-128.yaml
```

`variant` accepts `small`, `base`, `medium`, or `large`. `checkpointing`
accepts `none`, `all-expansion`, `whole-block`, or `auto`; `auto` expands the
workload into all three policies. `objective` accepts `balanced`, `throughput`,
or `memory`. Profiling currently supports BF16. A supplied positive `maximum`
is tested first; omit it to discover the upper bound with exponential growth
and binary refinement. `memory_fraction` determines how much of total GPU VRAM
a successful model probe may consume before it is treated as infeasible.

The profiler measures **training only**. It detects the GPU and VRAM, searches
for the largest feasible batch with isolated subprocesses, profiles integrated
adaptive operators only at that selected batch, validates their output and
every requested gradient, and compares the provisional policy's model time and
memory. A
supplied `batch_search.maximum` is tried first and completes the search in one
model probe when feasible. Without an upper bound, the profiler uses
exponential growth followed by binary refinement. Rerun with another
`maximum` when you want measured evidence for another batch. After batch search,
a representative two-phase display looks like this (the counts are
illustrative):

```text
kernel plan: 20 planned groups, 192 unique experiments, 576 requested references
Kernel groups 20/20 · experiments 192/192
validation plan: 2 whole-model validations
Validation 2/2
```

An experiment is one unique kernel case. Requested references include repeated
uses of that case across model and checkpoint contexts, so the kernel is timed
once and its result is reused. Groups are child-process attempts; a failed group
is bisected for isolation, which can increase the displayed total and the final
group count. Whole-model checks remain specific to each context and run once at
its selected batch. Batch search is also context-specific because checkpointing
and model shape affect memory, so its diagnostic probes are not globally
deduplicated or promoted into kernel measurements.

The profiler reports elapsed time and ETA for each fixed-size phase and prints
both output paths under `~/.cache/mednext_accel/profiles/` (or
`$XDG_CACHE_HOME/mednext_accel/profiles/`):

- `smXX-local.policy.yaml`: the stable runtime overlay; use this in `optimization=`.
- `smXX-local.<timestamp>-<hash>.evidence.json`: immutable environment, campaign,
  ordered batch probes, component errors, timings, memory, and model comparisons.

Dispatch evidence includes the wrapper, policy resolution, autograd, autocast,
gradient mask, and campaign compile mode. Raw kernel timings remain diagnostics.
Generated rules interpolate bounded operator regions, with exact reference
exceptions before broader rules; missing or failed probes leave gaps. CUDA
Graph replay does not expose complete working memory through allocator peaks,
so those operator measurements cannot qualify for memory or balanced objectives.

The policy references the exact evidence hash. Evidence is written first, then
the policy is atomically replaced. A rejected model comparison retains all
kernel evidence and publishes only exact reference overrides for numerical
failures or measured losers. Its remaining bundled fallthrough was not validated
by that comparison; the CLI says so. Model comparisons measure performance and
memory; whole-model numerical equivalence is explicitly `not-measured`.
Model construction and first forward never run these benchmarks.

Use `fullgraph=True` when the complete training graph is supported. On RTX 5090,
`mode="default"` is a good general choice, while
`mode="max-autotune-no-cudagraphs"` can be worthwhile for long fixed-shape runs
that amortize its larger compile cost.

Inspect decisions without running a full training step:

```python
report = model.explain_optimization(
    input_shape=(3, 1, 128, 128, 128),
    dtype="bfloat16",
    device="cuda:0",
)
for decision in report.decisions:
    print(decision.disposition, decision.phase, decision.implementation)
```

Reports distinguish `custom` accelerated wrappers, `native` wrappers using
PyTorch, and `unwrapped` convolutions outside the adaptive domain. Output heads,
residual projections, and 2D convolutions remain visible in the report.

See [optimization and profiling](docs/optimization.md) for the YAML schema,
Python API, and migration from old JSON profiles; the
[SM86 RTX A6000 evidence](docs/benchmarks/sm86-profile.md), the
[SM89 L40S evidence](docs/benchmarks/sm89-profile.md), and the
[SM120 RTX 5090 evidence](docs/benchmarks/sm120-profile.md).

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
[`docs/benchmarks/`](docs/benchmarks/), alongside L40S and RTX A6000 policy
evidence. Those historical model numbers are not a fresh benchmark of every
policy-v2 combination. Results are hardware, software, and workload specific;
`auto` supplies inferred defaults immediately, and `mednext-accel profile`
optionally measures overrides for the target environment.

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
