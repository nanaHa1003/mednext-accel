# MedNeXt activation-memory optimization roadmap

## Scope

This document covers model-architecture and operator changes that can reduce
training activation memory. It does not define a trainer, loss, dataset, data
pipeline, or nnU-Net integration.

MedNeXt v1 and v2 should remain separate model implementations. Shared pieces
such as checkpoint policies, output containers, and optimized operators should
be independent modules when they are also useful to other 3D ConvNeXt-like
architectures. An optimization must preserve model parameters and mathematical
behavior unless its API explicitly selects different output semantics.

The priorities are:

1. Selective checkpointing of the expanded branch.
2. Resolution-aware checkpointing.
3. Fused 3D Global Response Normalization (GRN).
4. Memory-efficient GroupNorm.
5. Configurable deep-supervision output, including MONAI-compatible behavior.

## Measurement context

The initial feasibility measurements use an RTX 5090, PyTorch 2.12.0+cu132,
cuDNN 9.20, BF16 autocast, batch size one, input `128x128x128`, three output
classes, deep supervision, AdamW, and `torch.compile`. Inputs and targets remain
resident on the GPU. Each result below has two warmup steps and five measured
steps, so it establishes direction rather than publication-quality statistics.

Peak memory means PyTorch peak allocated memory. It excludes allocator reserve,
CUDA context memory, and unrelated processes.

### Initial checkpoint results

| Model | Activation policy | Step time | Peak allocated |
|---|---|---:|---:|
| v1 Base | none | 74.52 ms | 8,423 MiB |
| v1 Base | expanded branch | 79.76 ms | 3,851 MiB |
| v1 Base | whole block | 87.26 ms | 3,284 MiB |
| v1 Large | none | 138.15 ms | 17,731 MiB |
| v1 Large | expanded branch | 148.16 ms | 5,945 MiB |
| v1 Large | whole block | 159.89 ms | 5,674 MiB |

For Base, expanded-branch checkpointing reduces peak allocation by 54.3% for a
7.0% step-time increase. It obtains approximately 89% of the memory reduction
of whole-block checkpointing while remaining 8.6% faster than the whole-block
policy.

For Large, expanded-branch checkpointing reduces peak allocation by 66.5% for a
7.2% step-time increase. It obtains approximately 97.8% of the memory reduction
of whole-block checkpointing while remaining 7.3% faster than the whole-block
policy.

The Large split-depthwise path also composes with selective checkpointing:

| Configuration | Step time | Peak allocated |
|---|---:|---:|
| Native convolution, no checkpoint | 138.15 ms | 17,731 MiB |
| Split depthwise, no checkpoint | 113.96 ms | 17,731 MiB |
| Split depthwise, expanded-branch checkpoint | 125.40 ms | 5,947 MiB |

A double-precision single-block check found maximum absolute differences of
zero for output, input gradient, and every parameter gradient between ordinary
execution and expanded-branch checkpointing.

### Implemented-policy validation

The production policy was measured on an RTX 5090 with driver 595.84,
PyTorch 2.12.0+cu132, CUDA 13.2, cuDNN 9.20, and Triton 3.7.0. The workload is
MedNeXt Base, batch size one, `128x128x128`, BF16 autocast, deep supervision,
AdamW, and `torch.compile`. Each entry uses 20 warmup steps and reports the
median of three independent 50-step runs. Forward and backward are medians of
the per-run GPU-event medians. Peak memory is maximum PyTorch allocation.

Native convolution results:

| Classes | Policy | Raw step times | Step median ± SD | Forward | Backward | Peak allocated |
|---:|---|---|---:|---:|---:|---:|
| 3 | `none` | 73.64, 74.84, 74.95 ms | 74.84 ± 0.73 ms | 22.18 ms | 51.72 ms | 8,426 MiB |
| 3 | `expanded` | 79.07, 80.31, 80.49 ms | 80.31 ± 0.77 ms | 20.21 ms | 59.39 ms | 3,852 MiB |
| 3 | `block` | 85.38, 85.69, 86.91 ms | 85.69 ± 0.81 ms | 19.05 ms | 65.99 ms | 3,285 MiB |
| 8 | `none` | 75.54, 76.43, 76.55 ms | 76.43 ± 0.55 ms | 22.17 ms | 52.75 ms | 8,751 MiB |
| 8 | `expanded` | 80.24, 81.14, 82.05 ms | 81.14 ± 0.90 ms | 20.08 ms | 59.66 ms | 4,052 MiB |
| 8 | `block` | 86.78, 86.85, 88.64 ms | 86.85 ± 1.05 ms | 18.76 ms | 66.70 ms | 3,488 MiB |

With three classes, `expanded` reduces peak allocation by 54.3%, costs 7.3%
relative to `none`, and is 6.3% faster than `block`. With eight classes, it
reduces peak allocation by 53.7%, costs 6.2%, and is 6.6% faster than `block`.

The current pointwise-GEMM plus split-depthwise configuration produces:

| Classes | Policy | Raw step times | Step median ± SD | Forward | Backward | Peak allocated |
|---:|---|---|---:|---:|---:|---:|
| 3 | `none` | 60.56, 60.57, 60.69 ms | 60.57 ± 0.07 ms | 23.67 ms | 36.02 ms | 8,375 MiB |
| 3 | `expanded` | 70.35, 70.61, 70.66 ms | 70.61 ± 0.16 ms | 23.41 ms | 46.38 ms | 3,914 MiB |
| 3 | `block` | 79.49, 79.52, 79.60 ms | 79.52 ± 0.06 ms | 23.25 ms | 55.44 ms | 3,345 MiB |
| 8 | `none` | 62.37, 62.44, 62.59 ms | 62.44 ± 0.11 ms | 23.83 ms | 37.36 ms | 8,815 MiB |
| 8 | `expanded` | 72.34, 72.56, 72.73 ms | 72.56 ± 0.19 ms | 23.55 ms | 47.71 ms | 4,117 MiB |
| 8 | `block` | 80.94, 81.09, 81.13 ms | 81.09 ± 0.10 ms | 23.31 ms | 56.42 ms | 3,548 MiB |

With the custom operators, `expanded` still saves approximately 53.3% of peak
allocation and is 10.5--11.2% faster than `block`. Its cost relative to `none`
rises to 16.2--16.6% because accelerating depthwise backward makes expansion
recomputation a larger fraction of the remaining step time. This is a useful
memory/performance point, but callers that fit the no-checkpoint model should
prefer `none` for maximum throughput.

## 1. Selective expanded-branch checkpointing

### Motivation

A MedNeXt block contains:

```text
depthwise convolution -> GroupNorm -> pointwise expansion -> GELU
    -> optional GRN -> pointwise compression -> residual addition
```

The expanded activation has `C * R` channels. At the full-resolution stage of
MedNeXt Large, one BF16 `[1, 96, 128, 128, 128]` tensor is approximately
384 MiB. Autograd may need both pre-activation and post-activation values for
the GELU and compression-convolution backward passes. These long-lived expanded
tensors dominate the activation footprint.

### Implemented behavior

Checkpoint only the following branch:

```text
v1: pointwise expansion -> GELU -> pointwise compression
v2: pointwise expansion -> GELU -> GRN -> pointwise compression
```

The forward pass retains the normalized `C`-channel input to this branch.
Backward recomputes the `C * R` expansion instead of retaining it across the
forward/backward boundary. Depthwise convolution and GroupNorm are not
recomputed.

Use non-reentrant PyTorch checkpointing. Apply it only when the module is in
training mode and gradients are enabled. Evaluation must execute the branch
directly.

### Compatibility requirements

- Do not move or rename convolution Parameters.
- Preserve all state-dict keys.
- Preserve the eager and compiled forward result.
- Support AMP and `torch.compile`.
- Compose with native and custom depthwise operators.
- Treat this as an execution policy, not a new model variant.

The production package exposes both expansion-branch and whole-block
checkpointing through `CheckpointConfig`. The expansion policy is the default
because it provides a better time/memory tradeoff; the whole-block policy remains
available when minimizing activation memory is more important than throughput.

Unit coverage verifies exact double-precision output, input-gradient, and
parameter-gradient parity for normal, downsample, and upsample blocks. It also
verifies strict state-dict loading across policies, both deep-supervision
states, evaluation and `no_grad()` bypass, BF16 autocast, custom-operator
composition, and `torch.compile(fullgraph=True)`.

### Validation remaining

- Repeat the production matrix on L40S, RTX A6000, and RTX 8000.
- Add FP32 and FP16 model-level numerical comparisons; BF16 compiled execution
  and double-precision exact comparisons are covered locally.
- Test multi-process DDP. Single-process autograd and compiled execution are
  covered, but no distributed claim is made yet.
- Repeat the Large measurements with the implemented policy. Existing Large
  numbers above are feasibility measurements from the prototype.

## 2. Resolution-aware checkpointing

### Motivation

The highest-resolution stages dominate activation memory. Recomputing every
expanded branch spends compute on small tensors that contribute little to peak
allocation.

For v1 Base, a single full-resolution expanded BF16 activation is roughly:

```text
[1, 64, 128, 128, 128] = 256 MiB
```

Activations at successive resolutions shrink by approximately a factor of four
because spatial volume falls by eight while channels double. This makes the
`128^3` and `64^3` stages the first checkpoint candidates.

### Implemented behavior

`CheckpointConfig.stages` selects static architectural resolution levels:

```python
mednext_base(
    ...,
    checkpointing=CheckpointConfig(stages=(0, 1)),
)
```

The same stage selection applies to whole-block recomputation:

```python
mednext_base(
    ...,
    checkpointing=CheckpointConfig(style="block", stages=(0, 1)),
)
```

Level zero is full resolution and each increment is one factor-of-two spatial
downsampling. `None` selects all levels, preserving the original `expanded`
behavior. The tuple is validated and sorted at model construction. Selection
uses the resolution at which the expansion branch operates, so encoder,
decoder, downsample, and upsample blocks follow one definition without reading
runtime tensor shapes.

### Compatibility requirements

- The policy must not affect state dicts.
- A saved policy is configuration metadata, not a tensor weight.
- Dynamic input shapes must not silently select a different mathematical model.
- Checkpointing a stage changes compute and memory only.

### Decision rule

Keep resolution-aware selection only if it provides a useful point between
`none` and `expanded` on the Pareto curve. It should be removed if the added
configuration surface yields negligible speed improvement.

### RTX 5090 results

The following results use the same RTX 5090, software environment, Base
`128x128x128` BF16 workload, and measurement protocol as the implemented-policy
validation above. Every entry is the median of three independent 50-step runs
after 20 warmup steps. Time percentages and memory savings are relative to
`none` for the same class count and operator path.

Native convolution:

| Classes | Policy | Selected levels | Step median ± SD | Time vs. none | Peak allocated | Memory saved |
|---:|---|---|---:|---:|---:|---:|
| 3 | none | none | 73.39 ± 1.19 ms | baseline | 8,426 MiB | baseline |
| 3 | expanded | `(0,)` | 77.08 ± 1.03 ms | +5.0% | 5,883 MiB | 30.2% |
| 3 | expanded | `(0, 1)` | 77.96 ± 0.05 ms | +6.2% | 4,447 MiB | 47.2% |
| 3 | expanded | `(0, 1, 2)` | 78.35 ± 0.05 ms | +6.8% | 3,990 MiB | 52.6% |
| 3 | expanded | all | 78.54 ± 0.06 ms | +7.0% | 3,852 MiB | 54.3% |
| 3 | block | all blocks | 84.86 ± 0.03 ms | +15.6% | 3,285 MiB | 61.0% |
| 8 | none | none | 74.98 ± 0.07 ms | baseline | 8,751 MiB | baseline |
| 8 | expanded | `(0,)` | 78.66 ± 0.11 ms | +4.9% | 6,083 MiB | 30.5% |
| 8 | expanded | `(0, 1)` | 79.70 ± 0.17 ms | +6.3% | 4,647 MiB | 46.9% |
| 8 | expanded | `(0, 1, 2)` | 80.10 ± 0.07 ms | +6.8% | 4,190 MiB | 52.1% |
| 8 | expanded | all | 80.35 ± 0.11 ms | +7.2% | 4,052 MiB | 53.7% |
| 8 | block | all blocks | 86.77 ± 0.13 ms | +15.7% | 3,488 MiB | 60.1% |

Pointwise-GEMM plus split-depthwise:

| Classes | Policy | Selected levels | Step median ± SD | Time vs. none | Peak allocated | Memory saved |
|---:|---|---|---:|---:|---:|---:|
| 3 | none | none | 59.46 ± 0.04 ms | baseline | 8,375 MiB | baseline |
| 3 | expanded | `(0,)` | 65.93 ± 0.07 ms | +10.9% | 5,946 MiB | 29.0% |
| 3 | expanded | `(0, 1)` | 68.62 ± 0.05 ms | +15.4% | 4,510 MiB | 46.2% |
| 3 | expanded | `(0, 1, 2)` | 69.04 ± 0.05 ms | +16.1% | 4,053 MiB | 51.6% |
| 3 | expanded | all | 69.20 ± 0.03 ms | +16.4% | 3,914 MiB | 53.3% |
| 3 | block | all blocks | 77.79 ± 0.11 ms | +30.8% | 3,345 MiB | 60.1% |
| 8 | none | none | 61.03 ± 0.06 ms | baseline | 8,815 MiB | baseline |
| 8 | expanded | `(0,)` | 67.57 ± 0.06 ms | +10.7% | 6,149 MiB | 30.2% |
| 8 | expanded | `(0, 1)` | 70.33 ± 0.08 ms | +15.2% | 4,713 MiB | 46.5% |
| 8 | expanded | `(0, 1, 2)` | 70.77 ± 0.08 ms | +16.0% | 4,255 MiB | 51.7% |
| 8 | expanded | all | 71.04 ± 0.03 ms | +16.4% | 4,117 MiB | 53.3% |
| 8 | block | all blocks | 79.46 ± 0.08 ms | +30.2% | 3,548 MiB | 59.8% |

All sampled settings are strict Pareto points: each additional level trades
some time for lower peak allocation. The most useful practical choices are:

- `(0,)` when a roughly 30% memory reduction is sufficient;
- `(0, 1)` when approximately 46--47% is needed;
- all expanded levels when maximum selective-checkpoint memory reduction is
  preferred.

Adding level 1 after level 0 removes approximately another 1.4 GiB. Adding
level 2 removes approximately 457 MiB. Checkpointing the remaining levels
removes only about 138 MiB, so `(0, 1, 2)` and all-level execution are close in
both time and memory. The tuple API retains every choice without adding preset
names. Portability measurements on L40S, RTX A6000, and RTX 8000 remain
outstanding.

## 3. Fused 3D GRN

### Motivation

GRN is specific to MedNeXt v2 and other ConvNeXt-v2-like models. A naive eager
implementation materializes several large pointwise intermediates and caused a
large time and memory increase in the initial v2 probe. Compilation eliminated
almost all additional peak memory, so the primary goal of a custom operator is
an efficient eager path without regressing compiled execution.

For `x` in NCDHW layout, use the ConvNeXt-v2 definition:

```text
g[n,c] = sqrt(sum_spatial(x[n,c]^2))
h[n]   = mean_channel(g[n,:]) + eps
r[n,c] = g[n,c] / h[n]
y      = x + gamma * x * r + beta
```

Use `eps=1e-6`, zero-initialized `gamma` and `beta`, and FP32 reduction
accumulation. Do not materialize full-sized `r` or `x * r` tensors.

### Proposed operator boundary

GRN should be an architecture-independent layer with:

- A pure PyTorch reference implementation.
- A Triton forward and backward implementation.
- Automatic fallback for CPU, unsupported dtypes, noncontiguous layouts, and
  unsupported shapes.
- Per-shape benchmarking before selecting the Triton implementation.

The forward kernels compute spatial sum-of-squares partials, finish channel
norms, reduce the channel mean, and apply the affine residual expression. The
backward kernels compute spatial partials for `sum(dy*x)` and `sum(dy)`, finish
the coupled channel derivative, and apply `dX`. FP16 and BF16 reductions use
FP32 partials.

Integrate the kernels with `torch.library.triton_op` and registered autograd so
the implementation remains visible to `torch.compile`. A standalone custom-op
boundary must not be enabled by default unless a full-model benchmark shows it
is beneficial.

### Success criteria

- Correct forward, dX, dGamma, and dBeta against the PyTorch reference.
- Correct zero-input behavior without division by zero.
- Batch-one and multi-batch support.
- `torch.library.opcheck` and compiled full-graph coverage.
- A substantial eager memory reduction.
- No material compiled full-step regression.
- Independent results on RTX 5090, L40S, RTX A6000, and RTX 8000.

Fusion with GELU or the compression GEMM is outside the first implementation.

## 4. Memory-efficient GroupNorm

### Motivation

MedNeXt uses `GroupNorm(C, C)`, which is mathematically instance normalization
for NCDHW tensors. GroupNorm operates on `C` channels rather than the expanded
`C * R` representation, so its memory opportunity is smaller than selective
checkpointing. It remains relevant after the expanded tensors are removed.

### Candidate approaches

1. Keep native GroupNorm and rely on `torch.compile` fusion.
2. Recompute normalized values in backward while saving only the input and
   per-instance/channel statistics.
3. Implement a Triton GroupNorm/InstanceNorm forward and backward using FP32
   reductions and fused affine application.
4. Investigate a fused depthwise-convolution-to-normalization boundary only
   after standalone GroupNorm measurements justify the complexity.

Kernel fusion alone does not remove an activation that autograd deliberately
saves for backward. A custom backward or checkpoint boundary is necessary for
a material lifetime reduction.

### Decision rule

Profile this only after selective expanded-branch checkpointing is integrated.
Proceed with a custom operator only if GroupNorm becomes a significant share of
the resulting compiled full step or materially limits peak allocation. Preserve
the native implementation as the default until then.

## 5. Configurable deep-supervision output

### Current repository behavior

During training with deep supervision enabled, every auxiliary logit tensor is
interpolated to the full output resolution and stacked:

```text
[N, heads, classes, D, H, W]
```

The head order is full resolution followed by `1/2`, `1/4`, `1/8`, and `1/16`
decoder resolutions, after interpolation. Evaluation returns only the final
full-resolution tensor.

For BF16, five full-resolution heads at `128^3` require approximately 60 MiB
for three classes and 160 MiB for eight classes for the final stacked tensor
alone. Interpolation results and backward state can raise the peak further.

### MONAI behavior

MONAI 1.5.2 returns native-resolution logits during training:

```python
(
    full_resolution,
    half_resolution,
    quarter_resolution,
    eighth_resolution,
    sixteenth_resolution,
)
```

The return type is a tuple. MONAI does not interpolate the auxiliary heads to
full resolution and does not stack them. When the model is in evaluation mode,
it returns only the full-resolution tensor even if deep supervision is enabled.

The head ordering in the current repository and MONAI is semantically the same;
the differences are native versus interpolated spatial shapes and tuple versus
stacked-tensor representation.

### Proposed switch

Keep deep supervision independently switchable from its output representation:

```text
deep_supervision = false
    train/eval -> full-resolution Tensor

deep_supervision = true, output = "monai"
    train -> tuple of native-resolution Tensor objects
    eval  -> full-resolution Tensor

deep_supervision = true, output = "stacked"
    train -> all heads interpolated and stacked at full resolution
    eval  -> full-resolution Tensor
```

A proposed future argument is:

```python
deep_supervision_output: Literal["monai", "stacked"]
```

The default is deliberately undecided until the public API and backward-
compatibility policy are designed. The existing implementation uses `stacked`;
MONAI interoperability favors `monai`.

### Compatibility boundaries

- `monai` mode should match MONAI's type, head ordering, native spatial shapes,
  and train/eval behavior.
- `stacked` mode should preserve the current repository behavior.
- The model returns logits only. It does not resize targets, assign auxiliary
  loss weights, or compute a loss.
- Checkpoint weights must be independent of the selected output representation.
- Changing output representation is an explicit API behavior change, even
  though it does not change model Parameters.
- Output-mode tests must cover odd valid input sizes because upsampling and
  native auxiliary shapes can otherwise expose alignment differences.

### Validation still required

- Direct output-contract comparison against the supported MONAI version.
- Peak-memory and step-time comparison for three and eight classes.
- Eager and compiled coverage for both output modes.
- Training/evaluation transition tests.
- Tests showing identical final full-resolution logits between modes.

## Recommended implementation order

1. Turn the expanded-branch prototype into a tested execution policy without
   changing state-dict keys.
2. Measure high-resolution-only policies and retain only useful Pareto points.
3. Add the deep-supervision output switch and MONAI contract tests.
4. Re-profile the resulting Base and Large models.
5. Implement fused GRN if eager v2 remains a supported performance target.
6. Reconsider GroupNorm only if the new profile justifies it.

This order prioritizes the largest measured memory reduction, establishes the
output compatibility contract early, and postpones custom kernels until the
remaining bottlenecks are measured in the updated model.
