# Optimization, profiling, and compilation

`optimization="auto"` supplies optimized defaults without a tuning run:

```python
from mednext_accel import mednext_base

model = mednext_base(in_channels=1, out_channels=3)
```

Construction loads small bundled policies once and installs adaptive operators.
Neither construction nor first forward benchmarks or writes cache files. Use
`optimization="reference"` for ordinary PyTorch operators. Parameters, tensor
layouts, and state-dict keys are identical in every mode. Checkpoint style and
batch size remain independent user choices.

## Runtime policy YAML

Runtime policies describe **which implementation to use**. Generated evidence
JSON describes **what was measured**. A small policy can express a defensible
inference without pretending every batch was benchmarked. Arithmetic launch
recipes live in tested Python, not YAML expressions.

Load a policy path or a YAML-compatible mapping through `optimization=`:

```python
model = mednext_base(
    in_channels=1,
    out_channels=3,
    optimization="~/.cache/mednext_accel/profiles/sm89-local.policy.yaml",
)
```

This complete example is a manually authored policy for one regular depthwise
shape. Save it as `my-policy.yaml` to use it with the same argument:

```yaml
version: 2
kind: mednext-accel-policy
name: my-sm89-policy
target: {vendor: nvidia, sm: [8, 9]}
scope: {dtype: bfloat16}
rules:
  - id: regular-64
    when:
      family: depthwise_conv3d
      direction: regular
      kernel_size: [3, 3, 3]
      spatial_shape: [64, 64, 64]
      channels: [64, 64]
      batch: {min: 1}
    use:
      backward_input:
        implementation: triton_depthwise_dx
        parameters: auto
      backward_weight:
        implementation: triton_split_dw
        parameters: auto
    confidence: inferred-same-sm
```

Policies use stable operator/phase/implementation identifiers. Regular depthwise
dX (input gradient) and dW (weight gradient) resolve and execute independently;
either can use a custom kernel while the other uses native backward. Other
implementations cover downsample dX, transpose dW, and pointwise training.

Rules are ordered: the first matching rule containing the requested phase wins.
Matches include family, direction, role, kernel size, stride, spatial shape,
input/output channels, dtype, model family/variant, checkpoint context, and
ranges for input/output channels, batch, spatial volume, work, reduction work,
or VRAM. `channels: [in, out]` remains an exact pair; `in_channels` and
`out_channels` separately accept inclusive numeric ranges.
`work = batch × spatial_volume × input_channels`;
`reduction_work = batch × spatial_volume`. An integer means equality;
`{min: ..., max: ...}` is inclusive and either bound may be omitted. Document
`scope` provides common conditions that rules may narrow but cannot contradict.

`parameters: auto` calls the recipe using the actual execution SM and shape.
An explicit mapping overrides named recipe values; omitted parameters produce
an empty mapping. Unknown fields, invalid parameters, and `auto` without a
recipe are rejected. SM86/SM89 regular dW uses
`clamp(round(batch × spatial_volume / 32768), 1, 512)` splits; transpose dW uses
divisor 4096, both with block 512. SM120 retains shape-specific launches and two
measured split-count exceptions. Python `round` uses ties-to-even.

### Layers and reference overrides

Resolution follows these layers, checking implementation guards at each match:

1. External policy supplied by the user.
2. Bundled policy for the execution SM: `sm86`, `sm89`, or `sm120`.
3. `shared-nvidia` policy.
4. Built-in reference implementation.

A missing rule or phase continues to the next layer. A matching custom rule that
fails a hard implementation guard also continues to the next layer. It does
not try later rules in the same layer. If no lower layer supplies an executable
choice, the report retains the highest-priority guard diagnostic. An explicit
`implementation: reference` is a **tombstone**: it stops lower-priority rules for
that matched phase. Policies have no blanket defaults. This rule can precede
positive rules to retain native dW for a known losing region:

```yaml
# A rule to insert under an existing policy's rules list.
- id: regular-16-b10-counterexample
  when:
    family: depthwise_conv3d
    direction: regular
    kernel_size: [3, 3, 3]
    spatial_shape: [16, 16, 16]
    channels: [256, 256]
    batch: 10
  use:
    backward_weight: {implementation: reference}
  confidence: measured-exact-context
```

Unknown NVIDIA SMs use the shared layer; they do not require per-batch lookup
coverage. An external policy targeting a different SM remains active with a
once-only warning. Recipes still use the execution SM. Policies are selected at
execution, so constructing on CPU and moving between CUDA devices does not
freeze the wrong architecture layer.

Hard guards cover device/backend availability, dtype, geometry, evaluation,
export, approximation permission, and launch parameters. Tensor-time checks
such as contiguity run when inputs exist. Unsupported contexts remain runnable
through native operators. The validated generalized domain is NVIDIA CUDA BF16
training with contiguous three-dimensional NCDHW inputs, cubic k3 depthwise
operators, and regular k1 pointwise operators. Other dtypes use native execution
under bundled policies; larger kernels, unsupported layouts/geometries, and
two-dimensional models retain native execution where acceleration is unavailable.

### Inference assumptions and confidence

Confidence describes the policy author's evidence; it never changes dispatch:
`measured-exact-context`, `interpolated-bounded`, `inferred-same-sm`,
`validated-cross-sm`, `extrapolated`, and `guard` are supported labels.
An inferred rule is active in `auto`, including unseen batches in its region.
A measured label does not identify a GPU product: the matcher uses SM, not a
device-name whitelist.

The shared BF16 policy uses these hypotheses across NVIDIA generations:

| Phase | Shared region |
|---|---|
| Pointwise training | Stem 1→32 at 128³, batch ≥1 |
| Downsample dX | Supported k3 stride 2, work ≥1,000,000 |
| Regular dX | Supported k3, work ≥1,500,000 |
| Transpose dW | Supported k3, reduction work ≥4,096 |
| Regular dW | C64–128 with work ≥4,194,304; C512 when reduction work ≥1,024 |

These boundaries extrapolate from operator evidence, not a claim that every
GPU/shape/batch was measured. Bundled policies do not restrict checkpoint style:
matching operator geometry can use them with no checkpointing, expansion, whole
block, or selected stages. Newly generated local overlays also match operators,
not model variants or checkpoint styles. Those contexts remain in evidence for
reproducibility and whole-model comparisons. Policy-v2 model/checkpoint match
keys remain readable for compatibility with existing files.

Generalized rules can select an unmeasured combination of batch, channels, and
spatial dimensions. New depthwise regions are bounded along `work` for dX or
`reduction_work` for dW. New pointwise regions are bounded in input channels,
output channels, and reduction work. Duplicate observations from different
model/checkpoint contexts collapse before synthesis. Positive interpolation does
not cross an observed loser or a missing-result point; exact negative exceptions
precede positive regions. One observation never creates an unbounded positive
range. Existing bundled unbounded rules remain explicit, reviewed same-SM or
cross-SM hypotheses, rather than newly measured coverage.

[SM86](benchmarks/sm86-profile.md) and [SM89](benchmarks/sm89-profile.md) overlays
handle conflicting regular-dW results at 128³ and 16³. SM89 batch 9 inherits
shared rules and the interpolated 128³ dW rule. It is not all-reference because
batch 9 was absent from an old campaign. [SM120](benchmarks/sm120-profile.md)
retains structurally scaled depthwise choices beyond measured RTX 5090 batches.
An SM120 device with 96 GB, such as an RTX PRO 6000, gets those defaults. This is
same-SM inference, not a measured speed claim on that device. Non-stem pointwise
choices remain bounded where timing and memory evidence supports them; VRAM
capacity alone does not establish a speed advantage.

Inspect layers and each phase without allocating the input:

```python
report = model.explain_optimization(
    input_shape=(3, 1, 128, 128, 128),
    dtype="bfloat16",
    device="cuda:0",
)
print(report.policies)
for decision in report.decisions:
    print(
        decision.disposition,
        decision.phase,
        decision.implementation,
        decision.policy,
        decision.rule,
        decision.confidence,
        decision.parameters,
    )
```

The report enumerates adaptive wrappers and native convolutions, including
output heads, residual projections, and 2D convolutions. Each decision exposes
one disposition:

| Disposition | Meaning |
|---|---|
| `custom` | A wrapped phase resolves to an accelerated implementation |
| `native` | A wrapped phase resolves to PyTorch, including reference tombstones and guard fallback |
| `unwrapped` | A convolution is outside the adaptive wrapper domain |

Decisions also expose warnings, guard reasons, and tensor-time execution guards.
The allocation-free report cannot observe tensor contiguity or every execution
guard. It describes routing, not a latency prediction or a guarantee that a
custom implementation is fastest at every point in a region.

## Run an optional profiling campaign

From an installed package:

```bash
mednext-accel profile
```

The default measures Small/Base/Medium/Large at 128³, BF16 training, without
checkpointing. For one workload, save this as `workload.yaml`:

```yaml
preset: mednext-v1
objective: balanced
compile_mode: max-autotune-no-cudagraphs
seed: 0
workloads:
  - variant: base
    spatial: [128, 128, 128]
    checkpointing: all-expansion
batch_search:
  memory_fraction: 0.90
  maximum: 16  # optional upper bound; omit to discover one
```

```bash
mednext-accel profile workload.yaml
```

Checkpoint values are `none`, `all-expansion`, `whole-block`, or `auto` (all three
contexts). The profiler does not choose checkpointing for application code.
Training/BF16 are the current scope; no `phases` argument is needed. Channel
counts default to one input and three outputs; `in_channels`, `out_channels`,
and `dtypes: [bfloat16]` may be stated explicitly in a workload. Unknown workload
fields are rejected, including unsupported `kernel_size` or `base_channels`
overrides, so a campaign cannot silently ignore them.

The profiler finds the largest feasible batch under `memory_fraction` of total
VRAM. An explicit `maximum` is tried first and ends search in one probe if it
fits. Otherwise binary refinement finds the boundary; without a maximum,
exponential growth first locates an upper bound. CUDA probes run in children so
an OOM can be recorded without poisoning the parent. Only the selected maximum
is kernel-profiled. Rerun with another maximum for evidence at another batch;
application code may use inferred policies at other batches without another
run. Batch probes use the reference model. The boundary is specific to the
profiler's model-only workload and budget, not a guarantee for an application's
optimizer, loss, or data pipeline.

Model probes compile with `fullgraph=True`, use AdamW, no deep supervision, and
mean squared primary logits as a synthetic loss. Peak allocated memory includes
model parameters, gradients, initialized optimizer states, activations, and
temporary buffers. Each comparison uses one warmup and one measured step, so it
is a lightweight policy gate rather than a repeated training benchmark. It is
not directly comparable to the README's cross-entropy/deep-supervision table.

Interactive terminals use Rich; redirected output and `tee` use plain text.
`--progress auto|plain|quiet` controls the display. Batch search shows elapsed
time because its length is adaptive. Fixed-size phases show counts and ETA:

```text
kernel plan: 20 planned groups, 192 unique experiments, 576 requested references
Kernel groups 20/20 · experiments 192/192
validation plan: 2 whole-model validations
Validation 2/2
```

These counts are illustrative. A unique kernel experiment may serve several
workloads; it is timed once and projected into their contexts. Groups are child
attempts and can increase when failed groups are bisected. Model comparisons and
batch searches remain workload-specific. The CLI uses `argparse`, with Rich
only for progress rendering.

### Integrated measurements and dispatch evidence

Each candidate runs through the actual adaptive module interface, including
policy resolution, custom autograd, BF16 autocast with FP32 trainable parameters,
the requested dX/dW/dB mask, and the campaign compile mode. Native PyTorch runs
through the same module-level interface. Output and every requested gradient
are validated before timing. Unrequested gradients are not computed. Coverage
includes regular depthwise dX/dW, downsample dX, transpose dW, and pointwise
training; other phases remain native within an isolated candidate comparison.

`benchmark_kind: integrated_operator` identifies dispatch evidence. Associated
`raw_kernel_diagnostic` measurements remain useful for kernel analysis, but
cannot create positive dispatch rules or reference tombstones. Implementation
and launch parameters identify evidence; comparison seeds exclude those fields
so competing candidates receive identical tensors. Invalid or losing candidates
are rejected individually. The fastest qualifying candidate wins for throughput
and balanced objectives; the lowest-memory qualifying candidate wins for memory.

CUDA Graph replay can hide retained working storage from measured allocator
peaks. Those operator results retain observed peak scalars but set
`memory_measured: false`; they cannot qualify for balanced or memory objectives
or count as measured losers for those objectives. Throughput can use valid
latency evidence. Use `eager`, `default` without CUDA Graphs, or
`max-autotune-no-cudagraphs` when memory evidence is required. Resetting peak
statistics alone does not make replay memory measurements complete.

### Two outputs and acceptance

The CLI prints both paths. Default directory:
`$XDG_CACHE_HOME/mednext_accel/profiles/`, or `~/.cache/mednext_accel/profiles/`.

| Artifact | Purpose |
|---|---|
| `smXX-local.policy.yaml` | Stable runtime overlay passed to `optimization=` |
| `smXX-local.<timestamp>-<hash>.evidence.json` | Immutable measurements and diagnostics; never passed to `optimization=` |

Evidence records GPU/SM/VRAM, driver, platform, Python/PyTorch/CUDA/cuDNN/Triton,
package version, normalized campaign and seed, ordered batch attempts plus a
batch-indexed view, kernel timings/memory and component errors, failure stages,
model reference/candidate metrics, acceptance reasons, and effective policy
identities. It omits usernames and hostnames. Pointwise forward, dX, dW, and dB
validation is streamed one component at a time, with FP32 trainable parameters
under BF16 autocast and a 2% relative-L2 threshold. This reduces **validation
scratch memory**; it does not reduce candidate runtime training memory.

The provisional overlay contains bounded operator winner regions, exact launch
exceptions, and exact reference tombstones for numerical failures or complete
measured objective losers when no qualifying candidate wins.
Infrastructure errors, missing metrics, and validation OOM leave gaps. The
complete overlay is tested over the same bundled layers used by application
models. Objective gates differ between integrated operators and the model:

| Objective | Integrated operator gate | Model gate |
|---|---|---|
| `balanced` | time ≤97%, peak ≤115% of reference | time <100%, peak ≤115% |
| `throughput` | time <100% | time <100%, peak ≤125% |
| `memory` | peak <100%, time ≤110% | peak <100%, time ≤110% |

Missing, invalid, or nonfinite metrics cannot pass. All model comparisons must
accept before positive rules are published. If any rejects, or no comparison
exists, the published overlay keeps only negative tombstones. Evidence preserves
all numerical results and records that remaining bundled fallthrough was **not
validated by that comparison**. A rejection does not mean every kernel is
invalid. Whole-model numerical equivalence is explicitly `not-measured`;
component validation supplies the numerical evidence.

Full-model comparisons are a release sentinel and publication gate for a new
bundled SM policy. They protect against graph-level regressions even when
isolated operators win; they are not required at every point in an operator
grid. A model pass does not establish optimal dispatch or numerical equivalence
for every generalized shape.

The SHA-256 in YAML covers exact evidence bytes. Evidence is published first
under a unique filename and never overwritten; the complete policy is then
atomically replaced. Interrupted publication may leave orphan evidence but not
a policy referencing overwritten evidence. Loading a policy does not require
its evidence file. Generated local artifacts are excluded from Git and package
archives; keep custom outputs under `artifacts/` in this repository.

### Python API

```python
from pathlib import Path
from mednext_accel import mednext_base
from mednext_accel.profiling import profile

result = profile("workload.yaml")  # runs probes and publishes both default files
print(result.artifacts.policy_path)
print(result.artifacts.evidence_path)
print(result.evidence.publication.reason)

model = mednext_base(in_channels=1, out_channels=3, optimization=result.artifacts.policy_path)

# Optional: publish another pair without rerunning probes.
artifacts = result.save(policy_path=Path("artifacts/my-local.policy.yaml"))
```

`ProfilingResult` contains `policy`, `evidence`, `artifacts`, and immutable
`environment`. `ProfilingArtifacts` contains `policy_path` and `evidence_path`.
`save(*, policy_path=None, evidence_directory=None)` returns new paths and leaves
the original result unchanged. With no arguments it updates the original stable
policy and writes new immutable evidence. An optional separate evidence directory
changes where evidence is written; the policy stores its basename/hash, and
runtime never follows it as a dependency.

## Migrating old profiles

- `optimization="auto"` and `optimization="reference"` retain their meanings;
  model checkpoints need no conversion for this policy change.
- Schema-v1 runtime JSON is rejected. Rerun the profiler and pass the printed
  `.policy.yaml` path, or author a policy-v2 mapping. Renaming a v1 file is not a
  conversion. The loader also accepts JSON encoding of a **v2 policy**, but the
  profiler writes YAML and evidence JSON is never a runtime policy.
- Campaign `phases` was removed because profiling measures training only.
  Remove it from existing YAML. Obsolete `dense_until`/dense-batch options are
  rejected; use the optional `batch_search.maximum` upper bound.
- Python callers use `result.policy`, `result.evidence.kernel_measurements`, and
  `result.artifacts.policy_path` instead of the old `profile`, `measurements`,
  and `output_path` result attributes. `save()` is keyword-only and returns
  `ProfilingArtifacts`, not a single path.
- Old raw JSON remains useful historical evidence, not a runtime input. Its
  measurements can inform reviewed bundled rules. Historical design plans under
  `docs/superpowers/` carry superseded notices.

## torch.compile and portable weights

Compile after model construction and before the first measured step:

```python
model = model.cuda().train()
model.compile(mode="default", fullgraph=True)
```

`fullgraph=True` is recommended when the complete graph is supported. RTX 5090
measurements found `default` a useful general choice and
`max-autotune-no-cudagraphs` useful for long fixed-shape runs that amortize its
larger compile cost. `reduce-overhead` and `max-autotune` may retain extra CUDA
Graph memory. Warm up before timing and report compile cost separately.

Compilation changes execution graphs, not weights. Saving `model.state_dict()`
remains portable. The checkpoint loader removes `_orig_mod.` from a state dict
produced by `torch.compile(model)`. Compiler artifacts are never checkpoint
tensors. Evaluation uses ordinary PyTorch operators for `jit.trace`, strict
`torch.export`, and ONNX; see [export](export.md) and
[checkpoint compatibility](checkpoints.md) for the tested scope.
