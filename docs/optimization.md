# Optimization, profiling, and compilation

MedNeXt factories select optimized operators by default:

```python
from mednext_accel import mednext_base

model = mednext_base(in_channels=1, out_channels=3)
```

`optimization="auto"` loads immutable bundled profiles. It does no timing and
writes no cache during construction or first forward. `optimization="reference"`
uses ordinary PyTorch operators. A JSON/YAML path or YAML-compatible mapping
loads a profile generated elsewhere. Parameters, tensor layouts, and state-dict
keys are identical in every mode.

Profiles resolve each operator phase separately. Current identifiers cover
pointwise training, regular depthwise dX and dW, transpose depthwise dW,
downsample depthwise dX, reference export paths, and opt-in approximate GELU.
Resolution order is external exact overrides, external rules, exact-SM bundled
rules, then generic family defaults. Unsupported kernel geometry uses a reported
correctness guard. Unknown batches and shapes always receive a decision.

An external profile records its source SM. A mismatch emits one warning and the
profile remains active. An unknown NVIDIA SM uses `generic-nvidia`, seeded from
RTX 5090 measurements, with one warning. This behavior lets users knowingly test
profiles across machines while keeping the provenance visible.

Inspect decisions without allocating the requested input:

```python
report = model.explain_optimization(
    input_shape=(3, 1, 128, 128, 128),
    dtype="bfloat16",
    device="cuda:0",
)
```

## Generate a local profile

Run the standalone profiler from an installed package:

```bash
mednext-accel profile
```

It detects GPU architecture and VRAM, profiles installed MedNeXt v1 variants,
and writes one profile to `~/.cache/mednext_accel/profiles/`. Each memory probe
and kernel benchmark runs in a separate process, so CUDA OOM does not poison the
campaign. Batch search tests every integer through eight when feasible, expands
by powers of two, then uses binary refinement at the memory boundary.

Interactive terminals show a Rich progress display. Redirected output and
`tee` automatically use stable plain-text lines. Override the selection with
`--progress auto`, `--progress plain`, or `--progress quiet`. Batch search shows
elapsed time while its total is still unknown; after feasible batches are known,
operator profiling shows completed/total work and ETA.

The generated profile records the GPU name, SM, VRAM, NVIDIA driver, platform,
Python, mednext-accel, PyTorch, CUDA, cuDNN, Triton, objective, and compile mode
under `profile.provenance`. It omits hostnames and usernames.

A minimal campaign can narrow the workload:

```yaml
preset: mednext-v1
workloads:
  - variant: base
    spatial: [128, 128, 128]
    checkpointing: auto
batch_search:
  strategy: auto
  memory_fraction: 0.90
```

```bash
mednext-accel profile workload.yaml
```

`checkpointing: auto` measures no checkpointing, all expansion branches, and
whole blocks. It does not select checkpointing for application code; the user
still chooses checkpoint style and batch size. The profiler validates numerical
results, records latency and peak allocation, merges adjacent batches with the
same winner, and retains measured launch parameters. Generated output is a
normal factory input.

## torch.compile

Compile after model construction and before the first measured step:

```python
model = model.cuda().train()
model.compile(mode="default", fullgraph=True)
```

PyTorch modes trade cold-start cost, kernel search, and CUDA Graph memory.
`default` is the general recommendation. RTX 5090 measurements also found
`max-autotune-no-cudagraphs` useful for long fixed-shape runs. `reduce-overhead`
and `max-autotune` may retain additional memory through CUDA Graphs. Warm up the
compiled graph before timing and report compilation separately.

Compilation changes execution graphs, not weights. Saving the original model's
state dict remains portable. The checkpoint loader also removes `_orig_mod.`
when a state dict comes from the wrapper returned by `torch.compile(model)`.
Compiler artifacts are never part of a checkpoint.
