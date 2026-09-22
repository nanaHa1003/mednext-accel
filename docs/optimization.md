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
profile remains active. SM89 selects the measured L40S profile automatically,
and SM120 selects the RTX 5090 profile. An unknown NVIDIA SM uses
`generic-nvidia`, seeded from RTX 5090 measurements, with one warning. This
behavior lets users knowingly test profiles across machines while keeping the
provenance visible.

The bundled SM89 evidence covers MedNeXt v1 Base, 128³ input, BF16 training,
all-expansion checkpointing, and legacy dense batches 1–8 and 10. Its 78 rules
are guarded by tensor shape, channels, batch, dtype, phase, and checkpoint
policy. Batch 9 and every nonmatching context use reference PyTorch operators.
The pointwise evidence is deliberately narrower than the depthwise evidence
because the legacy campaign's pointwise validator produced false negatives;
those old outcomes were not relabeled. See the
[SM89 evidence and limitations](benchmarks/sm89-profile.md) for the exact
environment, operator-level results, and rerun procedure.

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
and writes one profile to `~/.cache/mednext_accel/profiles/`. CUDA work runs in
child processes, so an OOM does not poison the campaign. Batch search locates
one maximum feasible batch for each workload. A supplied `maximum` is tested
first and needs only one model probe when it fits; without one, exponential
growth finds an infeasible upper bound and binary search refines the boundary.

Interactive terminals show a Rich progress display. Redirected output and
`tee` automatically use stable plain-text lines. Override the selection with
`--progress auto`, `--progress plain`, or `--progress quiet`. Batch search shows
elapsed time while its total is still unknown. It remains specific to model
variant, spatial shape, channels, checkpoint context, and compile mode because
those inputs affect whole-model memory and step time; its probes are therefore
not globally deduplicated.

Once the maximum feasible batch is selected, the fixed-size kernel and
validation phases show completed/total work and ETA. Diagnostic model probes
used to locate the boundary are not kernel experiments. A compact
representative display is:

```text
kernel plan: 20 planned groups, 192 unique experiments, 576 requested references
Kernel groups 20/20 · experiments 192/192
validation plan: 2 whole-model validations
Validation 2/2
```

These numbers illustrate the display rather than report a benchmark. An
`experiment` is one unique logical kernel case, identified by the operator,
phase, batch, tensor shape, dtype, candidate, and launch parameters. Each use by
a workload and checkpoint context is one requested reference, so the same
experiment may satisfy several references without being timed again.
`Groups` count actual child-process attempts. If a child fails, the profiler
bisects its cases and retries both halves, so the group total and final count can
grow beyond the original plan.

Shared kernel results are projected back into each workload before synthesis.
Only the selected maximum is kernel-profiled and forms a measured claim in the
generated profile. Whole-model validation runs once for that selected batch.
Those checks remain context-specific, so a result from one model or checkpoint
configuration is never used to validate another. Rerun the campaign with a
different `maximum` to collect evidence for another batch.

The generated profile records the GPU name, SM, VRAM, NVIDIA driver, platform,
Python, mednext-accel, PyTorch, CUDA, cuDNN, Triton, objective, and compile mode
under `profile.provenance`. Its `execution` object records unique kernel cases,
actual child attempts (including bisection retries), actual whole-model endpoint
probes, and the requested references removed by kernel deduplication. Its
`campaign` object preserves the complete normalized campaign, including batch
search settings and every workload's model family, variant, shape, dtype,
phase, checkpoint policy, and channel counts. New measurements also retain the
child status, kernel numerical result, whole-model result, validator version,
per-component relative-L2 and maximum-absolute errors, and a rejection reason.
Profiles omit hostnames and usernames. Profiles without `execution`, `campaign`,
or the newer validation diagnostics remain valid schema-v1 documents.

A minimal campaign can narrow the workload:

```yaml
preset: mednext-v1
workloads:
  - variant: base
    spatial: [128, 128, 128]
    checkpointing: auto
batch_search:
  memory_fraction: 0.90
  maximum: 16  # optional upper bound
```

```bash
mednext-accel profile workload.yaml
```

`checkpointing: auto` measures no checkpointing, all expansion branches, and
whole blocks. It does not select checkpointing for application code; the user
still chooses checkpoint style and batch size. The profiler validates numerical
results, records latency and peak allocation at the selected maximum, and
retains measured launch parameters. Generated output is a normal factory input.

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
