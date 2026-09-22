# Profiler Deduplication and Execution Planning

> Historical design record. Runtime profile/schema examples and output instructions
> in this document are superseded by [policy/evidence v2](../specs/2026-09-22-policy-evidence-v2-design.md).
> Use the current [optimization guide](../../optimization.md) for runnable YAML,
> CLI commands, and Python APIs.

## Purpose

Before grouped execution, `mednext-accel profile` launched one subprocess for
every isolated operator shape and repeated those measurements for each model
variant and checkpoint context. MedNeXt Base at 128³ needs about 48 isolated
kernel comparisons plus one compiled whole-model validation at its selected
batch. Selecting all three checkpoint contexts could approach three times that
work even though checkpointing does not change an isolated kernel invocation.

The profiler will separate context-free kernel evidence from context-dependent
whole-model validation. It will benchmark every unique kernel case once, group
related cases into a small number of subprocesses, and validate only the
maximum feasible batch selected for each workload. Diagnostic batch-search
probes never enter the kernel matrix or validation plan. The generated
optimization-profile schema and model construction API remain compatible.

## Scope

This change covers:

- campaign-wide discovery and batch search before kernel benchmarking;
- deduplication across model variants, output classes, and checkpoint contexts;
- grouped kernel subprocesses with failure isolation;
- selected-batch compiled whole-model validation;
- an experiment plan, completed counts, average durations, and ETA;
- context-specific materialization of shared kernel measurements for the
  existing profile synthesizer.

Persistent benchmark caches, interrupted-campaign resume, Typer migration, new
model families, and new dtype support are separate work. Existing BF16 training
coverage is preserved.

## Execution phases

### 1. Environment and workload discovery

The profiler collects environment provenance once and performs a meta-tensor
operator discovery pass for every workload. Discovery produces pointwise and
depthwise cases without allocating model activations.

Before running CUDA work, the CLI reports:

```text
Static plan
  workloads: 1
  unique pointwise shapes: 30
  unique depthwise phase/shape pairs: 18
  kernel groups per selected batch: 5
```

This is a static shape count rather than a promised total because the selected
batch is not known yet.

### 2. Context-specific batch search

Batch search remains separate for each unique model workload because model
variant, spatial shape, checkpointing, channels, and compile mode affect memory
and step time. Exact duplicate workload definitions are searched once.

The search is adaptive, so its final probe count is unknowable at start. Each
completed probe increments the visible count:

```text
Batch search 3/? · batch=4 feasible · peak=37.2 GiB
avg 94.3 s/probe · elapsed 00:04:43
```

The display must never remain at `0/?` after a probe finishes and must not show
a fabricated remaining time while the search space is still open.

### 3. Global kernel-case deduplication

After all batch searches finish, the profiler constructs one global kernel
matrix. A kernel case is identified by fields that can affect its execution:

```text
family
direction
phase
batch
spatial_shape
in_channels
out_channels
kernel_size
dtype
candidate implementation
candidate launch parameters
```

Model variant, checkpoint context, output-class count, and module path are not
part of this key. They do not alter an isolated invocation once the tensor and
operator descriptors above are fixed.

Each workload retains references from its discovered operators to the shared
kernel cases it needs. Results are measured once and subsequently projected
back into every referring workload.

### 4. Grouped subprocess execution

For each batch size, cases are grouped into at most five subprocess categories:

1. pointwise training;
2. regular depthwise backward-input;
3. regular depthwise backward-weight;
4. downsample depthwise backward-input;
5. transpose depthwise backward-weight.

A child process runs its cases sequentially and returns a result keyed by a
stable case identifier. It releases tensor references and calls the CUDA cache
cleanup between cases. A numerical mismatch is a normal per-case result and
does not stop the group.

Subprocess isolation remains the failure boundary. If a grouped child times
out, crashes, or cannot recover from OOM, the parent bisects the group and
retries both halves. Recursion stops at a single case, which is recorded as a
failed measurement. Healthy workloads therefore pay for five subprocesses at
their selected batch; problematic cases retain the same isolation.

Once batch search supplies one maximum feasible batch per workload, the CLI
reports both logical experiments and physical subprocess groups:

```text
kernel plan: 5 planned groups, 48 unique experiments, 48 requested references
```

Progress includes group and experiment counts. Average time and ETA use
completed groups, while the experiment count shows coverage:

```text
Kernel groups 3/5 · experiments 31/48
avg 18.4 s/group · elapsed 00:00:55 · ETA 00:00:37
```

### 5. Selected-batch validation

Shared isolated results are converted into a complete decision signature for
each workload at its selected maximum. The signature contains every selected
implementation and its launch parameters for the workload's discovered
operators.

Each feasible workload contributes exactly one whole-model validation endpoint
at that selected batch. Intermediate feasible search probes are diagnostics;
they do not create kernel cases, decision segments, or validation endpoints.
An infeasible workload contributes none.

Validation remains specific to model variant, checkpoint context, spatial
shape, channel configuration, and compile mode. If the selected endpoint passes
the campaign objective, its candidate measurements remain eligible. If it
fails, every candidate measurement for that workload and selected batch is
marked invalid and the generated profile uses reference implementations there.

Whole-model validation has its own progress and average duration because its
compiled steps are much slower than kernel groups. Mixing both populations into
one average would produce unstable ETAs.

## Measurement materialization and profile compatibility

The in-memory kernel result is context-free. Before synthesis, the runner
materializes an existing `Measurement` record for each workload reference,
adding that workload's checkpoint name. This may repeat small measurement
records in the generated JSON, but it does not repeat GPU work and preserves the
current schema, rule matching, and downstream loaders.

The profile provenance adds execution statistics:

```json
{
  "kernel_case_count": 192,
  "kernel_group_count": 20,
  "whole_model_validation_count": 2,
  "deduplicated_reference_count": 384
}
```

Existing profiles remain loadable. Models do not need changes to consume a new
profile.

## Failure behavior

- A workload that cannot run batch one records the failed search and contributes
  no kernel cases.
- An individual invalid or failed kernel case selects the reference
  implementation for all workload references to that exact case.
- A grouped child infrastructure failure triggers recursive bisection before a
  case is declared failed.
- A failed whole-model endpoint invalidates that workload's selected-batch
  candidate measurements.
- An unexpected parent-process exception remains fatal; the CLI closes its Rich
  display and reports the exception normally.
- `--progress quiet` emits no progress diagnostics. Final CLI success output and
  fatal errors remain visible.

## Public and internal interfaces

The public commands remain:

```bash
mednext-accel profile [campaign.yaml] [--progress auto|plain|quiet]
```

No new required campaign fields are introduced. Internally, the runner gains
immutable records for discovered workloads, kernel-case keys, grouped requests,
raw kernel results, and selected-batch validation requests. The child protocol
gains a batched kernel request while retaining the existing single model-probe
request.

Progress events distinguish adaptive search probes, kernel groups, logical
experiments, and whole-model validations. Reporters calculate averages only
within the corresponding stage.

## Verification

CPU tests will cover:

- identical cases across checkpoint contexts and variants deduplicate to one
  kernel case;
- distinct batch, shape, phase, dtype, or launch parameters remain distinct;
- group construction creates the five expected categories;
- group failure recursively bisects and preserves successful results;
- every feasible workload contributes at most one selected batch;
- diagnostic search probes do not enter kernel or validation plans;
- the selected endpoint is validated exactly once;
- one failed endpoint invalidates that workload's selected-batch candidates;
- context-specific measurements retain checkpoint matching;
- batch progress advances with an unknown total and reports average time;
- post-search stages expose exact totals and ETA;
- old profile documents remain loadable.

Small CUDA probes will verify grouped pointwise and depthwise execution returns
the same validity and timing fields as the existing single-case child protocol.
The complete CPU suite, Ruff lint and formatting, distribution build, and Twine
metadata checks remain required before integration.
