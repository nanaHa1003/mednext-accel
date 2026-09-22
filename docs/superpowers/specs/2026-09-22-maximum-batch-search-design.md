# Maximum-Only Batch Search

## Purpose

The profiler currently mixes two responsibilities: finding the largest batch
that fits the available VRAM and deciding which smaller batches deserve kernel
measurements. `dense_until` causes every small integer batch to be measured,
and supplying `maximum` currently causes every integer through that limit to be
measured. Those probes are then all promoted into kernel experiments even
though most existed only to locate a memory boundary.

Batch search will answer one question only: what is the largest feasible batch
at or below the user's optional limit? Kernel profiling and whole-model
validation will run only for that selected batch. Users who want evidence for a
different batch rerun the campaign with that value as `maximum`.

This design supersedes the dense-search behavior in the original
profile-driven optimization design. Backward compatibility for the unreleased
`strategy` and `dense_until` configuration fields is intentionally not kept.

## Configuration

The YAML surface becomes:

```yaml
batch_search:
  memory_fraction: 0.90
  maximum: 16
```

`memory_fraction` remains in `(0, 1]`. A model step that succeeds but allocates
more than this fraction of total device memory is treated as infeasible.

`maximum` is an optional positive integer and means the highest batch the user
is willing to use. It is an upper bound, not a request to enumerate every
smaller batch. When omitted, the profiler discovers an upper bound
automatically.

`strategy`, `dense_until`, and any other unknown `batch_search` keys are
rejected with a configuration error. The Python `BatchSearch` type removes
those fields rather than retaining deprecated aliases.

## Search algorithm

Feasibility is assumed to be monotonic: after a batch is infeasible, larger
batches are not expected to become feasible. This is a practical VRAM-capacity
assumption. Every probe remains isolated in the existing subprocess boundary,
so an OOM or failed compile is an infeasible observation rather than a parent
process failure.

### User supplies `maximum`

1. Probe `maximum` first.
2. If it is feasible within `memory_fraction`, return it immediately.
3. If it is infeasible and greater than one, probe batch one.
4. If batch one is infeasible, return zero.
5. Otherwise binary-search the open interval between the known-feasible lower
   bound and known-infeasible `maximum` until they are adjacent.
6. Return the feasible lower bound.

This makes the common case—"can I train with batch N?"—one model probe.

### User omits `maximum`

1. Probe batch one; return zero if it is infeasible.
2. Probe powers of two (`2, 4, 8, ...`) until the first infeasible batch.
3. Binary-search between the last feasible power of two and the first
   infeasible power of two until they are adjacent.
4. Return the feasible lower bound.

All results are cached within the search so a batch is never probed twice.
`BatchSearchResult.probes` remains available as sorted diagnostic evidence,
but only `maximum_feasible` is a profiling target.

## Campaign execution

Batch search remains specific to the complete workload context: model variant,
input shape, channel counts, checkpoint policy, compile mode, dtype, and the
current GPU environment. Exact duplicate workloads continue to share one
search.

For each feasible workload, the execution planner uses:

```python
batches = (search_result.maximum_feasible,)
```

When `maximum_feasible == 0`, the workload contributes no kernel cases or
whole-model validation. Intermediate feasible probes from exponential or binary
search remain diagnostic results and never enter the kernel matrix, generated
profile, or validation plan.

The chosen batch receives the existing globally deduplicated kernel benchmark
and context-specific whole-model validation. Generated schema-v1 rules therefore
have an exact batch interval where `min == max`. A generated profile provides
measured guarantees only for the selected batch. Using another batch falls back
according to the profile's normal resolver behavior; users can generate new
evidence by rerunning with another `maximum`.

## Reporting

Adaptive search retains unknown-total progress:

```text
Batch search probes 3/? · 24.1 s/probe
```

No ETA is invented. When the search completes, each workload reports exactly
one outcome:

```text
maximum feasible batch: 13
```

or:

```text
no feasible batch
```

The subsequent static and exact execution plans count kernel work only for the
selected batch. Search probes are never described as kernel experiments.

## Memory interpretation

The current training probe performs a compiled BF16 forward/backward step with
AdamW and includes parameters, gradients, activations, and lazily created
optimizer state in peak allocated memory. The selected maximum is therefore a
recommendation for that probe, not a universal promise for arbitrary training
pipelines. Different optimizers, augmentations, retained tensors, and external
framework state may require a smaller runtime batch. Users can lower the batch
directly or rerun with a smaller `maximum`.

## Public interfaces

The resulting Python records are:

```python
@dataclass(frozen=True, slots=True)
class BatchSearch:
    memory_fraction: float = 0.90
    maximum: int | None = None


@dataclass(frozen=True, slots=True)
class BatchSearchResult:
    probes: tuple[ProbeResult, ...]
    maximum_feasible: int
```

`search_batches()` no longer accepts `transition_batches`. Campaign loading,
the runner, examples, and tests use only the two supported configuration
fields.

## Failure behavior

- An invalid configuration fails during YAML loading, before CUDA discovery or
  any profiling subprocess.
- A failed or over-budget requested `maximum` triggers boundary search rather
  than aborting the campaign.
- An infeasible batch one produces `maximum_feasible == 0` and no kernel work.
- A probe returning a result for another batch remains a fatal programming
  error.
- Unexpected parent-process exceptions retain their current fatal behavior.

## Verification

CPU tests will prove:

- a feasible supplied `maximum` requires one probe;
- an infeasible supplied `maximum` uses binary search and returns the exact
  boundary;
- an unlimited search uses exponential growth followed by binary search;
- over-budget successful steps are treated as infeasible;
- batch one failure produces zero;
- intermediate feasible probes never become execution-plan batches;
- separate workload contexts select their own single maxima;
- old and unknown YAML keys fail before profiling side effects;
- progress reports every completed probe and the final selected maximum;
- generated rules match only the selected batch.

The complete CPU suite, Ruff checks, package build, Twine validation, and one
bounded RTX 5090 campaign remain required before integration.
