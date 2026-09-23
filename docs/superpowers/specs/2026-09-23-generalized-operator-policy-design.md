# Generalized Operator Policy Design

## Goal

Replace exact model-context lookup behavior with compact operator rules that
remain useful for unmeasured batch sizes, channel counts, and spatial sizes,
while retaining explicit capability guards, negative evidence, and full-model
release validation.

The first supported generalized domain is NVIDIA CUDA training with BF16,
three-dimensional contiguous NCDHW tensors, kernel size 3 depthwise operators,
and kernel size 1 regular pointwise operators. Other dtypes, kernel sizes,
layouts, and two-dimensional execution remain native and are reported clearly.

## Scope and non-goals

This change does not alter MedNeXt parameters, state-dict keys, checkpointing,
export paths, or the mathematical definitions of existing kernels. It does not
add a learned runtime predictor or a dependency on a machine-learning package.
It does not claim that a policy measured on one SM is optimal on another SM.

Two-dimensional MedNeXt remains supported through native PyTorch. Its
optimization report must work and describe those operators as unwrapped.

## Runtime resolution

Policy precedence remains external policy, exact-SM bundled policy, shared
NVIDIA policy, then built-in reference. Within a policy, the first matching
rule for a phase remains authoritative.

A matching custom selection that fails a hard implementation guard is not a
tombstone. The resolver records it as the highest-priority diagnostic and
continues with lower-priority policy layers. An explicit `reference` selection
is a tombstone and stops resolution. If every matching custom selection is
guarded, the highest-priority guard decision is returned.

Reports distinguish three dispositions without replacing existing report
types:

- `custom`: a wrapped operator resolves to an accelerated implementation;
- `native`: a wrapped operator resolves to PyTorch, including guard fallback;
- `unwrapped`: an operator is outside the adaptive wrapper domain.

Reports enumerate native convolution modules as well as adaptive wrappers.
This makes output heads, residual projections, two-dimensional convolutions,
and other deliberately unsupported operators visible.

## Policy conditions

Depthwise generalized rules use existing `work` and `reduction_work` ranges:

```
work = batch * spatial_volume * in_channels
reduction_work = batch * spatial_volume
```

Pointwise rules may additionally match numeric `in_channels` and
`out_channels` ranges. Exact `channels: [in, out]` remains valid for measured
exceptions and tombstones. Model family, model variant, and checkpoint policy
remain accepted when reading policy v2 for compatibility, but generated and
bundled operator rules do not use them.

Positive generalized rules are bounded by observations or reviewed
cross-device hypotheses. A single observation cannot create an unbounded
positive range. Numerical failures and complete measured losers create exact
reference tombstones before broader positive rules. OOM, timeout, and
infrastructure failures remain gaps rather than negative performance evidence.

## Profiling correctness

Candidate identity and comparison identity are separate. Evidence identity
includes implementation and launch parameters. Comparison identity excludes
those fields, so competing implementations receive the same random tensors.

For each comparison context, invalid or losing candidates are rejected
individually. They do not veto another valid winner. The synthesizer selects
the fastest qualifying candidate for throughput and balanced objectives, or
the lowest-memory qualifying candidate for the memory objective. Stable
implementation and parameter ordering breaks exact ties.

Campaign workload mappings reject unknown fields. This prevents an apparently
successful run from silently ignoring fields such as `kernel_size` or
`base_channels`.

## Integrated operator evidence

Raw kernel timings are retained as diagnostics, but dispatch evidence uses the
actual adaptive operator path. The integrated benchmark includes the wrapper,
policy resolution, custom autograd, autocast casts, requested dX/dW/dB masks,
and the campaign compile mode. Native PyTorch is measured through the same
module-level interface.

The initial integrated matrix covers current regular depthwise dX/dW,
downsample dX, transpose dW, and pointwise training implementations in the
validated domain. Each comparison checks forward output and every requested
gradient before timing. Backward implementations must honor
`ctx.needs_input_grad`; an unused gradient cannot be computed merely because a
custom phase exists.

Full-model comparisons remain a release sentinel and a publication gate for a
new bundled SM policy. They are not required for every operator grid point.
This preserves protection against graph-level counterexamples such as the
previous downsample-dX and RTX A6000 aggregate regressions.

## Generalization and artifacts

Generated evidence remains immutable JSON. Runtime configuration remains
compact YAML. The profiler collapses duplicate observations projected from
different model variants or checkpoint policies before synthesizing operator
rules.

Depthwise observations are segmented along the appropriate work metric. A
known negative observation splits a positive interval. Pointwise observations
are grouped into reviewed input/output-channel and spatial-work regions; exact
negative exceptions precede them. The runtime does not require that a complete
batch/channel/spatial tuple was previously measured.

Bundled SM86, SM89, and SM120 policies retain architecture-specific launch
recipes and exceptions. Shared NVIDIA rules remain the conservative default
for unknown SMs. Loading an external policy for another SM remains allowed
with a warning.

## Compatibility

Official and MONAI checkpoint import remains unchanged because adaptive
wrappers retain the original parameter objects and keys. TorchScript, ONNX,
and `torch.export` continue through native execution. Existing policy-v2 files
remain readable; newly generated policies omit model/checkpoint match keys.

## Validation

CPU tests cover parsing, resolution, reporting, synthesis, and artifact
round-trips. CUDA tests cover eager and `torch.compile(fullgraph=True)`
integrated operators with BF16 autocast and gradient masks. The full suite,
Ruff, package build, Twine validation, and installed-wheel smoke test run before
merge. RTX 5090 full-model comparisons validate the resulting SM120 policy;
SM86 and SM89 defaults remain bounded by their existing evidence until new
integrated evidence is collected on those devices.
