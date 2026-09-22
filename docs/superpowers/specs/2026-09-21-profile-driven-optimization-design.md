# Profile-Driven Optimization Architecture

> Historical design record. Runtime profile/schema examples and output instructions
> in this document are superseded by [policy/evidence v2](../specs/2026-09-22-policy-evidence-v2-design.md).
> Use the current [optimization guide](../../optimization.md) for runnable YAML,
> CLI commands, and Python APIs.

**Status:** Draft for final review

**Date:** 2026-09-21

## Purpose

MedNeXt-Accel provides optimized MedNeXt model architectures that work out of
the box across NVIDIA GPU generations, VRAM capacities, input shapes, batch
sizes, training, and inference. Users choose their model, batch size, and
checkpointing policy. The package chooses execution implementations without
requiring manual kernel tuning.

The existing optimization policy does not meet this goal. It is an exact table
for a small set of RTX 5090, BF16, `128^3`, and batch-size combinations. An
unknown GPU, dtype, spatial shape, or batch size can produce an empty selection
and silently use the reference implementation. Experimental validation has
therefore become product behavior instead of evidence used to build a general
policy.

This design replaces exact-table policy selection with a versioned,
profile-driven operator registry and deterministic resolver. RTX 5090 results
seed the first generic NVIDIA and SM 12.0 profiles. Additional SM profiles can
be added as data without changing model or resolver code.

## Goals

- Make optimized execution the default for all MedNeXt factories.
- Preserve reference model parameters, state-dict keys, checkpoint
  compatibility, and mathematical structure.
- Select implementations by operator, execution phase, hardware, dtype,
  model context, shape, batch size, checkpointing, and VRAM.
- Produce a useful decision for unmeasured batch sizes and input shapes rather
  than returning an empty selection.
- Use a generic RTX-5090-derived profile for NVIDIA SM architectures that do
  not yet have a bundled profile, with a visible warning.
- Keep profiling separate from model construction and normal execution.
- Let the profiling tool run a complete campaign and synthesize one usable
  profile without requiring the user to compare experiments manually.
- Let profiling campaigns expand their batch search to the available GPU VRAM.
- Support MedNeXt v2 and future models by registering new operator families and
  implementations rather than redesigning the profile format.
- Preserve eager execution, `torch.compile(fullgraph=True)`, `torch.jit.trace`,
  and ONNX export in their documented contexts.

## Non-goals

- The package does not choose the user's training batch size.
- The package does not automatically choose whether checkpointing is enabled.
- The package does not run benchmarks during model construction or first
  forward.
- The package does not promise that a cross-SM external profile is optimal.
- The package does not silently enable mathematically approximate operations.
- The package does not provide a training framework, optimizer, data pipeline,
  or experiment manager.

## Public model API

Every MedNeXt factory accepts a YAML-compatible `optimization` argument:

```python
model = mednext_base(
    in_channels=1,
    out_channels=3,
    optimization="auto",
)
```

`"auto"` is the default. It loads the bundled profile registry and resolves
implementations from the actual execution context.

```python
model = mednext_base(
    in_channels=1,
    out_channels=3,
    optimization="reference",
)
```

`"reference"` uses export-safe native PyTorch implementations. The former
`"torch"` policy name is removed because it is ambiguous and there are no
external compatibility obligations.

An external profile can be supplied as a path string or a mapping produced by
a YAML loader:

```python
model = mednext_base(..., optimization="profiles/l40s.json")

model = mednext_base(..., optimization=application_config["optimization"])
```

The public interface never requires a Python-only configuration object. The
loader may use internal immutable typed objects after validating a string or
mapping.

The current `optimize(model, policy=...)` interface, the `conservative` policy,
and runtime `autotune` policy are removed. Profiling becomes an explicit tool
and API described below.

## Resolution lifecycle

Factories construct adaptive operator wrappers while preserving the reference
parameter tensors and state-dict layout. Wrappers contain immutable resolution
rules and registered implementations. They do not benchmark or mutate the
model in `forward`.

At eager execution or `torch.compile` tracing, rules use static execution
properties:

- CUDA device and compute capability;
- total device VRAM;
- dtype;
- training, inference, or export context;
- batch size;
- input and output channels;
- spatial dimensions and volume;
- kernel size, stride, padding, dilation, groups, and convolution direction;
- MedNeXt family, variant, and architecture fingerprint;
- user-selected checkpoint policy.

`torch.compile` specializes these branches for the traced context and removes
unused branches from the compiled graph. Eager mode evaluates the small
deterministic rule set. Resolution contains no timing, random choice, disk
write, or mutable cache operation.

The model exposes an inspection API that uses the same resolver without
running a training step:

```python
report = model.explain_optimization(
    input_shape=(3, 1, 128, 128, 128),
    dtype="bfloat16",
    device="cuda:0",
)
```

The report lists every operator decision, selected implementation, matched
rule, profile source, extrapolation, semantic class, and warning.

## Operator and phase descriptors

Profiles describe primitive operations instead of Python module paths. A
descriptor consists of a stable operator family and properties relevant to
implementation selection. Initial operator families are:

- `depthwise_conv3d` with `regular`, `downsample`, and `transpose` directions;
- `pointwise_conv3d`;
- `group_norm`;
- `gelu`;
- export-safe reference operations.

The registry separates phases where implementations can be chosen
independently:

- `forward`;
- `backward_input`;
- `backward_weight`;
- `backward_bias`;
- combined `training` implementations where autograd behavior is integral;
- `inference`;
- `export`.

This separation represents the current implementation accurately. Regular
depthwise convolution can use native forward, custom Triton dX, split Triton
dW, and native dB. Transpose and downsample depthwise operations can select
their backward phases independently. Pointwise GEMM is a combined training
implementation. Export can resolve to a native implementation even when eager
training uses custom code.

Each registered implementation declares:

- a stable implementation identifier and implementation version;
- supported operator families and phases;
- dtype, layout, dimensionality, kernel, stride, and device constraints;
- configurable parameters and validation rules;
- its semantic equivalence class;
- its export and compilation capabilities.

## Semantic equivalence

Implementation metadata classifies numerical behavior as:

- `exact`: identical mathematical and supported numerical behavior;
- `numerical`: mathematically equivalent and validated within declared dtype
  tolerances;
- `approximate`: intentionally changes the mathematical approximation.

`auto` permits `exact` and validated `numerical` implementations. Approximate
GELU or future approximate operations require an explicit YAML-compatible
option such as `allow_approximate: true`. A performance profile cannot silently
change this setting.

## Profile document

Profiles use a versioned JSON-compatible schema. YAML files are accepted after
parsing to the same primitive mapping. A profile contains:

- schema version and profile identity;
- target vendor and optional SM architecture;
- provenance including collection time, driver, GPU identity, SM, VRAM,
  platform, Python, mednext-accel, PyTorch, CUDA, cuDNN, Triton, generator,
  objective, compile mode, execution counts, and the complete normalized
  campaign with every workload identity;
- implementation identifiers and retained launch parameters;
- operator-family defaults;
- ordered interval and formula rules;
- exact full-model overrides;
- raw measurement summaries;
- runtime recommendations such as compile mode;
- coverage and validation metadata.

`PyYAML>=6` becomes a core dependency so profile and campaign YAML files work
in the default installation. JSON remains the canonical serialized form for
bundled profiles and machine-generated output.

A depthwise rule can select phases independently and calculate split parameters:

```yaml
rules:
  - operator:
      family: depthwise_conv3d
      direction: regular
      kernel_size: [3, 3, 3]
      stride: [1, 1, 1]
      channels: 32
      spatial_volume: {min: 1800000, max: 2300000}
      batch: {min: 1}
    select:
      forward:
        implementation: reference
      backward_input:
        implementation: triton_depthwise_dx
        parameters: {block_size: 256}
      backward_weight:
        implementation: triton_split_dw
        parameters:
          splits:
            rule: scale_work
            anchor:
              batch: 1
              spatial: [128, 128, 128]
              value: 64
      backward_bias:
        implementation: reference
```

Candidate records may include latency and temporary-memory models. Full-model
overrides capture interactions that isolated operator benchmarks miss:

```yaml
overrides:
  - match:
      model_family: mednext_v1
      variant: base
      input_shape: [6, 1, 128, 128, 128]
      checkpointing: all-expansion
      total_vram_gib: {max: 40}
    select:
      pointwise_32_64_128: reference
      pointwise_64_32_128: reference
```

Checkpointing remains a user choice. It appears in match context because it
changes graph-level latency and memory behavior.

Compile recommendations are advisory and do not compile the model implicitly:

```yaml
runtime:
  recommended_compile:
    enabled: true
    fullgraph: true
    mode: max-autotune-no-cudagraphs
```

## Profile registry and precedence

The package initially bundles:

- `generic-nvidia`, derived from the RTX 5090 evidence and capable of resolving
  every supported NVIDIA context;
- `sm120`, containing SM 12.0 defaults and measured overrides.

Future `sm89`, `sm86`, `sm75`, and other profiles are data additions. They do
not require resolver changes.

Resolution precedence is:

1. external-profile exact override;
2. external-profile matching rule;
3. bundled exact-SM override or matching rule;
4. bundled generic operator-family rule;
5. correctness guard when no registered implementation can execute the
   operation safely.

An external profile records its source SM but is not blocked on a different
SM. The resolver emits one clear warning and applies it. An unknown device SM
uses `generic-nvidia` and emits one warning. Warnings are also included in the
inspection report.

A reference implementation selected by a measured rule is a deliberate
optimization decision. A correctness guard is reported separately and is not
described as an optimized selection.

## Generalization over batch and spatial shape

The resolver never requires an exact batch entry to produce a selection.

1. Exact full-model overrides have highest priority.
2. Interval rules apply inside their measured or synthesized range. If batch 2
   and batch 4 select the same pointwise implementation, a synthesized
   `batch: {min: 2, max: 4}` rule covers batch 3.
3. Outside an interval, the resolver evaluates the same operator signature's
   latency and memory formulas. Depthwise split values scale from anchors using
   batch multiplied by spatial volume, then clamp and round to legal kernel
   granularities.
4. A channel or spatial signature without a specialized rule uses the
   operator-family default from the selected profile.
5. A context outside an exact-SM profile uses the generic NVIDIA profile rather
   than an empty selection.

The profile records whether a decision is measured, interpolated,
extrapolated, or inherited from a family default. This status affects reporting
but does not disable the implementation.

## VRAM-aware backend selection

Users select batch size and checkpointing. The resolver uses total device VRAM
only to choose between backend implementations with different temporary-memory
costs.

Operator benchmarks provide parameterized temporary-memory measurements.
Whole-model benchmarks provide overrides for graph-level buffer lifetime,
checkpoint recomputation, fusion, and compiler scheduling effects. Rules can
therefore choose a lower-memory backend on a 32 GiB GPU and a faster,
higher-memory backend on a 96 GiB GPU with the same SM architecture.

Memory decisions use total physical VRAM and the profile's declared headroom,
not transient free memory at model construction. The inspection report exposes
the estimated memory cost and rule threshold. The resolver does not reduce
batch size or enable checkpointing on the user's behalf.

## Profiling user experience

The standard command requires no workload flags:

```bash
mednext-accel profile
```

It detects the local GPU and software environment, loads the package's standard
MedNeXt campaign, benchmarks candidates, performs whole-model validation,
synthesizes rules, and writes one merged profile to the standard user profile
directory.

Built-in campaign presets are available:

```bash
mednext-accel profile --preset mednext-v1
mednext-accel profile --preset mednext-v2
mednext-accel profile --preset all
```

The default preset covers all model families implemented by the installed
package. `mednext-v2` becomes available with the v2 reference model.

A targeted campaign accepts one configuration file:

```bash
mednext-accel profile workload.yaml
```

The smallest useful file is:

```yaml
preset: mednext-v1
workloads:
  - variant: base
    spatial: [128, 128, 128]
```

Defaults cover BF16, training and inference, automatic batch expansion, all
supported checkpoint contexts, a balanced objective, and automatic output
location. Advanced fields narrow or extend the campaign; they do not expose
individual kernel commands.

The Python API performs the same campaign:

```python
from mednext_accel.profiling import profile

result = profile()
result.save()
```

Raw measurements remain in or alongside the merged profile for review. The
tool selects implementations and synthesizes rules; users do not compare and
merge experiment files manually.

## Adaptive batch search

The final behavior is specified by
[Maximum-Only Batch Search](2026-09-22-maximum-batch-search-design.md), which
supersedes this design's earlier dense-sampling proposal. Search locates one
maximum feasible batch separately for each model, spatial shape, phase, dtype,
and checkpoint context.

When `maximum` is supplied, the profiler probes it first. A feasible maximum
finishes the search after that single model probe. If it is infeasible, batch
one establishes the lower boundary and binary search finds the largest batch
that fits. When `maximum` is omitted, powers of two establish an infeasible
upper boundary before binary refinement. Every probe runs in an isolated
subprocess so an OOM cannot corrupt later measurements.

Search probes are diagnostic evidence only. Kernel profiling and whole-model
validation run only at the selected maximum. To collect measured evidence for
another batch, the user reruns the campaign with that batch as `maximum`.

The default usable-memory fraction is 0.90. It is configurable in a campaign:

```yaml
batch_search:
  memory_fraction: 0.90
  maximum: 16  # optional upper bound
```

The maximum validated batch is provenance, not a runtime limit. A 32 GiB
profile may extrapolate its rules on a 96 GiB device. Reprofiling on the larger
device extends and validates the higher-batch regions.

## Profiling pipeline

Each campaign has three stages:

1. Construct requested reference models and enumerate stable operator
   descriptors and execution contexts.
2. Validate every compatible implementation numerically, then measure latency
   and temporary memory for unique descriptors.
3. Benchmark complete model steps for candidate combinations and checkpoint
   contexts to catch compilation, scheduling, buffer-lifetime, and VRAM effects
   not visible in isolated kernels.

The synthesizer rejects candidates that fail correctness or the campaign's
memory objective. It selects implementations automatically and produces
intervals, formulas, exact overrides, confidence classification, and complete
provenance.

Campaign objectives include throughput, memory, and balanced modes. They use
documented defaults. Users can override objective constraints in the campaign
file, but no manual result selection is required.

## MedNeXt v2 and future architectures

Profiles are operator-based and reusable across model families. MedNeXt v2 can
reuse depthwise, pointwise, activation, compilation, and export rules wherever
its descriptors match.

V2 adds a `global_response_norm` operator family and registered reference and
fused implementations. Forward and backward GRN implementations are separate
phases unless one combined implementation requires an integral autograd path.
No profile schema or resolver redesign is required.

Full-model overrides use a model-family identifier, variant, and architecture
fingerprint. This prevents a v1 graph-specific memory override from being
applied to a v2 graph while allowing primitive operator rules to remain shared.
The same mechanism supports future 3D segmentation models that reuse registered
operator families.

## Export behavior

Export is a distinct execution context. `torch.jit.trace` and ONNX resolution
select implementations declared export-safe. Current custom training kernels
can therefore use native forward during export without altering training
selection.

Export mode must be explicit in the export helper so tracing never captures a
device-profile warning, disk load, mutable resolver action, or custom backward
operation. Exported model parameters and numerical behavior follow the
reference model unless the user explicitly permits an approximate semantic
class.

## Validation and release requirements

CPU CI validates:

- profile schema parsing, validation, and version errors;
- registry uniqueness and implementation parameter validation;
- precedence across external, exact-SM, and generic profiles;
- batch 3 interpolation and unseen spatial-shape family defaults;
- unknown-SM warning and nonempty generic resolution;
- cross-SM external-profile warning without rejection;
- semantic-class enforcement;
- state-dict key and tensor-layout identity;
- `reference` behavior;
- deterministic inspection reports;
- export-safe resolution, TorchScript tracing, and ONNX export.

CUDA tests validate each implementation's forward and backward numerical
tolerances on small representative tensors. They are optional in GPU-less
GitHub CI and required when producing a bundled profile.

A bundled SM profile requires:

- complete recorded environment, campaign, and workload provenance;
- successful schema and registry validation;
- numerical validation for every selected implementation;
- operator-level latency and memory measurements;
- whole-model validation for published contexts;
- resolution coverage showing that supported contexts do not produce empty
  selections;
- retained raw summaries sufficient to audit synthesized rules.

GPU performance numbers are evidence files, not unit-test constants. Release
tests verify the profile's structure and decisions; reproducible profiler runs
establish performance.

## Package structure

The optimization subsystem is divided by responsibility:

```text
src/mednext_accel/
  optimization/
    descriptors.py       Stable operator and execution-context descriptors
    implementations.py   Implementation metadata and registry
    schema.py             Versioned profile parsing and validation
    profiles.py           Bundled and external profile loading
    resolver.py           Deterministic precedence and rule evaluation
    report.py             Explainable decisions and warnings
  profiles/
    generic-nvidia.json
    sm120.json
  profiling/
    campaign.py           Campaign configuration and preset expansion
    discovery.py          Model operator/context enumeration
    benchmark.py          Isolated subprocess measurement
    whole_model.py        End-to-end validation
    synthesize.py         Rules, formulas, and override generation
    cli.py                `mednext-accel profile`
```

Adaptive model operators remain under `ops/` and consume resolved
implementation identifiers through the registry. Model factories accept and
load the public optimization argument.

## Delivery sequence

Implementation proceeds in independently testable slices:

1. Introduce descriptors, implementation registry, schema, resolver, and a
   hand-authored generic/SM120 profile that reproduces current measured choices
   while generalizing unknown batches.
2. Integrate adaptive operators and the factory `optimization` API; remove the
   old policy API; preserve state dicts, compile, and export.
3. Add the inspection report and VRAM-aware full-model overrides.
4. Add campaign configuration, zero-argument CLI, adaptive batch search,
   subprocess isolation, and raw measurement output.
5. Add automatic rule synthesis and whole-model selection.
6. Generate the bundled SM120 profile through the new profiler and replace the
   hand-authored bootstrap data with reproducible generated data.
7. Add further SM profiles as hardware measurements become available.

Each slice leaves a working package. The package does not wait for every NVIDIA
architecture profile before adopting generic optimized behavior.
