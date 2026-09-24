# MedNeXt-Accel package design

MedNeXt-Accel is an architecture-only PyTorch package. It provides official and
MONAI-compatible MedNeXt v1 models, MedNeXt v2 Base and Wide models, checkpoint
conversion for v1, explicit activation checkpointing, portable evaluation
export, and policy-driven CUDA execution. Datasets, trainers, losses,
preprocessing, and nnU-Net workflows stay outside the package.

Public Small/Base/Medium/Large factories default to `optimization="auto"`;
`optimization="reference"` selects native PyTorch, and a policy-v2 YAML path or
mapping supplies an external overlay. JSON encoding of a v2 policy is also
accepted. Evidence JSON and obsolete schema-v1 runtime documents are rejected.
Checkpoint style and resolution stages use `CheckpointConfig`, independently of
optimization. A parsed YAML mapping can initialize it with
`CheckpointConfig(**mapping)`; factories currently require that object or `None`.

Optimized wrappers own the original `Parameter` objects. Module paths, parameter
shapes, tensor layouts, and state-dict keys remain stable across reference,
bundled/external policies, eager, and compiled execution. Official checkpoints
are losslessly renamed. MONAI Base/Medium/Large use explicit compatibility
factories because their early down-block widths differ.

## Package boundaries

```text
src/mednext_accel/
  models/                 MedNeXt architecture and immutable configuration
  checkpoints/            format detection and lossless key conversion
  compat/                 explicit MONAI-compatible factories
  ops/                    adaptive wrappers and optional Triton kernels
  optimization/           descriptors, implementation registry, policies, recipes
  policies/               compact shared NVIDIA and SM86/SM89/SM120 YAML
  profiling/              campaigns, isolated probes, immutable evidence, CLI
  export.py               evaluation trace and ONNX helpers
```

Policies use stable operator, phase, and implementation identifiers, not Python
class paths. Layers resolve external → execution-SM → shared NVIDIA → reference.
Missing rules/phases fall through; explicit reference tombstones terminate a
matched phase. Confidence is descriptive. Correctness guards enforce backend,
dtype, geometry, and export requirements; approximate implementations require
opt-in. An external policy targeting another SM warns and remains active.
The actual execution device selects the SM layer, including after model moves.
Regular depthwise dX and dW can independently use native/custom implementations.

The profiler is explicit. `mednext-accel profile` detects GPU/VRAM, searches one
maximum feasible batch per workload, benchmarks deduplicated operator phases
there, validates components, and compares the complete provisional overlay over
bundled policies. Kernel numerical evidence remains unchanged when the model
performance/memory gate rejects a policy. The CLI uses argparse and Rich.

Each run publishes immutable evidence JSON first, then atomically replaces a
stable runtime policy YAML referencing its exact SHA-256. Failed publication
can leave orphan evidence, never overwritten evidence behind an older policy.
Positive rules require aggregate acceptance; otherwise only measured negative
tombstones are published and evidence marks remaining bundled fallthrough as
untested by that comparison. `ProfilingResult` exposes policy, evidence,
environment, and both artifact paths. See [optimization](../optimization.md)
for the public API, schema, gates, and migration.

Triton imports remain lazy. CPU/reference execution, checkpoint loading, and
evaluation export do not require Triton. Evaluation uses standard PyTorch
branches so `jit.trace`, strict `torch.export`, and ONNX graphs contain no custom
MedNeXt operators. Neither construction nor first forward benchmarks or writes
files.

MedNeXt v2 is exposed through `mednext_v2_base` and `mednext_v2_wide`. Its GRN
operator has a native PyTorch reference path and an optional
`triton_fused_grn` training implementation. The fused implementation remains
evidence-gated: this release has no bundled performance rule selecting it until
a profiling campaign passes numerical validation and complete-model acceptance.
This states the dispatch status only; no MedNeXt v2 performance result is
claimed here.

Tests, benchmarks, tools, docs, traces, compiler caches, and Nsight captures are
outside the wheel. Generated local policy/evidence files are excluded from Git
and distribution archives; bundled YAML is shipped. The v1 and v2 model families
use separate modules while sharing the operator/phase registry and policy
machinery.

Release verification covers CPU forward/backward, state-dict identity, official
and MONAI conversion, checkpoint modes, direct `jit.trace`, strict
`torch.export`, optional ONNX Runtime, fullgraph compilation, CUDA numerical
checks, policy resolution, evidence/publication failures, archive content, wheel
installation, and CLI discovery without eager CUDA/Triton imports. GitHub CI is
CPU-only; CUDA checks run on local hardware.
