# MedNeXt-Accel package design

MedNeXt-Accel is an architecture-only PyTorch package. It provides official and
MONAI-compatible MedNeXt v1 models, checkpoint conversion, explicit activation
checkpointing, portable evaluation export, and profile-driven CUDA execution.
Datasets, trainers, losses, preprocessing, and nnU-Net workflows stay outside
the package.

Public factories build Small, Base, Medium, and Large. They default to
`optimization="auto"`; `optimization="reference"` selects native PyTorch, and a
JSON/YAML path or mapping supplies an external profile. `CheckpointConfig`
selects expansion-branch or whole-block recomputation and optional resolution
stages independently of optimization.

Optimized wrappers own the original `Parameter` objects. Module paths, parameter
shapes, tensor layouts, and state-dict keys therefore remain stable across
reference, bundled-profile, external-profile, eager, and compiled execution.
Official checkpoints are losslessly renamed. MONAI Base/Medium/Large use explicit
compatibility factories because their early down-block widths differ.

## Package boundaries

```text
src/mednext_accel/
  models/                 MedNeXt architecture and immutable configuration
  checkpoints/            format detection and lossless key conversion
  compat/                 explicit MONAI-compatible factories
  ops/                    adaptive wrappers and optional Triton kernels
  optimization/           descriptors, implementation registry, schema, resolver
  profiles/               bundled generic NVIDIA and exact-SM JSON profiles
  profiling/              campaign, batch search, isolated probes, synthesis, CLI
  export.py               evaluation trace and ONNX helpers
```

Profiles use stable operator, phase, and implementation identifiers. They never
refer to Python class paths. Resolution is deterministic and side-effect free:
external overrides and rules precede bundled exact-SM rules, then generic family
defaults. Approximate implementations require opt-in. Unknown architectures and
cross-SM user profiles warn once and continue.

The profiler is a separate explicit operation. `mednext-accel profile` detects
GPU and VRAM, probes full compiled steps to find one maximum feasible batch per
workload, benchmarks operator phases at that selected batch in isolated child
processes, validates numerical output, and writes one merged configuration.
Model construction and first forward never benchmark or write files.

Triton imports remain lazy. Importing the package, using CPU/reference execution,
loading checkpoints, and exporting evaluation graphs do not require Triton.
Evaluation resolves standard PyTorch branches so TorchScript trace,
`torch.export`, and ONNX graphs contain no custom MedNeXt operators.

Repository-only tests, benchmarks, tools, docs, raw profiles, traces, compiler
caches, and Nsight captures are excluded from the wheel. Generated artifacts are
also excluded from Git. MedNeXt v2 will use a separate model module; the registry
and profile schema already accept new families such as global response
normalization without changing the public profile format.

Release verification covers CPU forward/backward, state-dict identity, official
and MONAI conversion, checkpoint modes, direct `jit.trace`, strict
`torch.export`, optional ONNX Runtime, full-graph compilation, CUDA numerical
checks, profile schema/resolution, subprocess OOM behavior, wheel installation,
and command-line discovery without eager CUDA/Triton imports.
