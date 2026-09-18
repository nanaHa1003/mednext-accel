# MedNeXt-Accel Package Design

**Status:** Approved for implementation

**Date:** 2026-09-18

**Distribution:** `mednext-accel`
**Import namespace:** `mednext_accel`

## Purpose and scope

MedNeXt-Accel is a production-oriented PyTorch package for MedNeXt model
architectures. It provides a clean reference implementation, lossless import of
compatible historical checkpoints, explicit memory policies, and optional
hardware-specific acceleration. It does not provide datasets, trainers, losses,
optimizers, preprocessing, postprocessing, or an nnU-Net workflow.

The first release supports MedNeXt v1 Small, Base, Medium, and Large. MedNeXt v2
will be a separate model module after a reference implementation is validated;
v1 and v2 do not share a public model class merely to reduce duplication.

## Public construction API

The common path is a normal PyTorch factory:

```python
from mednext_accel import CheckpointConfig, mednext_base

model = mednext_base(
    in_channels=1,
    out_channels=3,
    spatial_dims=3,
    kernel_size=3,
    deep_supervision=True,
    checkpointing=CheckpointConfig(style="expansion", stages=(0, 1)),
)
```

`mednext_small`, `mednext_base`, `mednext_medium`, and `mednext_large` construct
the official MedNeXt v1 architecture. `MedNeXtV1` and `MedNeXtV1Config` remain
available for advanced construction. The factories default to the reference
PyTorch backend and perform no device work, benchmarking, compilation, or cache
writes.

`CheckpointConfig` combines a recomputation boundary with resolution selection.
`style="expansion"` checkpoints only the expanded branch; `style="block"`
checkpoints complete MedNeXt and resampling blocks. `checkpointing=None` disables
checkpointing. Stage zero is the highest spatial resolution and stage four is the
bottleneck. A tuple selects specific stages; `stages=None` selects all stages.
Checkpointing runs only while training with gradients enabled.

Deep supervision is enabled independently of checkpointing. Training output can
use `tuple`, `list`, or `stacked` form; `tuple` is the default and matches MONAI's
MedNeXt behavior. Evaluation always returns the primary logits tensor so that the
inference contract remains stable for tracing and export.

## Architecture and parameter ownership

The reference architecture owns ordinary PyTorch parameters and uses standard
`nn.Conv2d`/`nn.Conv3d`, `nn.ConvTranspose2d`/`nn.ConvTranspose3d`, `nn.GroupNorm`,
and GELU operations. Optimized modules retain the same parameter objects, shapes,
and state-dict paths. Optimization changes execution policy, never the model's
mathematical structure or checkpoint schema.

Model configuration is immutable and records spatial dimensions, channels,
kernel size, block counts, per-stage expansion ratios, downsample expansion
ratios, residual behavior, and compatibility family. The official and MONAI
compatibility families are explicit configurations because they are not identical
for Base, Medium, and Large.

## Package boundaries

```text
src/mednext_accel/
  __init__.py                 Stable top-level factories and configuration types
  checkpointing.py           CheckpointConfig and checkpoint execution helper
  models/
    __init__.py               Model exports
    config.py                 Immutable v1 variant definitions
    blocks.py                 MedNeXt v1 blocks
    mednext_v1.py             Network and public factories
  checkpoints/
    __init__.py               Stable checkpoint API
    api.py                    Loading, format detection, and reports
    official_v1.py            Official v1 key conversion
    monai.py                  MONAI key conversion and compatibility checks
  compat/
    __init__.py
    monai.py                  Explicit MONAI-compatible factories
  optimization/
    __init__.py
    api.py                    optimize() and backend context
    policy.py                 Static and autotuned per-shape policies
    report.py                 Immutable OptimizationReport
  ops/
    pointwise.py              Reference/GEMM pointwise module
    depthwise.py              Dispatching depthwise modules
    _triton/
      depthwise.py            Optional Triton kernels and autograd registrations
  export.py                   Optional validated trace/export conveniences
```

Triton imports are lazy. Importing `mednext_accel`, constructing a model, loading
weights, running on CPU, tracing, and ONNX export must work without Triton.

Repository-only directories are:

```text
tests/                        Unit, compatibility, export, and CUDA tests
benchmarks/                   Reproducible kernel and whole-model benchmarks
tools/                        Profile collection and result summarization
docs/                         User and maintainer documentation
```

Generated profiles, Nsight captures, autotune caches, compiled artifacts, virtual
environments, archives, and benchmark outputs are ignored and never included in
the wheel or source distribution.

## Checkpoint formats and compatibility

The native package has a versioned state-dict schema. Public loading uses:

```python
from mednext_accel.checkpoints import load_checkpoint

report = load_checkpoint(
    model,
    "model_final_checkpoint.model",
    source="auto",
    strict=True,
)
```

`source` accepts `native`, `official-v1`, `monai`, or `auto`. Auto detection uses
key patterns and fails on ambiguity. The loader accepts a path, a raw state dict,
or a mapping containing a state dict. Path loading uses safe weight-only loading
where supported. Prefixes introduced by wrappers such as `module.` are normalized
before format detection.

`CheckpointLoadReport` records the detected format, renamed keys, ignored
compatibility-only keys, missing keys, unexpected keys, shape mismatches, and
whether conversion was lossless. Strict loading never silently truncates, pads,
or initializes incompatible tensors.

Official MedNeXt v1 S/B/M/L checkpoints with kernel sizes 3 or 5 are losslessly
importable. The obsolete `dummy_tensor` checkpointing parameter is consumed and
reported but is not stored as a native model parameter. Deep-supervision head
weights are imported when the target model contains those heads.

MONAI Small is structurally compatible after key renaming. MONAI Base, Medium,
and Large use different expansion ratios in early downsampling blocks, so they
must be loaded into models constructed by `mednext_accel.compat.monai`. Loading
them into the official-compatible factories fails with a precise shape mismatch.
The compatibility namespace has no runtime dependency on MONAI.

## Optimization API

Optimization is explicit and occurs after model construction and device transfer,
before `torch.compile` or distributed wrapping:

```python
from mednext_accel.optimization import optimize

report = optimize(
    model,
    input_shape=(1, 1, 128, 128, 128),
    dtype=torch.bfloat16,
    policy="autotune",
)
```

`policy` accepts `torch`, `conservative`, or `autotune`. `torch` restores the
reference path. `conservative` applies checked static choices. `autotune`
benchmarks eligible implementations per unique shape and records its choices in a
versioned cache keyed by GPU identity, compute capability, dtype, shape, PyTorch,
Triton, CUDA, and package versions.

`optimize()` mutates execution policy in place and returns an
`OptimizationReport`; it does not return a copied model. Unsupported devices,
dtypes, shapes, or software versions use the reference implementation. The report
lists every selection and fallback.

The initial conservative policy contains only already-validated choices:

- pointwise GEMM only for shapes where the policy selects it;
- split depthwise weight-gradient for eligible odd cubic kernels;
- validated transpose depthwise weight-gradient configurations;
- stride-2 depthwise input-gradient only when selected;
- native PyTorch forward convolution by default.

## Inference export contract

The following are release requirements for pure evaluation mode:

- direct `torch.jit.trace(model.eval(), example, check_trace=True)`;
- TorchScript save and load;
- `torch.export.export(model.eval(), (example,), strict=True)`;
- `torch.onnx.export(model.eval(), (example,), dynamo=True)`;
- ONNX validation and ONNX Runtime execution.

The exported inference graph contains no `mednext_accel::*` custom operator.
Backward-only Triton paths automatically bypass custom autograd operators when the
model is in evaluation mode or gradients are disabled. Pointwise GEMM, when
selected, is expressed through traceable standard tensor operations.

The first release supports static spatial shapes and dynamic batch where the
exporter can preserve it. Fully dynamic spatial export is outside the first
release because skip additions require compatible dimensions through four
downsample/upsample levels. Exported models always return primary logits.

`torch.jit.script` is not supported because TorchScript scripting is deprecated
and conflicts with modern checkpoint/custom-op code. Training graph export and
portable ONNX representation of Triton kernels are also outside scope.

## Packaging and dependencies

The repository uses a `src` layout and `pyproject.toml`. PyTorch is a required
dependency with a documented minimum compatible version but no exact PyTorch or
CUDA pin. CUDA is never declared as a Python dependency. Triton, ONNX tooling,
MONAI compatibility tests, documentation, and development tools are optional
extras.

The wheel contains only `mednext_accel`. Benchmark and profiling programs remain
repository tools. The project uses the Apache-2.0 license and preserves derivative
notices for official MedNeXt and MONAI-derived compatibility information.

## Verification and release gates

CPU CI verifies construction, forward/backward correctness, checkpoint mapping,
state-dict stability, TorchScript trace/save/load, `torch.export`, ONNX export,
ONNX checker, ONNX Runtime output, package build, and wheel installation.

CUDA verification covers custom-op registration, reference equivalence, gradients,
mixed precision, `torch.compile(fullgraph=True)`, conservative policy selection,
autotune cache behavior, and whole-model benchmarks. Kernel numerical tests use
both representative tolerances and higher-precision references where appropriate.

Release validation covers output classes 3 and 8, kernel sizes 3 and 5,
deep-supervision modes, exact and approximate evaluation GELU, official and MONAI
compatibility configurations, and a 128-cubed MedNeXt Base workload.
