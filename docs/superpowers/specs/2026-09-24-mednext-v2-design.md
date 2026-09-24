# MedNeXt v2 Reference and Fused GRN Design

**Date:** 2026-09-24  
**Status:** Approved direction; implementation specification  
**Scope:** Model architecture, reusable optimization paths, fused GRN, and
evaluation export. Training pipelines, losses, datasets, and pretrained weights
are outside this project.

## Objective

Add a production-quality, paper-faithful MedNeXt v2 model family to
`mednext-accel`. The implementation must stand on its own as a documented
reference design, preserve the existing v1 API and checkpointing behavior, use the
current convolution optimizations where their operator descriptors match, and
ship a fused CUDA GRN implementation with safe reference fallbacks.

The implementation is derived from the MedNeXt-v2 paper and the published
MedNeXt-v1 Large architecture configuration referenced by that paper.

## Source of truth and explicit decisions

The paper defines a symmetric five-resolution 3D U-Net, uses the MedNeXt-v1
Large depth/expansion configuration for its Base model, inserts 3D Global
Response Normalization (GRN) after GELU in every standard, downsample, and
upsample block, and fixes depthwise kernels at 3x3x3.

The numerical architecture configuration is therefore:

```text
stage channels:       C, 2C, 4C, 8C, 16C
block counts:         3, 4, 8, 8, 8, 8, 8, 4, 3
expansion ratios:     3, 4, 8, 8, 8, 8, 8, 4, 3
depthwise kernel:     3x3x3
base C:               32
wide C:               64
```

The paper leaves several implementation details implicit. This project makes
the following decisions and documents them as part of its reference design:

- Support 3D only. MedNeXt v2 is not presented as a 2D architecture.
- Use `GroupNorm(C, C)` for the paper's instance-like normalization. This has no
  running statistics and matches the normalization used by MedNeXt v1.
- Use additive encoder/decoder skip connections and residual projections with
  the existing MedNeXt shape semantics.
- Use convolution biases, same padding, transposed-convolution padding, and odd
  output padding behavior matching the current v1 implementation. Before each
  decoder skip addition, reconcile the decoder tensor to the saved skip's
  spatial shape by right-cropping excess voxels and right-padding deficits.
  This makes every positive input size that survives four downsampling stages
  deterministic, including odd sizes, and restores the final output to the
  input spatial shape.
- Initialize GRN `gamma` and `beta` to zero, so GRN begins as an identity map.
- Add `eps=1e-6` to the GRN denominator to define finite zero-input behavior.
- Follow the paper's channel **sum** of spatial norms rather than silently using
  the channel mean from the original ConvNeXt-v2 formulation.
- Keep context size out of model identity. A Base model can consume 128-cubed or
  192-cubed inputs with the same weights.
- Keep deep-supervision output behavior consistent with this package: native
  resolution tuple by default during training, configurable list/stacked output,
  and one primary tensor during evaluation.

## Package structure

MedNeXt v1 and v2 remain separate model stacks:

```text
src/mednext_accel/
  models/
    config_v2.py       immutable YAML-friendly v2 configuration
    blocks_v2.py       standard/down/up v2 blocks
    mednext_v2.py      v2 encoder-decoder and factories
  ops/
    grn.py             public reference GRN layer and adaptive dispatch
    _triton/grn.py     fused forward/backward kernels
```

V2 does not inherit from `MedNeXtV1`, and v1 blocks do not gain a GRN switch.
Small stable helpers may be shared where doing so does not alter v1 module paths,
state-dict keys, or scripted graphs. A generalized U-Net base class is outside
scope.

## Public API

The package exports:

```python
MedNeXtV2
MedNeXtV2Config
GlobalResponseNorm3d
get_mednext_v2_config
mednext_v2_base
mednext_v2_wide
```

The factories accept the established model-facing options where applicable:

```python
mednext_v2_base(
    *,
    in_channels: int,
    out_channels: int,
    deep_supervision: bool = False,
    deep_supervision_output: str = "tuple",
    checkpointing: CheckpointConfig | None = None,
    optimization: OptimizationSource = "auto",
)
```

`MedNeXtV2Config` contains only YAML-serializable primitives. Sequence inputs
are normalized to tuples during validation. Architecture-defining values are
available for inspection, but published factory configurations do not expose a
kernel-size option.

The existing `mednext_small`, `mednext_base`, `mednext_medium`, and
`mednext_large` names continue to mean v1.

## Reference block mathematics

For input `x` in NCDHW layout, a standard block computes:

```text
z = depthwise_conv3d(x, kernel=3, stride=1, padding=1)
z = group_norm(z, groups=C)
u = pointwise_expand(z)                 # C -> C * R
v = gelu(u)
w = grn(v)
y = pointwise_project(w) + x            # C * R -> C
```

For expanded input `v` with shape `(N, E, D, H, W)`:

```text
g[n,c] = sqrt(sum_spatial(v[n,c] ** 2))
s[n]   = sum_channel(g[n,:]) + eps
r[n,c] = g[n,c] / s[n]
grn(v) = v + gamma * v * r + beta
```

`gamma` and `beta` have broadcast shape `(1, E, 1, 1, 1)`. Spatial reductions
accumulate in FP32 for FP16 and BF16 inputs, and the returned tensor uses the
input dtype. The reference uses a vector-norm operation whose subgradient at a
zero vector is defined as zero; the fused backward implements the same rule
explicitly rather than dividing by a zero norm.

Down blocks replace the depthwise convolution with stride two and add a
stride-two pointwise residual projection. Up blocks use a stride-two depthwise
transposed convolution and a stride-two pointwise transposed residual projection.
Both apply normalization, expansion, GELU, GRN, and projection in the same order
as the standard block. The four down blocks use expansion ratios `(4, 8, 8, 8)`
and the four up blocks use `(8, 8, 4, 3)`, corresponding to entries 1:5 and 5:9
of the nine-stage ratio tuple. Decoder tensors are reconciled to the saved skip
shape before addition using the deterministic right-crop/right-pad rule above.

## Activation checkpointing

The existing `CheckpointConfig` is reused without a new user-facing policy
type.

- `style="expansion"` checkpoints `expand -> GELU -> GRN -> project`.
- `style="block"` checkpoints a complete standard/down/up block.
- `stages=(...)` selects resolution levels zero through four exactly as in v1.
- Evaluation and disabled-gradient execution bypass checkpointing.

Checkpointing changes saved activations only. It does not change parameters,
state-dict keys, or numerical model semantics within the normal floating-point
recomputation tolerance.

## Operator descriptors and existing convolution optimization

V2 construction uses `ModelOptimizationContext(family="mednext_v2", ...)` and
the existing operator/phase/implementation registry. Adaptive depthwise,
transpose-depthwise, downsample, and pointwise wrappers are installed using the
same descriptor guards as v1. Rules remain operator based, so a matching v2
shape can use existing evidence without a v2-specific kernel copy.

The registry is generalized before adding GRN. `OperatorDescriptor` retains a
small common identity (`family`, `direction`, input/output channels, and role),
while convolution geometry becomes optional and family-specific. Named
constructors create convolution and normalization descriptors. A GRN descriptor
uses `family="global_response_norm3d"`, equal expanded input/output channels,
its block role, and no kernel, stride, padding, dilation, or group geometry.
Policy facts expose only fields present on the descriptor, so a condition that
requires an absent geometry field cannot match.

Eligibility is registered per implementation rather than inferred from an
implementation-name prefix. Convolution implementations keep their current
geometry checks; `triton_fused_grn` receives a GRN-specific device, dtype,
layout, rank, and autograd guard. This removes the current assumption that every
`triton_*` implementation is a depthwise convolution and provides the intended
extension point for future non-convolution operators.

Architecture-level evidence and future full-model overrides remain scoped to
the `mednext_v2` family. `optimization="reference"` constructs only native
PyTorch convolution paths. `optimization="auto"` uses bundled policy layers.

## GRN implementations and dispatch

The stable operator family is `global_response_norm3d`, with two initial
implementations:

```text
reference
triton_fused_grn
```

The reference implementation defines correctness and is used for CPU,
unsupported dtype/layout/device/shape, evaluation export, and explicit reference
mode. The fused implementation is a numerical-equivalence CUDA training path.
Both use the same `gamma` and `beta` Parameter objects and therefore share one
state-dict format.

The adaptive GRN layer resolves one integral implementation for phase
`"training"`; forward, dX, dGamma, and dBeta cannot select different
implementations. Their separate timings are profiler result fields used to
explain the combined result. `explain_optimization()` reports GRN alongside
convolution decisions. A failed guard falls through to reference.

Initial fused eligibility is deliberately narrow:

- CUDA NCDHW contiguous tensors;
- FP16 or BF16 inputs;
- five-dimensional batched tensors;
- finite positive dimensions and supported Triton/PyTorch integration APIs;
- training with gradients required.

Evaluation mode, execution with gradients disabled, and all export contexts
take a direct reference branch before policy resolution, Triton import, or
custom-operator invocation. Wider training eligibility is added from evidence
rather than assumption.

## Fused GRN forward and backward

The implementation uses `torch.library.triton_op`, `wrap_triton`, registered
autograd, fake/meta behavior, and `torch.library.opcheck` coverage.

Forward performs:

1. tiled FP32 spatial sum-of-squares partial reductions;
2. completion of per-`(N,C)` norms and per-`N` channel sums;
3. fused application of `v + gamma * v * r + beta` without materializing
   full-sized `r` or `v * r` tensors.

Backward returns dX, dGamma, and dBeta. It uses FP32 reductions for channel
statistics and affine gradients, guards zero norms explicitly, and applies the
coupled derivative caused by the channel-normalized norm. Saved tensors are
limited to values that beat recomputation in measured full-model memory and
latency. Expansion checkpointing naturally recomputes the forward branch.

For FP16 and BF16 inputs, the spatial squared norm, square root, channel sum,
ratio, and affine expression are computed in FP32; the final activation is cast
once to the input dtype. `gamma` and `beta` remain FP32 parameters, and their
gradients remain FP32. FP32 input stays FP32. These rules define the oracle for
both the reference and fused paths and avoid backend-dependent autocast
semantics.

Kernel launch parameters are recipes in the implementation registry/policy,
not architecture constructor arguments. The profiler has an internal explicit
implementation override so correctness and performance tests can force either
path without publishing a policy rule. This override is a testing/profiling
mechanism, not a model constructor argument.

The first public v2 release is gated on at least one evidence-backed positive
`triton_fused_grn` rule for RTX 5090. The rule is published only after isolated
numerical validation and a full-model acceptance run both pass. Other SM
families remain on reference GRN until their own evidence supports a positive
rule; users may still supply an external policy for measured configurations.

GELU+GRN and GRN+projection fusion are outside the first release. They remain
possible later without changing the public model or checkpoint schema.

## Profiling and acceptance

The profiling model factory becomes family aware rather than adding another
v1-only conditional. It can discover v2 convolution shapes and GRN shapes from
Base and Wide workloads. GRN policy dispatch uses the combined `training`
phase. The profiler additionally records forward, dX, dGamma, and dBeta timing
and memory as diagnostic submetrics; those submetrics do not form independently
selectable policy phases.

Initial performance evidence targets RTX 5090, L40S, and RTX A6000 when those
machines are available. RTX 5090 evidence is required for the first fused-GRN
release. Absence of another machine does not block the model; it only prevents
a positive GRN rule for that SM family.

A fused GRN configuration is eligible for automatic dispatch only when:

- forward, dX, dGamma, and dBeta meet dtype-appropriate numerical tolerances;
- zero input, small norms, batch one, and multi-batch cases are finite;
- eager isolated latency or memory improves materially for its bounded region;
- compiled full-model training does not materially regress;
- the provisional policy passes the existing full-model publication gate.

Negative or inconclusive results publish no positive rule. Users can still
select reference behavior without Triton installed.

## Export behavior

The evaluation graph uses only standard PyTorch operations. Export detection is
not the safety boundary: evaluation mode itself takes the reference GRN path
before resolver/backend access, including when the model was created by the
default `optimization="auto"` factory. The following must work for v2 Base in
pure eval mode from that default factory:

- direct `torch.jit.trace`, save, and load;
- strict `torch.export.export`;
- dynamo ONNX export, ONNX checker, and ONNX Runtime parity.

Exported graphs contain no `mednext_accel::*` custom nodes. Deep supervision does
not alter the evaluation signature. Export does not require Triton.

## Validation strategy

### GRN reference

- formula tests against an independent tensor expression;
- zero-input and small-norm finite behavior;
- double-precision gradcheck for dX, dGamma, and dBeta;
- BF16/FP16 comparison with FP32 reduction reference;
- stable zero initialization and state-dict shape tests.

### Fused GRN

- forward and every gradient against the reference over multiple batches,
  channels, and spatial sizes;
- noncontiguous/unsupported fallback tests;
- autocast, registered-autograd, fake tensor, opcheck, eager, and fullgraph
  compile tests;
- isolated timing and peak-memory evidence;
- full-model acceptance before automatic dispatch.

### V2 model

- Base and Wide topology and stage counts;
- exact parameter counts of 61,969,507 and 246,536,387 respectively for one
  input channel and three output classes, consistent with the paper's rounded
  62M and 247M values;
- standard/down/up block shapes and odd-size behavior;
- training/evaluation deep-supervision contracts;
- expansion and whole-block checkpoint forward/gradient parity;
- `optimization="auto"` versus reference parameter identity and state-dict
  schema identity;
- eager forward/backward and CUDA fullgraph compilation;
- TorchScript, torch.export, and ONNX Runtime evaluation parity.

The test suite does not attempt to reproduce segmentation accuracy because this
package does not supply a training pipeline or pretrained weights.

## Delivery sequence

1. Add v2 config, reference GRN, blocks, model, factories, and CPU correctness
   tests.
2. Add checkpointing and existing convolution-policy integration.
3. Add TorchScript, torch.export, ONNX, packaging, and documentation coverage.
4. Add GRN operator metadata, adaptive dispatch, and fused forward/backward.
5. Run isolated and full-model CUDA validation; publish only evidence-supported
   bundled rules.
6. Document architecture decisions, supported execution modes, performance
   evidence, and limitations. Update the earlier activation-memory note whose
   provisional GRN formula used a channel mean; this design follows the paper's
   channel sum.

Each step leaves the reference model usable and testable. Fused GRN development
cannot redefine the model's mathematical behavior or parameter schema.

## Primary references

- [MedNeXt-v2 paper](https://arxiv.org/abs/2512.17774)
- [Published MedNeXt-v1 Large configuration](https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/network_architecture/mednextv1/create_mednext_v1.py)
- [PyTorch custom Triton operator integration](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
