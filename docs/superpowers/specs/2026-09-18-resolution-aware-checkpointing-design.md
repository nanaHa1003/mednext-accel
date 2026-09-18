# Resolution-Aware Selective Checkpointing Design

**Date:** 2026-09-18

## Objective

Reduce the recomputation cost of expanded-branch activation checkpointing by
applying it only at selected resolution levels. Preserve the existing
mathematical model, state-dict keys, legacy whole-block behavior, and the
meaning of `checkpoint_style="expanded"` without a level selector.

The immediate experiment is MedNeXt Base with a `128x128x128` input. It will
compare checkpointing level 0, levels 0--1, levels 0--2, every level, and the
whole-block policy. The public interface must describe model topology rather
than a particular input shape so it remains predictable for other patch sizes.

## Public Interface

Keep `checkpoint_style` as the answer to “what is recomputed?” and add
`checkpoint_levels` as the answer to “where is it recomputed?”:

```python
model = mednext_base(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    checkpoint_style="expanded",
    checkpoint_levels=(0, 1),
)
```

The constructor type is:

```python
checkpoint_levels: Optional[Tuple[int, ...]] = None
```

The accepted combinations are:

| `checkpoint_style` | `checkpoint_levels` | Meaning |
|---|---|---|
| `"none"` | `None` | No activation checkpointing. |
| `"expanded"` | `None` | Checkpoint every expanded branch. This preserves current behavior. |
| `"expanded"` | `(0,)` | Checkpoint expanded branches at full resolution. |
| `"expanded"` | `(0, 1)` | Checkpoint expanded branches at the two highest resolutions. |
| `"block"` | `None` | Existing whole-block checkpointing. |

Passing explicit levels with `none` or `block` raises `ValueError`. Partial
whole-block checkpointing is outside this change. `use_grad_checkpoint=True`
continues to resolve to `block`; combining it with either an explicit style or
explicit levels raises `ValueError`.

An empty sequence, duplicate levels, Boolean elements, non-integer elements,
negative levels, and levels greater than the model depth raise `ValueError`.
Input order has no runtime meaning. The model stores a sorted tuple in
`self.checkpoint_levels` for deterministic introspection. `None` remains stored
as `None` and means all levels only when the style is `expanded`.

## Resolution-Level Semantics

Level zero is the input resolution. Each increment represents one factor-of-two
spatial downsampling. For a `128x128x128` input to the standard depth-four
factories:

| Level | Spatial shape |
|---:|---:|
| 0 | `128x128x128` |
| 1 | `64x64x64` |
| 2 | `32x32x32` |
| 3 | `16x16x16` |
| 4 | `8x8x8` |

Assignment follows the spatial shape at the input to the pointwise expansion,
after the block's depthwise operation:

- encoder blocks at encoder stage `i` belong to level `i`;
- the down block after encoder stage `i` belongs to level `i + 1` because its
  stride-two depthwise convolution runs before expansion;
- bottleneck blocks belong to level `depth`;
- the up block targeting decoder stage `i` belongs to level `i` because its
  transpose depthwise convolution runs before expansion;
- decoder blocks at decoder stage `i` belong to level `i`.

This definition includes encoder, decoder, and transition blocks consistently.
It does not inspect runtime tensor shapes. Selection is resolved while the
model is constructed, so dynamic inputs do not select a different policy and
the selector does not add runtime branches or compiler specializations.

For MedNeXt Base, which has two normal blocks per stage, the 26 expansion
branches are distributed as follows:

| Level | Encoder | Down | Bottleneck | Up | Decoder | Total |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 0 | 0 | 1 | 2 | 5 |
| 1 | 2 | 1 | 0 | 1 | 2 | 6 |
| 2 | 2 | 1 | 0 | 1 | 2 | 6 |
| 3 | 2 | 1 | 0 | 1 | 2 | 6 |
| 4 | 0 | 1 | 2 | 0 | 0 | 3 |

## Internal Representation

`activation_checkpoint.py` remains responsible for policy validation. Add a
resolver that accepts the resolved style, user levels, and model depth and
returns `None` or a canonical sorted tuple. It must not import MedNeXt.

`MedNeXt` resolves the policy once, then decides each block's existing
`checkpoint_expanded` Boolean from its statically known level:

```python
checkpoint_expanded = (
    checkpoint_style == "expanded"
    and (
        checkpoint_levels is None
        or block_level in checkpoint_levels
    )
)
```

The block implementation and checkpoint boundary remain unchanged. No module,
Parameter, or buffer is added for the level selector, so state dictionaries are
independent of the selected levels.

## Profiler Interface

Add this option to `profile_mednext.py`:

```text
--checkpoint-levels LEVEL [LEVEL ...]
```

It is valid only with `--checkpoint-style expanded`. The summary records the
canonical list as `effective_checkpoint_levels`; `null` means all levels.
Legacy summaries without the field continue to mean all levels for an expanded
policy and no level selector for the other policies.

Add compiled matrix cases:

```text
compile_expanded_l0
compile_expanded_l01
compile_expanded_l012
```

Keep `compile_expanded` as the all-level case and `compile_ckpt` as the
whole-block case. Case names are conveniences for the profiler; they are not
model API values.

## Correctness and Compatibility

The implementation must preserve:

- all parameter names and tensor shapes;
- strict state-dict loading among every checkpoint configuration;
- exact evaluation output because checkpointing is inactive in evaluation;
- exact double-precision training output and gradients against `none`;
- current non-reentrant checkpoint behavior;
- deep-supervision output semantics;
- eager and `torch.compile(fullgraph=True)` execution;
- composition with pointwise-GEMM and split-depthwise replacements;
- existing `checkpoint_style="expanded"`, `checkpoint_style="block"`, and
  legacy CLI behavior.

Tests must verify level assignment for normal, down, bottleneck, up, and decoder
blocks. For the standard Base layout, `(0, 1)` must enable exactly 11 of the 26
expanded branches. Tests must also prove that levels outside `(0, 1)` execute
without checkpoint calls during the same forward pass.

## Performance Experiment

Measure MedNeXt Base, batch size one, `128x128x128`, three and eight classes,
BF16 autocast, deep supervision, AdamW, and `torch.compile`. Run at least 20
warmup steps, 50 measured steps, and three independent repeats per case.

The primary matrix uses the current pointwise-GEMM plus split-depthwise path,
because its all-level expanded policy currently costs 16.2--16.6% relative to
`none`. A native-convolution matrix checks that conclusions are not specific to
the custom operators.

The report table for each GPU and operator path uses this form:

| Policy | Selected levels | Median step | Time vs. none | Peak allocated | Memory saved vs. none |
|---|---|---:|---:|---:|---:|
| none | none | measured | baseline | measured | baseline |
| expanded | `(0,)` | measured | calculated | measured | calculated |
| expanded | `(0, 1)` | measured | calculated | measured | calculated |
| expanded | `(0, 1, 2)` | measured | calculated | measured | calculated |
| expanded | all | measured | calculated | measured | calculated |
| block | all blocks | measured | calculated | measured | calculated |

Include raw repeat times, sample standard deviation, forward and backward GPU
times, and peak allocation in the detailed record.

Retain each partial policy only if it creates a distinct useful Pareto point.
If `(0,)` or `(0, 1, 2)` is dominated by another setting, omit a named preset
and document that users can still express the tuple directly. No automatic
policy is added in this change.

## Out of Scope

- Runtime spatial-size thresholds.
- Automatically selecting levels from available VRAM.
- Per-encoder or per-decoder masks.
- Partial whole-block checkpointing.
- A policy dataclass or callback API.
- MedNeXt v2 and GRN checkpointing.
- Changes to deep-supervision output format.
