# SM120 RTX 5090 policy evidence

The [SM120 overlay](../../src/mednext_accel/policies/sm120.yaml) combines
generalized operator regions, retained launch exceptions, and bounded reference
regions for integrated RTX 5090 counterexamples. It inherits the shared NVIDIA
policy, targets BF16 training, and keeps evaluation/export on native operators.
Legacy bundled-policy SHA-256:
`d33cf9e7ea50575de8e48d3ed8a1793054ad2459e0f5d61314660ac258746c1b`.

## Historical measurements

Source experiments used PyTorch 2.12 and 2.14 development builds, CUDA 13.x,
BF16 autocast, AdamW, deep supervision, and compiled full-graph training. In an
earlier three-class Base B1 comparison, custom regular dW+dX, transpose dW, and
pointwise selections reduced step time from 96.176 ms to 59.765 ms. Under
all-expansion checkpointing, a later batch-aware policy measured:

| Batch | Step time | Peak allocated |
|---:|---:|---:|
| 1 | 64.02 ms | 3,762 MiB |
| 2 | 149.11 ms | 8,459 MiB |
| 4 | 306.12 ms | 16,729 MiB |
| 6 | 465.77 ms | 21,992 MiB |

These are retained historical measurements, not a fresh benchmark of policy v2.
They have different baselines from the direct official/MONAI comparison in the
[README](../../README.md#rtx-5090-training-results); do not combine their speedup
percentages. They are synthetic model-only measurements, not a VRAM estimate for
every application optimizer, loss, or data pipeline.

## Integrated validation, 2026-09-23

The validation device was an RTX 5090 (SM120, 32,102.5 MiB), driver 595.84, Python
3.11.15, PyTorch 2.12.0+cu132, CUDA 13.2, cuDNN 9.20, and Triton 3.7.0. Operator
measurements used BF16 autocast with FP32 parameters,
`torch.compile(mode="default", fullgraph=True)`, and requested dX/dW/dB. Each
candidate was measured through its adaptive module, with other phases native,
against the corresponding native module. Output and all three gradients were
checked before timing.

The initial 93-point grid covered all 48 Base 128³ operator/phase cases at B1,
24 representative Base cases at the previously unmeasured B3, and 21 boundary
points. Boundaries included depthwise C31/32/33 and C511/512/513; pointwise
64→31/32/80/128/129 and 144→64; and 64→80 at 80³/81³ and 116³/117³ around the
pointwise reduction-work limits. The screening runs used two warmups and five
timed iterations. All 93 passed numerical validation, with maximum component
relative L2 error 0.00589 against the 0.02 limit.

The 22 initially selected custom losers and 35 neighboring cases were each
measured twice more with five warmups and 30 timed iterations. Two additional
checks confirmed the C128/32³/B1 dW launch winner. All 209 integrated probes
passed numerical validation. No candidate peak exceeded its reference peak in
the initial grid; candidate/reference peak ratios ranged from 0.333 to 1.000.
Absolute peaks include the benchmark's resident input and both module parameter
sets and are not per-layer contributions to whole-model memory.

Raw and integrated timing preferred different implementations in 41 of the 93
screening cases, including close timings susceptible to noise. A repeatable
counterexample was downsample dX at B1/C256/16³: its raw kernel improved from
0.02269 to 0.01843 ms, while the integrated operation worsened from 0.13773 to
0.15727 ms. Longer integrated repeats were 27.4% and 23.7% slower than native.
B2 and neighboring spatial sizes confirmed the losing region. Raw timings
therefore remain diagnostics and never qualify new generated dispatch rules.

The revised SM120 policy adds six bounded reference regions. They are
conservative same-SM inference across the indicated channels/work, not a claim
that every tuple in each region was measured. The shared, SM86, and SM89 rules
were not changed.

| Phase | Channels | Inclusive reference region |
|---|---|---|
| Downsample dX | C128–256 | work 864,000–4,599,936 |
| Regular dX | C32–512 | work 175,616–1,257,728 |
| Regular dW | C32–512 | reduction work 343–9,826 |
| Regular dW, generic launch neighbors | C128 | reduction work 29,791–35,937, except the retained B1/32³ launch |
| Transpose dW | C256 | reduction work 4,096–4,913 |
| Pointwise training | 512→2048 | reduction work 1,024–2,048 |

Reference selections stop shared-policy fallthrough. The C128/32³/B1 dW launch
remains custom ahead of its neighboring reference region: its special two-split,
block-1024 recipe outperformed native in both longer checks, while the generic
31³/33³ recipes lost. B3/C256/16³ regular dX/dW showed mixed results across runs,
so their negative boundaries were not extended through that point. Every case
that lost by more than 3% in both longer repeats now resolves native.

Selection regret is `selected_time / min(native_time, custom_time) - 1` for the
isolated phase comparison. The following recalculates routing over the **same
initial timings**, before and after the policy update. Occurrence weighting
counts each covered operator/phase's appearances in Base; it is not a predicted
whole-model speedup, since regular dX/dW cases time separate complete operators.

| Grid | Cases / Base occurrences | Mean regret before → after | Occurrence-weighted mean before → after |
|---|---:|---:|---:|
| Base B1 | 48 / 97 | 6.60% → 4.39% | 6.82% → 4.41% |
| Base B3 subset | 24 / 55 | 3.30% → 1.14% | 3.55% → 1.92% |
| Boundary points | 21 | 16.71% → 10.03% | Not representative of Base |

Residual regret includes conservative native choices outside positive regions:
the B3 64→129/64³ boundary point measured 2.1123 ms native versus 0.9271 ms
custom (127.85% selection regret). This single observation did not widen the
positive channel strip. These results establish neither optimal dispatch nor
performance on other compiler modes or SMs.

The grid used no CUDA Graph replay and recorded `memory_measured: true`.
CUDA Graph modes can hide working storage from allocator peaks; their operator
evidence retains observed peak scalars but cannot qualify balanced or memory
objectives. See [integrated measurement semantics](../optimization.md#integrated-measurements-and-dispatch-evidence).

### Full-model release sentinel

The resulting effective bundled policy was compared with `optimization="reference"`
for Base B1 at 128³, one input channel, three outputs, BF16 autocast, AdamW,
no deep supervision, and mean squared primary logits. Each fresh subprocess
used five warmups and ten measured steps with `default`, `fullgraph=True`.
This is a performance/memory sentinel; whole-model numerical equivalence is
`not-measured`. It is not the README's cross-entropy/deep-supervision benchmark.

| Checkpointing | Reference step | Auto step | Reference peak | Auto peak | Step reduction |
|---|---:|---:|---:|---:|---:|
| none | 73.50 ms | 58.22 ms | 8,308.9 MiB | 8,336.8 MiB | 20.8% |
| all-expansion | 78.62 ms | 62.56 ms | 3,731.3 MiB | 3,764.6 MiB | 20.4% |
| whole-block | 84.79 ms | 70.39 ms | 3,205.5 MiB | 3,236.8 MiB | 17.0% |

All three comparisons passed the balanced model gate: lower step time and peak
allocation within 115% of reference. Actual peak increases were below 1%.

### Local validation artifacts

The ignored directory `artifacts/generalized-policy-sm120/` retains environment
metadata, policy hashes, exact payloads/seeds, component errors, integrated and
raw timings, allocator peaks, routing/regret calculations, and sentinel results.
The one-off drivers and commands used in this checkout were:

```bash
PYTHONPATH=src python artifacts/generalized-policy-sm120/validate.py grid
PYTHONPATH=src python artifacts/generalized-policy-sm120/repeat.py
PYTHONPATH=src python artifacts/generalized-policy-sm120/validate.py sentinel
```

`grid-00-*.json` through `grid-11-*.json` contain the initial measurements;
`repeat-[12]-*.json` contain the longer runs; `launch-exception-[12].json` checks
the retained C128 launch. `grid-final-routing.json` and `grid-final-summary.json`
record occurrence-weighted regret. `sentinel-final-*.json` and
`sentinel-summary.json` record the final model comparisons. The combined
`integrated-grid.evidence.json` is immutable and its hash is linked from the
bundled YAML. These local artifacts and drivers are not shipped in Git or package
archives. The installed CLI below produces its own immutable evidence and
runtime policy pair.

## How the policy generalizes

Regular depthwise dX/dW use C32–512, work ≥262,144, and reduction work ≥512,
subject to the preceding integrated reference regions. This admits unseen
channels and spatial sizes, while structural launch scaling remains in Python.
The 64³/C64 transpose dW region retains its legacy recipe. Only two dW launch
exceptions remain explicit: 32³/C128 uses four splits at B4 and eight at B6.
Other removed overrides changed only descriptive confidence, not execution.
A [legacy launch fixture](../../tests/optimization/fixtures/legacy_dw_launches.md)
checks retained parameter values independently of the recipe implementation.

Non-stem pointwise rules use bounded channel and reduction-work regions,
principally derived from B2–6 or narrower intervals. Complete positive channel
strips interpolate; sparse isolated observations remain exact. For example,
64→32–128 with reduction work 524,288–1,572,864 admits unseen 64→80 and spatial
81³ at B1. The shared 1→32 stem rule at 128³ covers every positive
batch. The previous VRAM-only rule extending high-resolution pointwise GEMM to
cards with at least 48 GiB was removed: extra memory establishes feasibility,
not faster GEMM.

A 96-GB RTX PRO 6000 using SM120 therefore receives the structural depthwise
choices outside the bounded negative regions at batches above the RTX 5090
range, including B9 and larger values.
This is same-SM inference, not a measured performance claim on that product.
Other NVIDIA architectures inherit the smaller cross-SM shared policy instead
of the entire RTX 5090 table. Bundled and newly generated local rules are
independent of model variant and checkpoint style. Reports show custom, native,
and unwrapped convolutions, including 2D models and unsupported output/residual
operators.

Large-address kernel variants use wide indexing where required. Compiler/IR
checks cover address and reduction-loop boundaries beyond signed 32-bit ranges;
normal-size numerical execution is checked on RTX 5090. Those compiler checks
do not represent an allocated or timed 96-GB workload.

## Optional local refinement

```bash
mednext-accel profile
```

The command searches the reference model's feasible batch boundary, validates
and times candidate operators, and compares the provisional effective policy
at the selected batch. It prints a stable `sm120-local.policy.yaml` path and a
unique immutable evidence JSON path under `~/.cache/mednext_accel/profiles/`
(or the XDG cache directory). Supply the YAML to `optimization=`. A narrower
[single-workload YAML campaign](../optimization.md#run-an-optional-profiling-campaign)
can target Base or a particular batch upper bound. Model construction and first
forward never profile automatically.
