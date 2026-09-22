# SM120 RTX 5090 policy evidence

The [SM120 overlay](../../src/mednext_accel/policies/sm120.yaml) preserves
shape-specific RTX 5090 choices while inheriting the shared NVIDIA policy.
It targets BF16 training and keeps evaluation/export on native operators.
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

## How the policy generalizes

Regular depthwise dX/dW for 128³/C32, 64³/C64, 32³/C128, 16³/C256, and 8³/C512
remain active for positive batches, using structural launch scaling in Python.
The 64³/C64 transpose dW region retains its legacy recipe. Only two dW launch
exceptions remain explicit: 32³/C128 uses four splits at B4 and eight at B6.
Other removed overrides changed only descriptive confidence, not execution.
A [legacy launch fixture](../../tests/optimization/fixtures/legacy_dw_launches.md)
checks retained parameter values independently of the recipe implementation.

Non-stem pointwise rules use measured contexts or bounded interpolation,
principally B2–6 or narrower intervals. B3 can inherit B2–4 choices without a
separate measurement. The shared 1→32 stem rule at 128³ covers every positive
batch. The previous VRAM-only rule extending high-resolution pointwise GEMM to
cards with at least 48 GiB was removed: extra memory establishes feasibility,
not faster GEMM.

A 96-GB RTX PRO 6000 using SM120 therefore receives the structural depthwise
choices at batches above the RTX 5090 range, including B9 and larger values.
This is same-SM inference, not a measured performance claim on that product.
Other NVIDIA architectures inherit the smaller cross-SM shared policy instead
of the entire RTX 5090 table. Bundled rules are checkpoint-independent.

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
