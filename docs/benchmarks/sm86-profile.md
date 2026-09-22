# SM86 RTX A6000 policy evidence

The [SM86 overlay](../../src/mednext_accel/policies/sm86.yaml) adds two regular-dW
rules to the [shared NVIDIA policy](../../src/mednext_accel/policies/shared-nvidia.yaml).
It is informed by an RTX A6000 campaign for MedNeXt Base, 128³, BF16 training,
all-expansion checkpointing, at selected batch 12. Raw-source SHA-256:
`b94e423468b0ee5839fe09a62d6e160fb6739fbcb0c909ab53b10c3640464fae`.
The machine-local JSON is not distributed.

## What the run establishes

Of 48 kernel cases, 46 passed component numerical validation; two pointwise
127³ cases ran out of memory in validation. Eighteen passed the balanced
operator objective. The legacy aggregate model time/memory gate rejected the
combined policy and changed final validity flags, producing zero runtime rules.
That aggregate result does not invalidate the retained per-kernel numerical
results. Policy v2 keeps those observations separate from model acceptance.

| Regular depthwise dW | RTX A6000 B12 speedup | Current SM86 choice |
|---|---:|---|
| 128³, C32 | 0.926× | Native dW; inferred shape-region reference tombstone |
| 16³, C256 | 1.583× | Split dW from B12 upward, inferred same-SM |

Speedup is reference time divided by candidate time for that isolated phase.
The 128³ result conflicts with L40S B10's win; the 16³ result conflicts with
L40S B10's 0.649× loss. Different batches, products, and environments are involved,
so these observations do not isolate SM architecture as the cause. They do
justify keeping these two regions out of the universal shared dW policy.

Shared regular/downsample dX, transpose dW, stable 64³/32³/8³ dW regions, and
stem pointwise GEMM still apply when their conditions hold. Other pointwise
shapes do not inherit the RTX 5090 pointwise table. Regular dX remains independent
of dW, including when the 128³ dW tombstone selects native backward.

The two SM86 rules extend beyond the measured batch as explicit hypotheses:
128³ retains native dW across positive batches, and the 16³ win extends upward
from B12. The former is a conservative exception; the latter assumes the larger
reduction preserves the observed advantage. Neither is a claim of measuring
every batch or every SM86 device. The rules apply across checkpoint styles;
application checkpointing and batch size remain user choices.

## Reproduce or refine

Use the [single-workload campaign example](../optimization.md#run-an-optional-profiling-campaign)
on the A6000, with `maximum: 12` to target the prior selected batch. Omit the
maximum to discover the boundary for the current software/environment.

```bash
mednext-accel profile a6000-base-128.yaml
```

Load the printed `sm86-local.policy.yaml` through `optimization=`. The adjacent
immutable evidence JSON preserves kernel and model results, including failures.
An external SM86 policy may also be intentionally used on another SM with a
warning. The current bundled overlay provides useful defaults without requiring
this profiling run. No whole-model A6000 speedup is claimed from the rejected
aggregate experiment.
