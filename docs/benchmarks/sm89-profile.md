# SM89 L40S policy evidence

The compact [SM89 policy](../../src/mednext_accel/policies/sm89.yaml) supplements
[shared NVIDIA rules](../../src/mednext_accel/policies/shared-nvidia.yaml).
Missing rules continue to the shared layer; explicit reference rules mark known
counterexamples. These policies apply to matching BF16 operators independently
of checkpoint style and model name. They are not a table requiring every batch
to have been measured.

## Sources

| Source | Workload | Observations |
|---|---|---|
| Legacy L40S campaign | Base, 128³, BF16, all-expansion; B1–8 and B10 | 432 cases; 78 retained policy rules, principally depthwise |
| New L40S campaign | Base, 128³, BF16, all-expansion; selected B10 | 48 cases; 46 numerical passes and two pointwise validation OOMs |

The newer raw source SHA-256 is
`fffa74b383d74e2bcc01e1edc189c5d01249aea7d10c6dd7cd595b836b5d538e`.
The legacy raw source hash was
`c17c79e3a752ab9871cec8ccd2ec4a71767ca3381f28d26cea5e85c005d81279`;
that raw file was overwritten locally. Its retained bundled-policy summary had
hash `a84d04d432e396cbc779b0a364387aa7ff964f8b0ab9d3dd61c8fa1f4799d668`.
Before removing the old JSON runtime policy, all 78 rules were captured in the
[legacy rule fixture](../../tests/optimization/fixtures/legacy_sm89_rules.md).
Raw local files and machine-local paths are not distributed.

The recorded legacy environment was L40S SM89 with 47,667,740,672 bytes VRAM,
driver 595.84, Linux 6.8.0-137-generic, Python 3.10.14, PyTorch 2.14.0+cu130,
CUDA 13.0, cuDNN 92400, Triton 3.8.0, and mednext-accel 0.1.0a0, using
`max-autotune-no-cudagraphs` and the balanced objective. The legacy artifact did
not record complete model identity; Base was confirmed by the person who ran
it. Current evidence stores complete normalized campaign identity directly.
These environment details describe that historical run, not all SM89 devices.

## Evidence and inference

Speedup here means reference operator time divided by candidate operator time.
It is not whole-model training speedup.

| Operator observation | Result | Current policy consequence |
|---|---|---|
| Downsample dX across four scales on new SM86/SM89 runs | About 2.1–2.2× | Shared rule above work 1,000,000 |
| Regular dX on those runs | About 1.09–1.31× | Shared rule above work 1,500,000 |
| Transpose dW on those runs | About 1.08–2.54× | Shared rule above reduction work 4,096 |
| Regular dW at 128³/C32 | L40S B10 wins; A6000 B12 loses | SM89 B1–10 interpolation; no universal shared rule |
| Regular dW at 16³/C256 | L40S B10 loses, 0.649× | SM89 B1–8 positive; exact B10 reference tombstone |
| Pointwise at B10 | Three balanced winners | Exact B10 rules below |

The three B10 pointwise winners are 1→32 at 128³, 64→32 at 128³, and 128→384
at 63³. The stem also has cross-SM support, so its shared rule covers every
positive batch. The other two remain exact B10 observations. Those labels mean
measured SM/shape/batch contexts, not an exact GPU product match.

B9 at 128³ receives the interpolated dW choice despite never appearing in the
legacy measurements. Shared dX and transpose dW rules cover B9 and batches above
10 when their work thresholds hold. At 16³ the B10 counterexample prevents an
unjustified interpolation across that boundary. Other unmatched phases still
continue through shared rules before native fallback. This preserves useful
inference while keeping observed exceptions explicit.

## Numerical-validation history

The legacy pointwise validator trained BF16 parameters and reduced elementwise
`allclose` checks into one flag. Its 263 pointwise rejections did not identify a
component or normalized error and were not valid evidence of a general B>1
GEMM failure. Those old observations were not relabeled.

The repaired FP32-parameter/BF16-autocast validator checks output, dX, dW, and dB
separately, requiring finite values and relative-L2 error below 2%. In the new
L40S B10 source, all 28 executed pointwise cases passed, with worst component
relative-L2 about 0.726%. The two remaining cases, 64→128 and 128→32 at 127³,
ran out of memory during validation. They are evidence gaps, not evidence that
the implementation is slow or numerically invalid.

Current validation streams one component at a time and uses bounded error
scratch. RTX 5090 smoke checks cover batches 1/2/3 and high-resolution cases;
they do not establish that these L40S B10 OOM cases now fit. A new L40S run is
needed for that conclusion. Reduced validator scratch does not reduce the
candidate's runtime training memory.

## Collect another run

Save this as `l40s-base-128.yaml`:

```yaml
preset: mednext-v1
objective: balanced
compile_mode: max-autotune-no-cudagraphs
workloads:
  - variant: base
    spatial: [128, 128, 128]
    checkpointing: all-expansion
batch_search:
  memory_fraction: 0.90
  maximum: 10
```

```bash
mednext-accel profile l40s-base-128.yaml
```

If B10 fits it is the only kernel-profiled batch; otherwise the maximum feasible
batch is found below it. Omit `maximum` to discover the upper bound. The CLI
prints `sm89-local.policy.yaml` and an immutable evidence JSON path. Load the
YAML through `optimization=`. The raw evidence remains available even if the
model time/memory comparison rejects the provisional policy. See
[output and acceptance semantics](../optimization.md#two-outputs-and-acceptance).
No new whole-model L40S speedup is claimed by this policy revision.
