# SM89 L40S optimization profile

The bundled `sm89` profile is selected automatically on NVIDIA SM89 devices.
It contributes exact-SM rules before the generic NVIDIA layer, and model
construction performs no benchmarking. The profile contains 78 rules derived
from an L40S campaign. Its defaults are reference PyTorch implementations, so
an unmatched context remains correct and does not inherit an unsupported
optimized decision.

## Source and environment

The source was a local schema-v1 artifact with SHA-256
`c17c79e3a752ab9871cec8ccd2ec4a71767ca3381f28d26cea5e85c005d81279`.
The raw 432 measurements and machine-local output details are intentionally not
shipped. The bundled profile retains the ordered rules, reference defaults, and
sanitized provenance, and records the source hash so the transformation can be
audited.

| Field | Recorded value |
|---|---|
| Collected at | 2026-09-22T05:31:58.102266Z |
| GPU | NVIDIA L40S, SM89, 47,667,740,672 bytes VRAM |
| Driver and platform | 595.84; Linux 6.8.0-137-generic, x86-64, glibc 2.39 |
| Python | 3.10.14 |
| PyTorch stack | PyTorch 2.14.0+cu130, CUDA 13.0, cuDNN 92400, Triton 3.8.0 |
| mednext-accel | 0.1.0a0 |
| Compile mode | `max-autotune-no-cudagraphs` |
| Objective | `balanced` |
| Workload | MedNeXt v1 Base, 1 input channel, 3 output classes, 128³, BF16 training |
| Checkpoint policy | `all-expansion` |
| Batch evidence | Legacy dense batches 1–8 and 10 |
| Execution | 432 kernel cases, 45 kernel groups, 9 whole-model validations |
| Output | 78 ordered rules; zero raw measurements in the bundled profile |

The legacy artifact did not record the full normalized campaign identity. The
Base workload identity above was confirmed by the person who ran the campaign
and is marked `identity_source: user-confirmed` in bundled provenance. Profiles
created by the current profiler record the complete normalized campaign
directly, including model family, variant, spatial shape, channels, dtype,
phase, checkpoint policy, compile mode, and batch-search settings.

## Directly supported decisions

The following ranges are isolated operator measurements for source cases that
passed numerical and whole-model validation and met the `balanced` objective.
Speedup is reference time divided by candidate time. They quantify only the
named operator phase; they are not MedNeXt step-time or end-to-end training
speedups.

| Operator | Phase and implementation | Winning source cases | Measured speedup | Bundled rules |
|---|---|---:|---:|---:|
| Downsample depthwise Conv3d | dX, `triton_downsample_dx` | 36 | 1.41–2.22× | 8 |
| Regular depthwise Conv3d | dX, `triton_depthwise_dx` | 39 | 1.07–1.29× | 10 |
| Regular depthwise Conv3d | dW, `triton_split_dw` | 43 | 1.20–2.80× | 30 |
| Transpose depthwise Conv3d | dW, `triton_transpose_split_dw` | 29 | 1.05–2.55× | 29 |
| Pointwise Conv3d | training, `pointwise_gemm_per_sample` | 1 | 1.89× | 1 |

The dense source measured the tensor shapes reached by this Base workload at
batches 1–8 and 10. Adjacent winning batches with identical decisions were
merged into interval rules; every integer inside those intervals was measured.
Batch 9 was absent and is not filled by interpolation. The one pointwise rule
is specifically batch 1, 128³, 1→32 channels. This profile therefore does not
claim that pointwise GEMM is faster throughout Base or at batch sizes above one.

Rules still match operator identity rather than a model class. A rule may apply
inside another model only when its direction, tensor shape, channel counts,
batch, dtype, phase, and checkpoint policy all match the recorded case.

## Pointwise validator limitation

The legacy pointwise probe used BF16 trainable weights and bias, then collapsed
separate elementwise `allclose` checks for forward, dX, dW, and dB into one
unlabelled validity flag.
Long BF16 gradient reductions can differ near zero because their accumulation
order changes. That check consequently produced a false-negative pattern: 263
pointwise measurements were rejected without recording which component failed
or the normalized error. Those rejections are not evidence that pointwise GEMM
is intrinsically invalid or ineffective at batch sizes above one.

The current probe uses FP32 trainable weights and bias under BF16 autocast and
checks output, dX, dW, and dB independently. Every component must be finite and
have relative-L2 error below 2%; the profile records the validator version,
relative-L2 and maximum-absolute error for each component, and a deterministic
rejection reason. RTX 5090 correctness smoke tests passed at batches 1, 2, and
3, including 128³, but those checks are not L40S performance evidence.

The old L40S measurements were not relabeled after this repair. Only the one
pointwise source case that was already valid and fast enough remains in the
bundled profile. Six other valid legacy pointwise cases were slower and did not
meet the objective. All other pointwise contexts use the reference path.

## Reference fallbacks

The `sm89` profile uses reference implementations when no exact rule matches.
This includes:

- batch 9;
- checkpoint policies `none` and `whole-block`;
- dtypes, phases, shapes, channel counts, and directions outside the recorded
  rules;
- pointwise Base cases other than the single batch-1 input projection above;
- inference and export, which stay on portable PyTorch operators.

Supplying an SM89 profile explicitly on another architecture remains allowed
and emits a warning. Automatic selection applies this profile only to SM89.

## Superseding the conservative pointwise coverage

Run the current profiler on an L40S with the exact workload to collect evidence
using the repaired validator. For example, save this as `l40s-base-128.yaml`:

```yaml
preset: mednext-v1
objective: balanced
compile_mode: max-autotune-no-cudagraphs
workloads:
  - variant: base
    spatial: [128, 128, 128]
    dtypes: [bfloat16]
    phases: [training]
    checkpointing: all-expansion
    in_channels: 1
    out_channels: 3
batch_search:
  memory_fraction: 0.90
  maximum: 2
```

Then run:

```bash
mednext-accel profile l40s-base-128.yaml
```

When `maximum` fits, it is probed first and becomes the only batch used for
kernel profiling and whole-model validation. Set it to the batch whose exact
rule is needed; rerun with another maximum to collect another batch. Without an
upper bound, exponential growth and binary refinement find the maximum feasible
batch, and only that result is profiled. The resulting profile records the full
campaign identity plus per-component validation diagnostics. A future bundled
update can replace the conservative pointwise fallbacks only where these new
L40S measurements pass numerical and whole-model validation and beat the
reference implementation under the selected objective.
