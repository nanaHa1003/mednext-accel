# SM89 Profile and Pointwise Validation Design

## Problem

The L40S profile artifact is valid schema-v1 evidence, but its pointwise
measurements expose a profiler defect. The pointwise probe uses BF16 trainable
parameters and rejects the complete result with one elementwise `allclose`.
Long dW and dB reductions can therefore fail on harmless rounding near zero.
The artifact records only `valid: false`, so the component and error magnitude
cannot be audited later.

The artifact also contains useful, whole-model-validated SM89 depthwise
measurements. Those decisions should become an exact-SM bundled default without
committing the raw local artifact or relabeling rejected pointwise results.

## Pointwise validation

The isolated pointwise probe will match real model training:

- BF16 activation and upstream gradient;
- FP32 trainable weight and bias;
- BF16 CUDA autocast around native convolution and per-sample GEMM;
- forward, dX, dW, and dB compared independently.

Each component must be finite and have relative L2 error below 2%. The result
records the validator name, per-component relative-L2 and maximum-absolute
errors, and a deterministic rejection reason. Tests must prove that equivalent
batch 1, 2, and 3 operations pass and deliberately corrupted components fail.
The large validation tensors are released before timing so they do not inflate
candidate or reference peak-memory measurements.

Measurement serialization will retain:

- the child probe status and message;
- kernel numerical validity;
- whole-model validation outcome;
- final validity used by synthesis;
- validator version, metrics, and rejection reason.

This is additive schema-v1 measurement metadata. Existing profiles remain
readable, and resolver behavior still depends on the final `valid` field.

## Bundled SM89 profile

`sm89-local.json` remains an ignored local artifact. Its SHA-256 is recorded in
sanitized provenance so the bundled profile can be reproduced. The bundled
`sm89.json` retains the artifact's 78 synthesized rules and reference defaults,
but omits all 432 raw measurements and machine-local output details.

The evidence applies directly to:

- NVIDIA L40S / SM89;
- BF16 training;
- MedNeXt v1 Base, input 128³, 1 input channel, 3 output classes;
- `all-expansion` checkpointing;
- measured batches 1–8 and 10 under the legacy dense campaign.

The normalized campaign identity is reconstructed from the user's explicit
confirmation and marked as such in provenance. Existing rules remain
shape-based because the optimized operators are model-independent once tensor
identity is fixed. Their checkpoint, dtype, direction, phase, channel, shape,
and batch guards remain unchanged.

Rejected pointwise measurements are not converted into rules. The one existing
pointwise rule stays because it was numerically and whole-model valid in the
source artifact. Batch 9 and unmatched contexts correctly fall back to the
reference implementation.

## Profile selection

The bundled-profile map will select:

- SM89 → `sm89`;
- SM120 → `sm120`;
- every other NVIDIA SM → `generic-nvidia` with the existing once-only warning.

An explicitly supplied profile targeting another SM continues to load with a
warning rather than being blocked.

## Verification

Required evidence:

- pure validator tests for finite checks, normalized-error acceptance, and
  deliberate corruption;
- RTX 5090 pointwise forward/backward smoke at batches 1, 2, and 3 using FP32
  parameters under BF16 autocast;
- measurement metadata round-trip and whole-model validity separation;
- all bundled profiles parse and ship in wheel/sdist;
- SM89 registry selection emits no unknown-SM warning;
- representative L40S rules resolve to their recorded implementations;
- batch 9, another checkpoint policy, and an unmatched pointwise case resolve
  to reference;
- source artifact hash and generated-rule equality are audited before commit;
- full CPU suite, Ruff, package build, Twine, and installed-wheel tests pass.
