# SM89 Profile and Pointwise Validation Implementation Plan

**Goal:** Repair pointwise profiling validity and ship the validated L40S evidence as the bundled SM89 default.

**Spec:** `docs/superpowers/specs/2026-09-22-sm89-profile-and-pointwise-validation-design.md`

## Task 1: Make pointwise validation training-realistic and auditable

**Files:** `src/mednext_accel/profiling/runner.py`, `src/mednext_accel/profiling/synthesize.py`, `src/mednext_accel/profiling/validation.py`, focused profiling/ops tests.

- [ ] Add failing pure tests for per-component finite/relative-L2 validation, accepted reduction-order noise, and deliberate corruption.
- [ ] Add failing measurement tests for probe status/message, kernel validity, whole-model outcome, validator version, metrics, and rejection reason.
- [ ] Use FP32 parameters with BF16 autocast in `_pointwise_probe`; validate output/dX/dW/dB separately and release comparison tensors before timing.
- [ ] Keep `valid` as the final synthesis decision while preserving numerical and whole-model evidence separately.
- [ ] Run focused tests and an RTX 5090 batch 1/2/3 forward/backward smoke.
- [ ] Run the full CPU/Ruff gate and commit `fix: make pointwise profiling numerically robust`.

## Task 2: Bundle the measured SM89 default

**Files:** `src/mednext_accel/profiles/sm89.json`, `src/mednext_accel/optimization/profiles.py`, bundled-profile/resolver/package tests.

- [ ] Audit `sm89-local.json` hash, parseability, rule count, implementation registry, and absence of sensitive fields.
- [ ] Generate a compact `sm89.json` with all 78 existing rules, reference defaults, no raw measurements, and sanitized provenance including the user-confirmed Base campaign identity and legacy dense coverage.
- [ ] Select `sm89` for `(8, 9)` without warnings; keep unknown-SM and explicit cross-SM behavior unchanged.
- [ ] Test representative positive decisions plus reference fallbacks for batch 9, non-all-expansion checkpointing, and invalid/unmeasured pointwise shapes.
- [ ] Verify the raw artifact remains ignored and is not staged.
- [ ] Run focused/full/package gates and commit `feat: add measured SM89 optimization profile`.

## Task 3: Document evidence and limits

**Files:** `README.md`, `docs/optimization.md`, `docs/benchmarks/sm89-profile.md`.

- [ ] Document automatic SM89 selection, exact L40S environment, coverage, source hash, legacy dense evidence, and reference fallbacks.
- [ ] Explain that the old pointwise rejection pattern came from the repaired validator and was not relabeled in the bundled profile.
- [ ] Document how a new maximum-only L40S rerun can supersede the conservative pointwise coverage.
- [ ] Run all CPU/style/package checks, inspect archives, and commit `docs: publish SM89 profile evidence`.

## Final review

- [ ] Independently review the complete diff from `030cb15`.
- [ ] Confirm the source artifact is untracked, bundled evidence is reproducible, and no unsupported performance claim is made.
- [ ] Resolve findings, rerun final gates, and leave the worktree clean.

