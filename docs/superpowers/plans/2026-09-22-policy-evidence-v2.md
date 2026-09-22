# Optimization Policy and Evidence v2 Implementation Plan

**Goal:** Replace exact-batch profile lookup tables with compact inferred policies and separate generated evidence.

**Spec:** `docs/superpowers/specs/2026-09-22-policy-evidence-v2-design.md`

## Task 1: Build the policy-v2 core beside the current integration

**Primary files:** new policy records/parser/resolver/parameter modules and focused tests. Keep the schema-v1 factory integration temporarily so the repository remains runnable until Task 2 switches it atomically.

- [ ] Write failing parser tests for policy v2, scope merging, numeric work ranges, tombstones, unknown fields, invalid recipes, and schema-v1 rejection.
- [ ] Write failing resolver tests for external → exact-SM → shared → reference precedence and omission versus tombstone behavior.
- [ ] Implement immutable policy records and YAML/JSON mapping loading without a formula DSL or blanket defaults.
- [ ] Implement recipe-default-plus-explicit-override semantics. Add SM86/89 recipes and the compact shape-aware SM120 recipe; snapshot every retained legacy launch value.
- [ ] Make policy selection depend on execution-context SM rather than construction-time GPU.
- [ ] Run focused policy tests and unaffected smoke tests; the full profile/factory gate remains a Task 2 integration requirement.
- [ ] Commit `refactor: introduce compact optimization policy v2`.

## Task 2: Execute phase-independent depthwise policies and switch bundled integration

**Primary files:** adaptive depthwise/autograd backend, factory optimization wiring, new YAML policies, registry selection, resolver behavior tests, removal of bundled JSON profiles.

- [ ] Encode a small shared NVIDIA policy from cross-device hypotheses.
- [ ] Add SM86, SM89, and SM120 overlays with documented evidence hashes, inferred regions, tombstones, and only real launch exceptions.
- [ ] Implement independent regular dX/dW execution for custom/custom, custom/native, native/custom, and native/native decisions; make reports agree with execution.
- [ ] Prove automatic parameters reproduce retained legacy launch choices.
- [ ] Test robust shared decisions, SM counterexamples, unknown-SM behavior, and RTX PRO 6000-like SM120 contexts.
- [ ] Test CPU/absent backend, unsupported geometry/dtype, inference/export, compile, and moving execution to a different SM from construction.
- [ ] Replace provisional schema-v1 synthesis/runner serialization with policy-v2 runtime mappings so profiling remains operational after the factory switch. Evidence records remain Task 3.
- [ ] Remove duplicated schema-v1 bundled JSON and verify package archives contain the YAML policies.
- [ ] Run focused/full/package gates and commit `feat: add compact architecture optimization policies`.

## Task 3: Evidence records and non-destructive validation semantics

**Primary files:** profiling execution/result models, batch search, validation, synthesis replacement, evidence tests.

- [ ] Define immutable evidence records for environment, campaign, ordered batch probes, kernel measurements, and model comparisons.
- [ ] Preserve kernel numerical results independently from aggregate policy acceptance.
- [ ] Rename/remove misleading whole-model validity fields and retain raw reference/candidate metrics and rejection reasons.
- [ ] Remove the unused campaign `phases` field and record training explicitly.
- [ ] Record deterministic seeds and scalar diagnostics while explicitly marking whole-model numerical equivalence `not-measured`.
- [ ] Preserve actual batch-search attempt order plus the indexed summary.
- [ ] Reject obsolete campaign `phases` before environment/CUDA side effects.
- [ ] Add provisional-effective-policy identity and separate performance, memory and final acceptance results using the specified objective formulas.
- [ ] Run focused/full gates and commit `refactor: separate profiling evidence from policy acceptance`.

## Task 4: Streaming pointwise validation and staged failures

**Primary files:** profiling runner/validation and GPU-focused tests.

- [ ] Reproduce current 127³ validator OOM accounting with a structural/lifecycle regression.
- [ ] Validate pointwise forward, dX, dW and dB sequentially, releasing each native/candidate pair and error temporaries before the next component.
- [ ] Record the exact validation or timing stage for OOM/errors.
- [ ] Preserve current FP32-parameter/BF16-autocast numerical semantics and 2% relative-L2 threshold.
- [ ] Seed each case from campaign seed plus stable case identity so grouped bisection cannot change inputs.
- [ ] Run CPU tests and RTX 5090 batch 1/2/3 plus bounded high-resolution smoke.
- [ ] Commit `fix: stream pointwise validation components`.

## Task 5: Emit evidence JSON and local policy YAML

**Primary files:** profiling API/CLI, output writers, policy-delta synthesis, progress/reporting tests.

- [ ] Replace schema-v1 generated profile output with immutable evidence JSON plus a stable policy YAML.
- [ ] Hash exact persisted evidence bytes, publish evidence first, and atomically replace the referencing policy; test injected failure between writes.
- [ ] Emit numerical-failure and valid-loser tombstones, omit infrastructure/OOM tombstones and blanket defaults.
- [ ] Validate the provisional overlay layered over bundled/shared policy. If rejected, publish a negative-only overlay and record that its bundled fallthrough was not the tested effective policy.
- [ ] Make external local policy fall through to bundled/shared layers for unmatched contexts.
- [ ] Replace the public result with `policy`, `evidence`, `ProfilingArtifacts`, both paths, and two-file `save()` semantics. Reject evidence JSON as an optimization policy with a migration error.
- [ ] Run focused/full/package gates and one RTX 5090 Small32 smoke; commit `feat: emit profiling evidence and policy overlays`.

## Task 6: Documentation, migration, and final integration

**Primary files:** README, optimization docs, SM evidence docs, package design, installed-package tests.

- [ ] Document policy/evidence separation, compact YAML format, confidence, inference assumptions, and reference opt-out.
- [ ] Document SM86/SM89 counterexamples and the intended RTX PRO 6000 behavior without unsupported whole-model speed claims.
- [ ] Replace schema-v1 examples and generated-output instructions.
- [ ] Verify TorchScript/export/checkpoints remain unaffected.
- [ ] Run full CPU/GPU-appropriate/style/build/Twine/archive/install gates.
- [ ] Commit `docs: explain optimization policy and evidence v2`.

## Final review and integration

- [ ] Independent integrated review from the branch base, with policy behavior matrices for SM86/89/120/unknown SM.
- [ ] Resolve findings and rerun only affected gates plus the final full suite.
- [ ] Merge to `main`, push, and clean the worktree/branch as authorized for this session.
