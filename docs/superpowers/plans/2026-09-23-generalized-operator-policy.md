# Generalized Operator Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize accelerated operator selection beyond exact model contexts while making profiling comparisons and optimization reports trustworthy.

**Architecture:** Preserve the policy-v2 layering and existing kernels. First fix resolution, reporting, campaign validation, candidate fairness, and deterministic ranking. Then benchmark actual adaptive operators and synthesize compact work/channel ranges, with full-model runs retained as the publication sentinel.

**Tech Stack:** Python 3.10+, PyTorch 2.6+, Triton, PyYAML, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-09-23-generalized-operator-policy-design.md`

## Global Constraints

- Preserve model parameter names, state-dict compatibility, checkpoint import, checkpointing behavior, and export-native execution.
- Generalized custom dispatch is limited to NVIDIA CUDA training, BF16, 3D contiguous NCDHW, depthwise K3, and regular pointwise K1.
- Keep external policy → exact SM → shared NVIDIA → reference precedence.
- Explicit reference selections remain tombstones; guarded custom selections may fall through to lower layers.
- Generated and bundled operator rules do not depend on model variant or checkpoint policy.
- Do not add a learned runtime model or a new runtime dependency.

---

### Task 1: Correct resolution and make reports complete

**Files:**
- Modify: `src/mednext_accel/optimization/policy_resolver.py`
- Modify: `src/mednext_accel/optimization/descriptors.py`
- Modify: `src/mednext_accel/optimization/report.py`
- Modify: `src/mednext_accel/ops/adaptive.py`
- Modify: `src/mednext_accel/models/mednext_v1.py`
- Modify: `src/mednext_accel/policies/sm120.yaml`
- Test: `tests/optimization/test_policy_resolver.py`
- Test: `tests/models/test_mednext_v1.py`
- Test: `tests/optimization/test_bundled_policies.py`

**Interfaces:**
- Produces: `PolicyDecision.disposition` with `custom`, `native`, or `unwrapped`.
- Produces: 2D/3D `OperatorDescriptor` and `ExecutionContext` reporting support.
- Preserves: every existing `PolicyDecision` field and policy-v2 document.

- [ ] **Step 1: Write failing resolver tests**

Add tests proving a guarded external custom choice falls through to a valid SM
or shared choice, an explicit reference tombstone stops, later rules in the
same policy are not searched, and all-guarded resolution returns the
highest-priority diagnostic.

- [ ] **Step 2: Run resolver tests and confirm the current early reference return fails**

Run: `PYTHONPATH=src python -m pytest tests/optimization/test_policy_resolver.py -q`

- [ ] **Step 3: Implement guard fallthrough**

Keep one retained guarded decision while iterating policy layers. Return valid
custom/native selections immediately, return explicit reference selections
immediately, and return the retained guard only if no lower layer succeeds.

- [ ] **Step 4: Write failing report and 2D tests**

Test that auto/reference reports enumerate adaptive and unwrapped convolution
modules, identify `head.conv` as unwrapped, accept `(N,C,H,W)` for a 2D model,
compute 2D spatial volume, and reject ranks inconsistent with the model.

- [ ] **Step 5: Implement report disposition and dimensional descriptors**

Generalize descriptor tuple validation to length two or three, use
`math.prod()` for spatial volume, enumerate `named_modules()` exactly once, and
create reference decisions for unwrapped Conv2d/Conv3d/transpose modules.

- [ ] **Step 6: Remove the unreachable SM120 `32→3` pointwise rule**

Delete the rule whose real output head is ConvTranspose3d and add a regression
asserting no bundled rule claims that head is accelerated.

- [ ] **Step 7: Run focused tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/optimization/test_policy_resolver.py tests/models/test_mednext_v1.py tests/optimization/test_bundled_policies.py -q`

Commit: `fix: make policy resolution and reports truthful`

### Task 2: Make profiler comparisons fair and deterministic

**Files:**
- Modify: `src/mednext_accel/profiling/campaign.py`
- Modify: `src/mednext_accel/profiling/matrix.py`
- Modify: `src/mednext_accel/profiling/grouped.py`
- Modify: `src/mednext_accel/profiling/synthesize.py`
- Test: `tests/profiling/test_campaign.py`
- Test: `tests/profiling/test_grouped.py`
- Test: `tests/profiling/test_synthesize.py`

**Interfaces:**
- Produces: `KernelCase.comparison_identifier`, excluding implementation and parameters.
- Preserves: `KernelCase.identifier` as the unique evidence/subprocess identity.
- Produces: objective-aware deterministic winner selection.

- [ ] **Step 1: Write failing strict-workload and seed tests**

Reject sorted unknown `workloads[index]` keys. Prove alternative
implementations/parameters share a seed, while geometry, phase, dtype, and
batch changes do not.

- [ ] **Step 2: Run focused tests and observe the silent-key and unequal-seed failures**

Run: `PYTHONPATH=src python -m pytest tests/profiling/test_campaign.py tests/profiling/test_grouped.py -q`

- [ ] **Step 3: Implement strict parsing and comparison identity**

Allow only `variant`, `spatial`, `dtypes`, `checkpointing`, `in_channels`, and
`out_channels` in workload mappings. Hash campaign seed with comparison
identity for tensors, while response matching continues to use full identity.

- [ ] **Step 4: Write failing multi-candidate synthesis tests**

Cover invalid alternative plus valid winner, fastest valid candidate regardless
of input order, objective-specific ordering, deterministic ties, all complete
losers becoming reference, and incomplete-only evidence remaining a gap.

- [ ] **Step 5: Implement candidate-local rejection and deterministic ranking**

Filter to complete valid candidates, apply the existing objective thresholds,
rank qualifying candidates by the objective and stable implementation/parameter
keys, and emit reference only from conclusive evaluated evidence.

- [ ] **Step 6: Run focused tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/profiling/test_campaign.py tests/profiling/test_grouped.py tests/profiling/test_synthesize.py -q`

Commit: `fix: compare profiling candidates fairly`

### Task 3: Benchmark integrated adaptive operators

**Files:**
- Create: `src/mednext_accel/profiling/operator_benchmark.py`
- Modify: `src/mednext_accel/profiling/runner.py`
- Modify: `src/mednext_accel/profiling/matrix.py`
- Modify: `src/mednext_accel/ops/_triton/depthwise.py`
- Test: `tests/profiling/test_operator_benchmark.py`
- Test: `tests/ops/test_phase_routing.py`

**Interfaces:**
- Produces: an internal integrated benchmark callable accepting a `KernelCase`, compile mode, seed, and requested gradient mask.
- Produces: the existing scalar measurement fields, plus benchmark-kind and gradient-mask evidence.
- Preserves: subprocess isolation and raw diagnostic probes.

- [ ] **Step 1: Write failing CPU planning tests for integrated payloads**

Assert payloads include compile mode, benchmark kind, and the exact requested
input/weight/bias gradient mask without changing evidence identity.

- [ ] **Step 2: Write failing CUDA gradient-mask equivalence tests**

For regular, downsample, and transpose adaptive depthwise modules, compare
forward and requested gradients with native PyTorch under BF16 autocast in
eager and `torch.compile(fullgraph=True)` execution. Assert unused gradients
remain absent.

- [ ] **Step 3: Implement gradient-mask-aware custom autograd**

Use `ctx.needs_input_grad` in every custom backward, invoke only requested
custom/native phases, and retain native bias reduction semantics.

- [ ] **Step 4: Implement the integrated operator benchmark**

Construct native and adaptive modules with identical FP32 parameters, feed the
same seeded BF16 input/upstream gradient, validate outputs and requested
gradients, then time each through the selected compile mode. Keep allocations
streamed so reference and candidate validation tensors are not retained
together across components.

- [ ] **Step 5: Wire integrated measurements into campaign execution**

Use integrated timings as dispatch evidence; keep raw kernel timings explicitly
labelled diagnostic. Preserve grouped subprocess isolation and progress counts.

- [ ] **Step 6: Run focused CPU/CUDA tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/profiling/test_operator_benchmark.py tests/ops/test_phase_routing.py tests/profiling/test_runner_progress.py -q`

Commit: `feat: profile integrated adaptive operators`

### Task 4: Synthesize and ship generalized operator rules

**Files:**
- Modify: `src/mednext_accel/optimization/policy.py`
- Modify: `src/mednext_accel/optimization/policy_resolver.py`
- Modify: `src/mednext_accel/profiling/synthesize.py`
- Modify: `src/mednext_accel/policies/shared-nvidia.yaml`
- Modify: `src/mednext_accel/policies/sm86.yaml`
- Modify: `src/mednext_accel/policies/sm89.yaml`
- Modify: `src/mednext_accel/policies/sm120.yaml`
- Test: `tests/optimization/test_policy_v2.py`
- Test: `tests/optimization/test_bundled_policies.py`
- Test: `tests/profiling/test_synthesize.py`
- Test: `tests/profiling/test_policy_overlay.py`

**Interfaces:**
- Produces: numeric policy conditions `in_channels` and `out_channels`.
- Produces: compact positive work/channel ranges and exact negative tombstones.
- Preserves: exact `channels` matching for exceptions and old policy files.

- [ ] **Step 1: Write failing parser/resolver tests for numeric channel ranges**

Test inclusive bounds, exact numeric shorthand, scope narrowing, contradiction
rejection, and coexistence with exact `channels` tombstones.

- [ ] **Step 2: Implement numeric channel facts and parsing**

Add `in_channels` and `out_channels` to numeric conditions and resolver facts.
Keep `model_family`, `variant`, and `checkpointing` readable but stop emitting
them from generated operator policies.

- [ ] **Step 3: Write failing generalized synthesis tests**

Prove duplicate model/checkpoint projections collapse, depthwise positive
observations form bounded work intervals, negative observations split those
intervals, recipes emit `parameters: auto`, pointwise channel regions are
bounded, and a singleton never becomes unbounded.

- [ ] **Step 4: Implement compact synthesis**

Emit exact tombstones first. Segment depthwise winners along `work` or
`reduction_work`; collapse only contiguous observations with the same winner.
Emit bounded pointwise input/output-channel and work regions only where all
observed points qualify. Leave incomplete regions as policy gaps.

- [ ] **Step 5: Migrate bundled positive rules conservatively**

Replace redundant exact depthwise rows with reviewed work ranges while keeping
SM-specific launch exceptions and negative tombstones first. Generalize only
pointwise regions supported by RTX 5090, L40S, and RTX A6000 evidence. Add tests
for B3, unseen aligned channels, unseen spatial sizes inside a supported work
range, interval boundaries, and unknown-SM shared behavior.

- [ ] **Step 6: Run policy/profiler tests and commit**

Run: `PYTHONPATH=src python -m pytest tests/optimization tests/profiling/test_synthesize.py tests/profiling/test_policy_overlay.py -q`

Commit: `feat: generalize operator optimization policies`

### Task 5: Validate, document, and integrate

**Files:**
- Modify: `README.md`
- Modify: `docs/optimization.md`
- Modify: `docs/benchmarks/sm120-profile.md`
- Modify: package tests if artifact contents change

**Interfaces:**
- Documents: generalized domain, reference tombstones, profile artifacts, report dispositions, and full-model sentinel semantics.

- [ ] **Step 1: Run an RTX 5090 integrated grid and full-model sentinel**

Cover Base at 128³ for feasible batches plus representative channel and
spatial boundary points. Record selection regret against native/custom
measurements. Do not widen a bundled rule across a measured loser.

- [ ] **Step 2: Update bundled SM120 policy from measured evidence**

Apply only evidence-supported rule boundaries and rerun the full-model candidate
against reference for every supported checkpoint policy used by the release
sentinel.

- [ ] **Step 3: Update user documentation**

Explain that unmeasured tuples use generalized operator rules, local policies
are operator-based rather than model/checkpoint-bound, and reports distinguish
custom/native/unwrapped execution.

- [ ] **Step 4: Run final verification**

Run:

```bash
PYTHONPATH=src python -m pytest -q --basetemp=/tmp/mednext-generalized-final
python -m ruff check src tests
python -m build
python -m twine check dist/*
```

Inspect wheel/sdist contents and run the installed-wheel smoke test in an
isolated target directory.

- [ ] **Step 5: Independent final review, merge, and push**

Review the full branch against this plan, resolve critical/important findings,
merge `feature/generalized-operator-policy` into `main`, rerun the full suite on
the merged tree, push `main`, and remove the worktree.
