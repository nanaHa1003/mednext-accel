# Maximum-Only Batch Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make profiling locate one maximum feasible batch per workload and benchmark only that selected batch.

**Architecture:** `search_batches()` becomes a pure capacity-boundary search: a supplied maximum is tried first and otherwise bounded by exponential growth, with binary refinement only after an infeasible upper bound exists. The runner converts the result into a zero-or-one-batch workload selection; discovery, global kernel deduplication, grouped execution, and context-specific validation remain unchanged downstream.

**Tech Stack:** Python 3.10+, PyTorch, Triton, PyYAML, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-22-maximum-batch-search-design.md`

## Global Constraints

- Remove `strategy`, `dense_until`, and `transition_batches`; do not keep deprecated aliases.
- Reject unknown `batch_search` YAML keys before CUDA discovery or profiling subprocesses.
- A supplied `maximum` is a positive upper bound and is probed first.
- An omitted `maximum` uses exponential growth followed by binary refinement.
- Treat feasibility as monotonic and apply `memory_fraction` to successful probes.
- Only `maximum_feasible` enters kernel profiling and whole-model validation.
- Intermediate search probes remain diagnostic evidence and never become profile measurements.
- Keep optimization profile schema version 1, the profile CLI surface, BF16-only profiling, global kernel deduplication, grouped failure isolation, and context-specific validation intact.
- Generated profiles make a measured claim only for the selected batch.
- Do not add explicit multi-batch profiling, caches, resume support, Typer, new dtypes, or new model families.

---

### Task 1: Replace Dense Exploration With Maximum Boundary Search

**Files:**
- Modify: `src/mednext_accel/profiling/batch_search.py`
- Modify: `src/mednext_accel/profiling/campaign.py`
- Modify: `tests/profiling/test_batch_search.py`
- Modify: `tests/profiling/test_campaign.py`
- Modify: `tests/profiling/test_api.py`
- Modify: `tests/tools/test_cli.py`

**Interfaces:**
- Produces `BatchSearch(memory_fraction: float = 0.90, maximum: int | None = None)`.
- Preserves `BatchSearchResult(probes, maximum_feasible)` as diagnostic search evidence.
- Produces `search_batches(config, *, total_vram_bytes, probe)` without transition sampling.

- [ ] **Step 1: Write failing tests for a supplied maximum**

Replace the dense-search test with these exact behavioral cases:

```python
def result_for_boundary(boundary: int, calls: list[int]):
    def probe(batch: int) -> ProbeResult:
        calls.append(batch)
        return ProbeResult(batch, batch <= boundary, batch * 100)

    return probe


def test_feasible_maximum_is_the_only_probe() -> None:
    calls: list[int] = []
    result = search_batches(
        BatchSearch(maximum=16),
        total_vram_bytes=10_000,
        probe=result_for_boundary(16, calls),
    )
    assert calls == [16]
    assert result.probed_batches == (16,)
    assert result.maximum_feasible == 16


def test_infeasible_maximum_is_refined_by_binary_search() -> None:
    calls: list[int] = []
    result = search_batches(
        BatchSearch(maximum=16),
        total_vram_bytes=10_000,
        probe=result_for_boundary(13, calls),
    )
    assert calls == [16, 1, 8, 12, 14, 13]
    assert result.probed_batches == (1, 8, 12, 13, 14, 16)
    assert result.maximum_feasible == 13
```

- [ ] **Step 2: Write failing tests for automatic upper-bound discovery**

```python
def test_unbounded_search_uses_exponential_growth_then_binary_search() -> None:
    calls: list[int] = []
    result = search_batches(
        BatchSearch(),
        total_vram_bytes=10_000,
        probe=result_for_boundary(13, calls),
    )
    assert calls == [1, 2, 4, 8, 16, 12, 14, 13]
    assert result.probed_batches == (1, 2, 4, 8, 12, 13, 14, 16)
    assert result.maximum_feasible == 13
```

Retain and adapt the existing batch-one failure and memory-fraction tests. Add
`maximum=1` feasible/infeasible cases and retain the mismatched probe-result
error contract.

- [ ] **Step 3: Run the focused tests and verify RED**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_batch_search.py`

Expected: constructor failures for removed-field expectations and probe-order
assertion failures against dense enumeration.

- [ ] **Step 4: Implement the minimal boundary algorithm**

Define only:

```python
@dataclass(frozen=True, slots=True)
class BatchSearch:
    memory_fraction: float = 0.90
    maximum: int | None = None
```

Within `search_batches()`, cache `run(batch)` as today. For a supplied maximum,
probe it first and return immediately if feasible; otherwise establish batch
one as the lower bound and binary-refine. Without a maximum, establish batch
one, double until failure, then binary-refine. Return probes sorted by batch and
set `maximum_feasible` to the final known-feasible lower bound, or zero when
batch one fails.

- [ ] **Step 5: Write failing strict-YAML tests**

```python
@pytest.mark.parametrize("obsolete", ["strategy", "dense_until"])
def test_obsolete_batch_search_keys_are_rejected(obsolete: str) -> None:
    with pytest.raises(ValueError, match=rf"unknown batch_search field.*{obsolete}"):
        load_campaign({"batch_search": {obsolete: "unused"}})


def test_batch_search_accepts_only_memory_fraction_and_maximum() -> None:
    campaign = load_campaign({"batch_search": {"memory_fraction": 0.8, "maximum": 12}})
    assert campaign.batch_search == BatchSearch(memory_fraction=0.8, maximum=12)
```

Add API and CLI regressions that prove configuration validation wins before
every side effect:

```python
def fail_side_effect(*args, **kwargs):
    pytest.fail("profiling side effect before campaign validation")


def test_api_rejects_obsolete_batch_search_before_environment(monkeypatch) -> None:
    monkeypatch.setattr(api, "collect_environment", fail_side_effect)
    with pytest.raises(ValueError, match=r"unknown batch_search field.*dense_until"):
        api.profile({"batch_search": {"dense_until": 8}})


def test_cli_rejects_obsolete_batch_search_before_environment(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "obsolete.yaml"
    path.write_text("batch_search:\n  strategy: auto\n")
    monkeypatch.setattr(api, "collect_environment", fail_side_effect)
    with pytest.raises(ValueError, match=r"unknown batch_search field.*strategy"):
        cli.main(["profile", str(path), "--progress", "quiet"])
```

- [ ] **Step 6: Enforce the new campaign schema**

Before constructing `BatchSearch`, compare `raw_search.keys()` with
`{"memory_fraction", "maximum"}`. Raise one deterministic `ValueError` listing
sorted unknown fields. Remove parsing of `strategy` and `dense_until`.

- [ ] **Step 7: Run focused and compatibility tests, then commit**

Run:

```bash
PYTHONPATH=src pytest -q tests/profiling/test_batch_search.py \
  tests/profiling/test_campaign.py tests/profiling/test_api.py tests/tools/test_cli.py
ruff check src/mednext_accel/profiling/batch_search.py \
  src/mednext_accel/profiling/campaign.py tests/profiling/test_batch_search.py \
  tests/profiling/test_campaign.py tests/profiling/test_api.py tests/tools/test_cli.py
ruff format --check src/mednext_accel/profiling/batch_search.py \
  src/mednext_accel/profiling/campaign.py tests/profiling/test_batch_search.py \
  tests/profiling/test_campaign.py tests/profiling/test_api.py tests/tools/test_cli.py
```

Expected: all commands pass.

```bash
git add src/mednext_accel/profiling/batch_search.py \
  src/mednext_accel/profiling/campaign.py tests/profiling/test_batch_search.py \
  tests/profiling/test_campaign.py tests/profiling/test_api.py tests/tools/test_cli.py
git commit -m "refactor: search only for maximum feasible batch"
```

---

### Task 2: Profile Only the Selected Maximum

**Files:**
- Modify: `src/mednext_accel/profiling/runner.py:480-585`
- Modify: `src/mednext_accel/profiling/execution.py:16-136`
- Modify: `tests/profiling/test_execution.py`
- Modify: `tests/profiling/test_runner_progress.py`
- Modify: `tests/profiling/test_synthesize.py`

**Interfaces:**
- `_batches()` continues returning `(tuple[int, ...], dict[int, dict[str, object]])`, but the tuple and mapping each contain zero or one selected batch.
- `execution.BatchSearchResult` is renamed to `WorkloadBatchSelection` to distinguish it from the boundary-search result; its existing `batches` and `reference_steps` fields remain zero-or-one collections.
- `WorkloadRun.batches` remains a tuple so downstream segmentation stays compatible, with a new invariant of length at most one for production searches.

- [ ] **Step 1: Write a failing runner selection test**

Patch `runner.search_batches` to call probes for batches 16, 1, 8, 12, 14,
and 13, then return the real batch-search result with maximum 13:

```python
def fake_search(config, *, total_vram_bytes, probe):
    probes = tuple(probe(batch) for batch in (16, 1, 8, 12, 14, 13))
    return BoundarySearchResult(probes, maximum_feasible=13)


model_results = {
    batch: {"status": "ok", "step_ms": float(batch), "peak_bytes": batch * 100}
    for batch in (1, 8, 12, 13)
}
model_results.update(
    {
        14: {"status": "oom", "message": "CUDA out of memory"},
        16: {"status": "oom", "message": "CUDA out of memory"},
    }
)
monkeypatch.setattr(runner, "search_batches", fake_search)
monkeypatch.setattr(runner, "_invoke", lambda payload: model_results[payload["batch"]])

batches, references = runner._batches(campaign, workload, total_vram, progress=reporter)
assert batches == (13,)
assert references == {13: model_results[13]}
assert [event.message for event in reporter.events if event.stage == "batch-search"][
    -1
] == "maximum feasible batch: 13"
```

Add the zero-feasible counterpart asserting `batches == ()`,
`references == {}`, and final message `no feasible batch`.

- [ ] **Step 2: Run the runner tests and verify RED**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_runner_progress.py -k 'batch_search'`

Expected: `_batches()` returns every feasible diagnostic probe and emits no
final selection message.

- [ ] **Step 3: Select only `maximum_feasible` in `_batches()`**

After `search_batches()` returns, construct:

```python
selected = result.maximum_feasible
batches = () if selected == 0 else (selected,)
references = {} if selected == 0 else {selected: model_results[selected]}
```

Emit exactly one final `status` event for the selected maximum or infeasible
outcome. Keep one `advance` event per diagnostic probe and keep the total
unknown.

- [ ] **Step 4: Rename the planner callback record and enforce zero-or-one batches**

Rename `execution.BatchSearchResult` to `WorkloadBatchSelection`. In its
`__post_init__`, reject `len(batches) > 1` and reject mismatched reference keys:

```python
if len(self.batches) > 1:
    raise ValueError("workload batch selection must contain at most one batch")
if set(self.reference_steps) != set(self.batches):
    raise ValueError("reference steps must match the selected batch")
```

Update runner imports and planner tests. Preserve empty infeasible workloads.

- [ ] **Step 5: Prove intermediate probes never become experiments**

Build two workload selections with different maxima and one shared pointwise
shape:

```python
selections = {
    "none": WorkloadBatchSelection((3,), {3: reference_step()}),
    "all-expansion": WorkloadBatchSelection((5,), {5: reference_step()}),
}
```

Assert every planned case has batch 3 or 5, no cases use diagnostic batches 1,
2, or 4, and each workload has exactly one validation endpoint. Update the
primary orchestration contract to expect one batch per context. Tests that
exercise generic segmentation with synthetic multiple batches remain in
`test_validation.py`; do not route production search through them.

- [ ] **Step 6: Run integration tests and commit**

Run:

```bash
PYTHONPATH=src pytest -q tests/profiling/test_execution.py \
  tests/profiling/test_runner_progress.py tests/profiling/test_synthesize.py \
  tests/profiling/test_validation.py
ruff check src/mednext_accel/profiling/runner.py \
  src/mednext_accel/profiling/execution.py tests/profiling
ruff format --check src/mednext_accel/profiling/runner.py \
  src/mednext_accel/profiling/execution.py tests/profiling
```

Expected: all commands pass.

```bash
git add src/mednext_accel/profiling/runner.py \
  src/mednext_accel/profiling/execution.py tests/profiling/test_execution.py \
  tests/profiling/test_runner_progress.py tests/profiling/test_synthesize.py
git commit -m "refactor: profile only the maximum feasible batch"
```

---

### Task 3: Align Documentation and Verify the Complete Workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/optimization.md`
- Modify: `docs/superpowers/specs/2026-09-21-profile-driven-optimization-design.md`
- Modify only when verification exposes a defect: affected source and regression test.

- [ ] **Step 1: Replace dense-search documentation**

Update the original design's adaptive-search section to point to the new
maximum-only design and describe the final algorithm. Update public YAML examples
to:

```yaml
batch_search:
  memory_fraction: 0.90
  maximum: 16  # optional upper bound
```

Explain that a feasible supplied maximum requires one model probe, an omitted
maximum uses exponential plus binary search, and only the selected maximum is
kernel-profiled. State that rerunning with another maximum is how users collect
evidence for another batch.

- [ ] **Step 2: Remove stale non-historical references**

Run:

```bash
rg -n "dense_until|strategy: (auto|explicit)|transition_batches" \
  README.md docs/optimization.md docs/superpowers/specs src tests
```

Expected: matches remain only where the new design explains removed fields.
Historical implementation plans may retain their original text.

- [ ] **Step 3: Run the complete CPU and package gate**

Use worktree-local scratch because system `/tmp` may be full:

```bash
export TMPDIR="$PWD/dist/maximum-search-check/tmp"
export TORCHINDUCTOR_CACHE_DIR="$PWD/dist/maximum-search-check/inductor"
export XDG_CACHE_HOME="$PWD/dist/maximum-search-check/cache"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src pytest -q
ruff check .
ruff format --check .
python -m build --outdir dist/maximum-search-check/packages
python -m twine check dist/maximum-search-check/packages/*
```

Expected: all commands exit zero; ONNX and CUDA-only tests may report their
existing optional skips.

- [ ] **Step 4: Run a bounded RTX 5090 campaign**

Create a scratch campaign using the current YAML schema:

```yaml
preset: mednext-v1
compile_mode: default
workloads:
  - variant: small
    spatial: [32, 32, 32]
    dtypes: [bfloat16]
    phases: [training]
    checkpointing: none
    in_channels: 1
    out_channels: 3
batch_search:
  memory_fraction: 0.90
  maximum: 2
objective: balanced
```

Run the profile command with plain progress and scratch `XDG_CACHE_HOME`. On the
RTX 5090, batch 2 is expected to be feasible, so verify:

- batch 2 is the only model reference probe;
- the final search status is `maximum feasible batch: 2`;
- every kernel measurement and generated rule uses batch 2;
- the kernel plan contains exactly five healthy-path groups;
- whole-model validation contains exactly one endpoint;
- schema version, environment provenance, and execution counts remain valid.

If the environment makes batch 2 infeasible, verify the binary result instead
and record the selected batch; do not change production behavior to force the
expected outcome.

- [ ] **Step 5: Clean artifacts, run final checks, and commit**

Remove only the owned `dist/maximum-search-check` tree and generated caches.
Confirm `git status --short` lists only intended tracked documentation or fixes.

```bash
git add README.md docs/optimization.md \
  docs/superpowers/specs/2026-09-21-profile-driven-optimization-design.md
git commit -m "docs: explain maximum-only batch profiling"
```

If verification required a code fix, add its regression and source files to the
same commit only when they form one coherent correction; otherwise commit the
fix separately before the documentation commit.
