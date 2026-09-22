# Profiler Deduplication Implementation Plan

> Historical design record. Runtime profile/schema examples and output instructions
> in this document are superseded by [policy/evidence v2](../specs/2026-09-22-policy-evidence-v2-design.md).
> Use the current [optimization guide](../../optimization.md) for runnable YAML,
> CLI commands, and Python APIs.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce automatic profiling from one subprocess per operator shape and checkpoint context to a campaign-wide deduplicated kernel matrix, grouped subprocess execution, and boundary-only whole-model validation with useful counts and ETA.

**Architecture:** Build immutable context-free kernel cases after context-specific batch searches, execute them in five failure-isolated groups per batch, then project shared results back into each workload. Consecutive batches with identical full decision signatures form validation segments whose endpoints determine whether the segment remains eligible for profile synthesis.

**Tech Stack:** Python 3.10+, PyTorch, Triton, Rich, argparse, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-21-profiler-deduplication-design.md`

## Global Constraints

- Keep `mednext-accel profile [campaign.yaml] [--progress auto|plain|quiet]` compatible.
- Keep optimization profile schema version 1 and all existing profile loaders compatible.
- Preserve BF16 training coverage; new dtype support is outside this change.
- Batch search remains specific to model variant, spatial shape, checkpoint context, channels, and compile mode.
- Isolated kernel evidence must not be keyed by variant, output-class count, checkpoint context, or module path.
- A failed validation representative invalidates its complete decision segment.
- Rich is used only for presentation; orchestration remains terminal-independent.
- Do not add persistent caches, resume support, Typer, or new model families.

---

### Task 1: Define Context-Free Kernel Cases and Global Deduplication

**Files:**
- Create: `src/mednext_accel/profiling/matrix.py`
- Create: `tests/profiling/test_matrix.py`

**Interfaces:**
- Consumes: `Workload` from `mednext_accel.profiling.campaign` and the existing pointwise/depthwise shape tuples discovered by `runner.py`.
- Produces: `KernelCaseKey`, `KernelCase`, `KernelGroup`, `build_workload_cases()`, `deduplicate_cases()`, and `group_cases()`.

- [ ] **Step 1: Write failing tests for case identity and workload-independent deduplication**

```python
from mednext_accel.profiling.campaign import Workload
from mednext_accel.profiling.matrix import build_workload_cases, deduplicate_cases


def test_same_kernel_across_checkpoint_contexts_is_measured_once() -> None:
    none = Workload("mednext_v1", "base", (128, 128, 128), checkpointing="none")
    expansion = Workload("mednext_v1", "base", (128, 128, 128), checkpointing="all-expansion")
    shapes = ((32, 64, (128, 128, 128)),)
    first = build_workload_cases(none, (1,), pointwise_shapes=shapes, depthwise_shapes=())
    second = build_workload_cases(expansion, (1,), pointwise_shapes=shapes, depthwise_shapes=())

    unique, references = deduplicate_cases((first, second))

    assert len(unique) == 1
    assert references[0] == references[1]


def test_batch_phase_shape_and_launch_parameters_remain_in_identity() -> None:
    workload = Workload("mednext_v1", "base", (128, 128, 128))
    cases = build_workload_cases(
        workload,
        (1, 2),
        pointwise_shapes=((32, 64, (128, 128, 128)),),
        depthwise_shapes=(("regular", 32, 3, (128, 128, 128)),),
    )
    assert len({case.key for case in cases}) == 6
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_matrix.py`

Expected: collection fails with `ModuleNotFoundError: mednext_accel.profiling.matrix`.

- [ ] **Step 3: Implement immutable case records and deterministic launch parameters**

```python
@dataclass(frozen=True, slots=True, order=True)
class KernelCaseKey:
    family: str
    direction: str
    phase: str
    batch: int
    spatial_shape: tuple[int, int, int]
    in_channels: int
    out_channels: int
    kernel_size: int
    dtype: str
    implementation: str
    parameters: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class KernelCase:
    key: KernelCaseKey

    @property
    def identifier(self) -> str:
        document = json.dumps(asdict(self.key), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(document.encode()).hexdigest()[:16]
```

Implement the current launch-parameter formulas exactly for pointwise, regular
dX/dW, downsample dX, and transpose dW. `deduplicate_cases()` returns sorted
unique cases plus one tuple of case keys per input workload case tuple. It must
never inspect checkpointing or variant.

- [ ] **Step 4: Add failing group-category assertions, then implement five-way grouping**

```python
groups = group_cases(cases)
assert {group.category for group in groups} == {
    "pointwise",
    "regular-dx",
    "regular-dw",
    "downsample-dx",
    "transpose-dw",
}
assert all(len({case.key.batch for case in group.cases}) == 1 for group in groups)
```

Define `KernelGroup(category: str, batch: int, cases: tuple[KernelCase, ...])` and
sort groups by `(batch, category)` and cases by key.

- [ ] **Step 5: Run focused tests and commit**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_matrix.py`

Expected: all tests pass.

```bash
git add src/mednext_accel/profiling/matrix.py tests/profiling/test_matrix.py
git commit -m "feat: deduplicate profiling kernel cases"
```

---

### Task 2: Add Grouped Child Execution With Recursive Failure Isolation

**Files:**
- Create: `src/mednext_accel/profiling/grouped.py`
- Create: `tests/profiling/test_grouped.py`
- Modify: `src/mednext_accel/profiling/runner.py:92-296`
- Test: `tests/profiling/test_runner_progress.py`

**Interfaces:**
- Consumes: `KernelCase` and `KernelGroup` from Task 1; existing `_pointwise_probe()`, `_depthwise_probe()`, and `_invoke()` behavior.
- Produces: `case_payload()`, `run_group_with_bisection(group, invoke, on_attempt=None) -> dict[str, dict[str, object]]`, and child protocol `{"kind": "kernel_group", "cases": [...]}`.

- [ ] **Step 1: Write failing tests for grouped success and recursive bisection**

```python
def make_case(out_channels: int) -> KernelCase:
    return KernelCase(
        KernelCaseKey(
            family="pointwise_conv3d",
            direction="regular",
            phase="training",
            batch=1,
            spatial_shape=(16, 16, 16),
            in_channels=8,
            out_channels=out_channels,
            kernel_size=1,
            dtype="bfloat16",
            implementation="pointwise_gemm_per_sample",
            parameters=(),
        )
    )


def ok_results(payload):
    return {
        "status": "ok",
        "results": [
            {"case_id": item["case_id"], "status": "ok", "valid": True} for item in payload["cases"]
        ],
    }


def test_successful_group_uses_one_child() -> None:
    group = KernelGroup("pointwise", 1, tuple(make_case(value) for value in (8, 16, 32)))
    calls = []

    def invoke(payload):
        calls.append(payload)
        return ok_results(payload)

    results = run_group_with_bisection(group, invoke)
    assert len(calls) == 1
    assert len(results) == 3


def test_failed_group_bisects_until_the_bad_case_is_isolated() -> None:
    cases = tuple(make_case(value) for value in (8, 16, 32))
    group = KernelGroup("pointwise", 1, cases)
    bad_id = cases[1].identifier
    calls = []
    attempt_events = []

    def invoke(payload):
        calls.append(tuple(item["case_id"] for item in payload["cases"]))
        if any(item["case_id"] == bad_id for item in payload["cases"]):
            return {"status": "infrastructure_error", "message": "child crashed"}
        return ok_results(payload)

    results = run_group_with_bisection(
        group,
        invoke,
        on_attempt=lambda event, physical, resolved: attempt_events.append(
            (event, physical, resolved)
        ),
    )
    assert results[bad_id]["status"] == "infrastructure_error"
    assert results[cases[0].identifier]["status"] == "ok"
    assert results[cases[2].identifier]["status"] == "ok"
    assert len(calls) == 5
    assert sum(item[1] for item in attempt_events if item[0] == "scheduled") == 4
    assert sum(item[1] for item in attempt_events if item[0] == "completed") == 5
    assert sum(item[2] for item in attempt_events if item[0] == "completed") == 3
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_grouped.py`

Expected: collection fails because `profiling.grouped` does not exist.

- [ ] **Step 3: Implement payload serialization and parent-side bisection**

```python
def response_matches(group: KernelGroup, result: Mapping[str, object]) -> bool:
    items = result.get("results")
    if not isinstance(items, list):
        return False
    identifiers = [item.get("case_id") for item in items if isinstance(item, dict)]
    expected = [case.identifier for case in group.cases]
    return len(identifiers) == len(items) == len(set(identifiers)) and set(identifiers) == set(
        expected
    )


def run_group_with_bisection(
    group: KernelGroup,
    invoke: Invoke,
    on_attempt: Callable[[Literal["scheduled", "completed"], int, int], None] | None = None,
) -> dict[str, dict[str, object]]:
    def execute(current: KernelGroup, *, already_scheduled: bool):
        if on_attempt is not None and not already_scheduled:
            on_attempt("scheduled", 1, 0)
        result = invoke({"kind": "kernel_group", "cases": [case_payload(c) for c in current.cases]})
        if result.get("status") == "ok" and not response_matches(current, result):
            result = {
                "status": "infrastructure_error",
                "message": "group response has missing or duplicate case identifiers",
            }
        terminal = result.get("status") == "ok" or len(current.cases) == 1
        if on_attempt is not None:
            on_attempt("completed", 1, len(current.cases) if terminal else 0)
        if result.get("status") == "ok":
            return {str(item["case_id"]): item for item in result["results"]}
        if len(current.cases) == 1:
            case = current.cases[0]
            return {case.identifier: {"case_id": case.identifier, **result}}
        midpoint = len(current.cases) // 2
        left = replace(current, cases=current.cases[:midpoint])
        right = replace(current, cases=current.cases[midpoint:])
        return execute(left, already_scheduled=False) | execute(right, already_scheduled=False)

    return execute(group, already_scheduled=True)
```

Reject duplicate or missing returned case identifiers as an infrastructure
error rather than silently accepting incomplete evidence. Callback arguments
are `(event, physical_attempt_delta, resolved_case_delta)`. They let the runner
grow the attempt total before bisection children execute, advance it after every
child response, and count logical cases only when a successful group or failed
singleton resolves them.

- [ ] **Step 4: Write a failing child-protocol test and implement sequential execution**

Patch `_pointwise_probe` and `_depthwise_probe`, call `_child()` with two
serialized cases, and assert ordered `results` plus one cleanup after each case.
Implement `_kernel_group_probe()` in `runner.py`: dispatch each case, convert OOM
and ordinary exceptions to per-case results, then run `gc.collect()` and
`torch.cuda.empty_cache()` in `finally`. Interpreter crashes, timeouts, and
malformed responses remain parent-side bisection failures.

- [ ] **Step 5: Run grouped and child tests and commit**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_grouped.py tests/profiling/test_runner_progress.py`

Expected: all tests pass.

```bash
git add src/mednext_accel/profiling/grouped.py src/mednext_accel/profiling/runner.py \
  tests/profiling/test_grouped.py tests/profiling/test_runner_progress.py
git commit -m "feat: batch profiling cases with failure isolation"
```

---

### Task 3: Build a Campaign-Wide Execution Plan

**Files:**
- Create: `src/mednext_accel/profiling/execution.py`
- Create: `tests/profiling/test_execution.py`
- Modify: `src/mednext_accel/profiling/runner.py:299-427`

**Interfaces:**
- Consumes: `Campaign`, `Workload`, Task 1 case builders, existing shape discovery and `_batches()`.
- Produces: `WorkloadShapes`, `BatchSearchResult`, `WorkloadRun`, `ExecutionPlan`, `deduplicate_workloads()`, and `build_execution_plan()`.

- [ ] **Step 1: Write failing tests for exact workload deduplication and shared kernel references**

```python
def test_duplicate_workloads_search_once() -> None:
    workload = Workload("mednext_v1", "base", (128, 128, 128))
    campaign = Campaign("mednext-v1", (workload, workload), BatchSearch(maximum=2))
    shapes = WorkloadShapes(
        pointwise=((32, 64, (128, 128, 128)),),
        depthwise=(),
    )
    calls = []
    plan = build_execution_plan(
        campaign,
        discover=lambda workload: shapes,
        search=lambda workload: (
            calls.append(workload)
            or BatchSearchResult((1, 2), {1: reference_step(), 2: reference_step()})
        ),
    )
    assert len(calls) == 1
    assert len(plan.workloads) == 1


def test_checkpoint_workloads_search_separately_but_share_kernel_cases() -> None:
    none = Workload("mednext_v1", "base", (128, 128, 128), checkpointing="none")
    expansion = replace(none, checkpointing="all-expansion")
    campaign = Campaign("mednext-v1", (none, expansion), BatchSearch(maximum=2))
    shapes = WorkloadShapes(
        pointwise=((32, 64, (128, 128, 128)),),
        depthwise=(),
    )
    plan = build_execution_plan(
        campaign,
        discover=lambda workload: shapes,
        search=lambda workload: BatchSearchResult(
            (1, 2), {1: reference_step(), 2: reference_step()}
        ),
    )
    assert len(plan.workloads) == 2
    assert len(plan.kernel_cases) == len(plan.workloads[0].case_keys)
    assert plan.workloads[0].case_keys == plan.workloads[1].case_keys
```

At the top of the test module, define the complete reference fixture used
above:

```python
def reference_step() -> dict[str, object]:
    return {"status": "ok", "step_ms": 10.0, "peak_bytes": 1_000}
```

- [ ] **Step 2: Run tests and verify the missing interface failure**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_execution.py`

Expected: collection fails because `profiling.execution` does not exist.

- [ ] **Step 3: Implement immutable workload results and the two-pass planner**

```python
@dataclass(frozen=True, slots=True)
class WorkloadShapes:
    pointwise: tuple[tuple[int, int, tuple[int, int, int]], ...]
    depthwise: tuple[tuple[str, int, int, tuple[int, int, int]], ...]


@dataclass(frozen=True, slots=True)
class BatchSearchResult:
    batches: tuple[int, ...]
    reference_steps: Mapping[int, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class WorkloadRun:
    workload: Workload
    shapes: WorkloadShapes
    batches: tuple[int, ...]
    reference_steps: Mapping[int, Mapping[str, object]]
    case_keys: tuple[KernelCaseKey, ...]


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    workloads: tuple[WorkloadRun, ...]
    kernel_cases: tuple[KernelCase, ...]
    kernel_groups: tuple[KernelGroup, ...]
    requested_case_count: int
```

The first pass deduplicates exact `Workload` values, discovers shapes, emits the
static shape plan, and performs every context-specific batch search. The second
pass builds each workload's cases from its discovered shapes and feasible
batches, creates the global unique case set, and groups it.
`requested_case_count` is the sum of per-workload case references before
deduplication.

- [ ] **Step 4: Move shape discovery behind injected planner callbacks**

Keep meta-tensor discovery behavior unchanged and expose:

```python
def discover_workload_shapes(workload: Workload) -> WorkloadShapes:
    return WorkloadShapes(
        pointwise=_pointwise_shapes(workload),
        depthwise=_depthwise_shapes(workload),
    )
```

Pure planning tests inject discovery and search, so they remain CPU-only.

- [ ] **Step 5: Run execution tests and commit**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_execution.py tests/profiling/test_batch_search.py`

Expected: all tests pass.

```bash
git add src/mednext_accel/profiling/execution.py src/mednext_accel/profiling/runner.py \
  tests/profiling/test_execution.py
git commit -m "feat: plan deduplicated profiling campaigns"
```

---

### Task 4: Segment Decisions and Select Validation Boundaries

**Files:**
- Create: `src/mednext_accel/profiling/validation.py`
- Create: `tests/profiling/test_validation.py`

**Interfaces:**
- Consumes: context-specific `Measurement` sequences and `candidate_wins()`.
- Produces: `DecisionSegment`, `decision_signature()`, `segment_decisions()`, `validation_batches()`, and `apply_validation_results()`.

- [ ] **Step 1: Write failing tests for signatures, gaps, and endpoint selection**

```python
def test_equal_consecutive_signatures_form_one_segment() -> None:
    gemm = (
        (
            "pointwise_conv3d",
            "regular",
            "training",
            (16, 16, 16),
            8,
            16,
            "pointwise_gemm_per_sample",
            (),
        ),
    )
    decisions = {1: gemm, 2: gemm, 3: gemm}
    segments = segment_decisions(decisions)
    assert segments == (DecisionSegment((1, 2, 3), gemm),)
    assert validation_batches(segments) == (1, 3)


def test_sample_gap_and_decision_change_split_segments() -> None:
    native = (("pointwise_conv3d", "regular", "training", (16, 16, 16), 8, 16, "reference", ()),)
    gemm = (
        (
            "pointwise_conv3d",
            "regular",
            "training",
            (16, 16, 16),
            8,
            16,
            "pointwise_gemm_per_sample",
            (),
        ),
    )
    decisions = {1: native, 2: gemm, 4: gemm}
    segments = segment_decisions(decisions)
    assert tuple(segment.batches for segment in segments) == ((1,), (2,), (4,))
    assert validation_batches(segments) == (1, 2, 4)
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_validation.py`

Expected: collection fails because `profiling.validation` does not exist.

- [ ] **Step 3: Implement deterministic complete decision signatures**

For every measurement in a batch, include this sorted tuple entry:

```python
(
    measurement.family,
    measurement.direction,
    measurement.phase,
    measurement.spatial_shape,
    measurement.in_channels,
    measurement.out_channels,
    measurement.implementation if candidate_wins(measurement, objective) else "reference",
    measurement.parameters if candidate_wins(measurement, objective) else (),
)
```

The complete tuple is the signature. Segment only when batch numbers differ by
exactly one and signatures match.

- [ ] **Step 4: Write a failing rejection test and implement segment invalidation**

```python
def measurement(batch: int) -> Measurement:
    return Measurement(
        family="pointwise_conv3d",
        direction="regular",
        phase="training",
        implementation="pointwise_gemm_per_sample",
        batch=batch,
        spatial_shape=(16, 16, 16),
        in_channels=8,
        out_channels=16,
        dtype="bfloat16",
        checkpointing="none",
        reference_ms=2.0,
        candidate_ms=1.0,
        reference_peak_bytes=1_000,
        candidate_peak_bytes=900,
        valid=True,
    )


def test_failed_endpoint_invalidates_every_measurement_in_segment() -> None:
    measurements = tuple(measurement(batch) for batch in (1, 2, 3))
    signature = decision_signature((measurements[0],), "throughput")
    segment = DecisionSegment((1, 2, 3), signature)
    updated = apply_validation_results(
        measurements,
        (segment,),
        accepted={1: True, 3: False},
    )
    assert {item.valid for item in updated} == {False}
```

Measurements in unrelated segments retain their original validity. A segment
is accepted only when every selected representative has an explicit `True`.

- [ ] **Step 5: Run validation tests and commit**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_validation.py`

Expected: all tests pass.

```bash
git add src/mednext_accel/profiling/validation.py tests/profiling/test_validation.py
git commit -m "feat: validate profiling decision boundaries"
```

---

### Task 5: Orchestrate Deduplicated Measurement and Boundary Validation

**Files:**
- Modify: `src/mednext_accel/profiling/runner.py:386-580`
- Modify: `src/mednext_accel/profiling/synthesize.py:13-175`
- Test: `tests/profiling/test_runner_progress.py`
- Test: `tests/profiling/test_synthesize.py`

**Interfaces:**
- Consumes: `ExecutionPlan`, grouped raw results, and decision segmentation from Tasks 1–4.
- Produces: compatible `run_campaign(campaign, progress=None) -> tuple[Measurement, ...]` plus internal `execute_campaign() -> CampaignRun`.

- [ ] **Step 1: Replace the old orchestration test with a failing deduplication contract**

Construct `none` and `all-expansion` workloads with the same Base 128³ shape.
Patch search to return batches `(1, 2)`, shape discovery to return one pointwise
shape plus one regular, one downsample, and one transpose depthwise shape, and
group execution to return a valid candidate result for every case. Record every
group and validation payload. Assert:

```python
run = execute_campaign(campaign, progress=reporter)

assert len(groups) == 10  # five categories for each of two feasible batches
assert {group.category for group in groups} == {
    "pointwise",
    "regular-dx",
    "regular-dw",
    "downsample-dx",
    "transpose-dw",
}
assert validations == {
    "none": [1, 2],
    "all-expansion": [1, 2],
}
assert {item.checkpointing for item in run.measurements} == {
    "none",
    "all-expansion",
}
assert run.statistics.requested_case_count == 20
assert run.statistics.kernel_case_count == 10
```

This proves the two checkpoint contexts share the same ten kernel executions,
while the whole-model validation stays context-specific. The test must not call
the old one-child-per-case path.

- [ ] **Step 2: Run the runner test and verify it fails against sequential orchestration**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_runner_progress.py`

Expected: the group-count assertion fails because the runner still invokes one
child per case per workload.

- [ ] **Step 3: Execute groups once and convert raw results to context measurements**

Add:

```python
def measurement_from_result(
    case: KernelCase,
    result: Mapping[str, object],
    *,
    checkpointing: str,
) -> Measurement:
    ok = result.get("status") == "ok"
    return Measurement(
        family=case.key.family,
        direction=case.key.direction,
        phase=case.key.phase,
        implementation=case.key.implementation,
        batch=case.key.batch,
        spatial_shape=case.key.spatial_shape,
        in_channels=case.key.in_channels,
        out_channels=case.key.out_channels,
        dtype=case.key.dtype,
        checkpointing=checkpointing,
        reference_ms=float(result.get("reference_ms", 0.0)),
        candidate_ms=float(result.get("candidate_ms", 0.0)),
        reference_peak_bytes=int(result.get("reference_peak_bytes", 0)),
        candidate_peak_bytes=int(result.get("candidate_peak_bytes", 0)),
        valid=ok and bool(result.get("valid", False)),
        parameters=case.key.parameters,
    )
```

Failed cases produce `valid=False` and zero candidate timing. For each
`WorkloadRun`, map case keys through the shared result table and materialize
measurements carrying that workload's checkpoint context.

- [ ] **Step 4: Validate only decision-segment representatives**

For each workload: calculate segments; synthesize a provisional profile; invoke
candidate model probes only for `validation_batches(segments)`; compare to the
matching batch-search reference result; invalidate complete rejected segments;
append finalized measurements. Never reuse whole-model results across variants
or checkpoint contexts.

- [ ] **Step 5: Preserve the public runner return type while exposing statistics**

```python
@dataclass(frozen=True, slots=True)
class ExecutionStatistics:
    requested_case_count: int
    kernel_case_count: int
    kernel_group_count: int
    whole_model_validation_count: int


@dataclass(frozen=True, slots=True)
class CampaignRun:
    measurements: tuple[Measurement, ...]
    statistics: ExecutionStatistics
```

`execute_campaign()` returns `CampaignRun`. The existing `run_campaign()` calls
it and returns `.measurements`; `profiling.api` uses the internal richer result.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
PYTHONPATH=src pytest -q tests/profiling/test_runner_progress.py \
  tests/profiling/test_synthesize.py tests/profiling/test_api.py
```

Expected: all tests pass.

```bash
git add src/mednext_accel/profiling/runner.py src/mednext_accel/profiling/synthesize.py \
  tests/profiling/test_runner_progress.py tests/profiling/test_synthesize.py \
  tests/profiling/test_api.py
git commit -m "perf: deduplicate profiling campaign execution"
```

---

### Task 6: Report Search Progress, Exact Execution Counts, Rates, and ETA

**Files:**
- Modify: `src/mednext_accel/profiling/progress.py:10-143`
- Modify: `src/mednext_accel/profiling/runner.py:386-580`
- Test: `tests/profiling/test_progress.py`
- Test: `tests/profiling/test_runner_progress.py`

**Interfaces:**
- Extends `ProgressEvent` with `secondary_completed` and `secondary_total`.
- Reports completed probes and mean seconds per probe while batch-search total is unknown.
- Reports exact physical group count and logical experiment count after planning.

- [ ] **Step 1: Write a failing unknown-total progress test**

```python
def test_plain_unknown_total_reports_count_and_average_without_eta() -> None:
    times = iter((0.0, 20.0, 50.0))
    output = StringIO()
    reporter = PlainProgressReporter(output, clock=lambda: next(times))

    reporter.emit(ProgressEvent("stage_start", "batch-search"))
    reporter.emit(ProgressEvent("advance", "batch-search", completed=1))
    reporter.emit(ProgressEvent("advance", "batch-search", completed=2))

    text = output.getvalue()
    assert "1/?" in text
    assert "2/?" in text
    assert "25.0 s/probe" in text
    assert "ETA" not in text
```

- [ ] **Step 2: Write a failing dual-count progress test**

```python
def test_plain_group_progress_reports_physical_and_logical_counts() -> None:
    times = iter((0.0, 70.0))
    output = StringIO()
    reporter = PlainProgressReporter(output, clock=lambda: next(times))

    reporter.emit(ProgressEvent("stage_start", "kernel-groups", total=20))
    reporter.emit(
        ProgressEvent(
            "advance",
            "kernel-groups",
            completed=7,
            total=20,
            secondary_completed=71,
            secondary_total=192,
        )
    )

    text = output.getvalue()
    assert "groups 7/20" in text
    assert "experiments 71/192" in text
    assert "10.0 s/group" in text
    assert "ETA 00:02:10" in text
```

- [ ] **Step 3: Run focused tests and verify both new contracts fail**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_progress.py`

Expected: the first test has no `1/?` or average, and constructing the second
event rejects the two new fields.

- [ ] **Step 4: Implement stage-local timing in the plain and Rich reporters**

Track each stage's start time and most recent completed count. For an unknown
total, render `completed/?`, elapsed time, and `elapsed / completed` with no
percentage or ETA. For a known total, render count, percentage, mean time per
physical group, and ETA. When secondary counts are present, render them as
`experiments completed/total` without using them to calculate ETA.

Implement a small Rich `ProgressColumn` that reads the same task fields so Rich
and plain mode expose the same facts. Pass `total=None` during batch search so
Rich uses a pulsing bar and does not show a remaining-time estimate.

- [ ] **Step 5: Emit planned and measured counts from campaign orchestration**

Emit one `advance` event for every completed batch probe. After the two-pass
plan exists, emit a status line containing:

```text
kernel plan: <groups> planned groups, <unique> unique experiments, <requested> requested references
```

During grouped execution, `completed/total` is the physical group count and the
secondary pair is the cumulative completed unique cases and total unique cases.
If recursive bisection schedules extra child calls, increase the group total
before running those children so the displayed completed count never exceeds
the total. After decision segmentation, emit the now-exact count:

```text
validation plan: <validations> whole-model validations
```

Whole-model validation uses its own stage and independently measured average.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
PYTHONPATH=src pytest -q tests/profiling/test_progress.py \
  tests/profiling/test_runner_progress.py
```

Expected: all tests pass, including plain, Rich, unknown-total, and exact-plan
assertions.

```bash
git add src/mednext_accel/profiling/progress.py src/mednext_accel/profiling/runner.py \
  tests/profiling/test_progress.py tests/profiling/test_runner_progress.py
git commit -m "feat: report exact profiling plans and rates"
```

---

### Task 7: Record Execution Statistics and Explain the New Campaign Model

**Files:**
- Modify: `src/mednext_accel/profiling/api.py:17-137`
- Modify: `src/mednext_accel/profiling/synthesize.py:58-156`
- Modify: `tests/profiling/test_api.py`
- Modify: `tests/profiling/test_synthesize.py`
- Modify: `README.md`
- Modify: `docs/optimization.md`

**Interfaces:**
- Passes `CampaignRun.statistics` through profile synthesis without changing
  schema version 1.
- Adds an `execution` object beneath existing profile provenance.

- [ ] **Step 1: Write a failing provenance test**

```python
def test_synthesize_profile_records_deduplicated_execution_counts() -> None:
    execution = {
        "kernel_case_count": 192,
        "kernel_group_count": 20,
        "whole_model_validation_count": 2,
        "deduplicated_reference_count": 384,
    }
    profile = synthesize_profile(
        (),
        name="sm89-local",
        sm=(8, 9),
        objective="balanced",
        execution=execution,
    )
    assert profile.provenance["execution"] == execution
```

- [ ] **Step 2: Run the test and verify the new argument is rejected**

Run: `PYTHONPATH=src pytest -q tests/profiling/test_synthesize.py`

Expected: `TypeError: synthesize_profile() got an unexpected keyword argument 'execution'`.

- [ ] **Step 3: Thread execution statistics through API and synthesis**

Add `execution: Mapping[str, int] | None = None` to `synthesize_profile()` and
include a primitive copy at `profile.provenance.execution` when supplied.
Change the lazy API wrapper to expose `execute_campaign()` internally, then use
its measurements and statistics in `profile()`. Keep public `run_campaign()`
returning only measurements. Serialize these exact fields:

```python
execution = {
    "kernel_case_count": statistics.kernel_case_count,
    "kernel_group_count": statistics.kernel_group_count,
    "whole_model_validation_count": statistics.whole_model_validation_count,
    "deduplicated_reference_count": statistics.requested_case_count - statistics.kernel_case_count,
}
```

- [ ] **Step 4: Document what is counted and why runtime is shorter**

Update the README profiling section and `docs/optimization.md` with one compact
example of the two-phase display. Explain that `experiments` are unique kernel
cases, `groups` are child processes, `requested references` include duplicate
checkpoint contexts, and whole-model checks occur only at decision-segment
boundaries. State that batch search remains context-specific and therefore is
not globally deduplicated.

- [ ] **Step 5: Run CPU checks and commit**

Run:

```bash
PYTHONPATH=src pytest -q tests/profiling/test_api.py \
  tests/profiling/test_synthesize.py tests/profiling/test_progress.py
ruff check src/mednext_accel/profiling tests/profiling
```

Expected: all tests and lint checks pass.

```bash
git add src/mednext_accel/profiling/api.py src/mednext_accel/profiling/synthesize.py \
  tests/profiling/test_api.py tests/profiling/test_synthesize.py README.md \
  docs/optimization.md
git commit -m "docs: explain deduplicated profiling campaigns"
```

---

### Task 8: Verify CPU Packaging and a Bounded CUDA Campaign

**Files:**
- Modify only if verification exposes a defect; add the smallest regression
  test beside the affected module before applying a fix.

- [ ] **Step 1: Run the complete CPU and packaging gate**

```bash
ruff check .
ruff format --check .
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src pytest -q
rm -rf dist build
python -m build
python -m twine check dist/*
```

Expected: every command exits zero.

- [ ] **Step 2: Run focused grouped CUDA correctness checks**

On a CUDA host, run the grouped child protocol on one small case from each
category: pointwise, regular dX, regular dW, downsample dX, and transpose dW.
Compare each returned validity and timing field with the existing single-case
probe. Then send one malformed case between two valid cases and verify recursive
bisection preserves both valid results and reports only the malformed case as an
infrastructure error.

- [ ] **Step 3: Run one bounded end-to-end profile campaign**

Create `/tmp/mednext-accel-dedup-smoke.yaml`:

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
  strategy: auto
  dense_until: 1
  maximum: 1
  memory_fraction: 0.90
objective: balanced
```

Run:

```bash
mednext-accel profile /tmp/mednext-accel-dedup-smoke.yaml --progress plain
```

Verify the output shows a completed batch-search count, exact group and
experiment totals, non-negative averages and ETA, and a separate whole-model
validation stage. Open the generated JSON and verify schema version 1,
measurements, existing environment provenance, and all four execution counts.

- [ ] **Step 4: Verify the built wheel in a clean virtual environment**

```bash
python -m venv --system-site-packages /tmp/mednext-accel-dedup-venv
/tmp/mednext-accel-dedup-venv/bin/pip install --no-deps dist/*.whl
/tmp/mednext-accel-dedup-venv/bin/mednext-accel profile --help
```

Expected: installation and CLI discovery succeed without importing CUDA-only
profiling modules during `--help`.

- [ ] **Step 5: Commit verification fixes only when needed**

For each discovered defect, first add a failing regression test, implement the
minimal fix, rerun the affected focused test, then rerun Steps 1–4. Commit fixes
with a message that names the corrected behavior. If no defect is found, do not
create an empty verification commit.
