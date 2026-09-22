# Profile-Driven Optimization Implementation Plan

> Historical design record. Runtime profile/schema examples and output instructions
> in this document are superseded by [policy/evidence v2](../specs/2026-09-22-policy-evidence-v2-design.md).
> Use the current [optimization guide](../../optimization.md) for runnable YAML,
> CLI commands, and Python APIs.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the RTX-5090 exact-table policy with an out-of-the-box operator/phase/implementation registry, deterministic profile resolver, optimized-by-default model factories, and a standalone profiler that automatically synthesizes one reusable configuration.

**Architecture:** Model factories load an immutable profile bundle and replace eligible native operators with adaptive wrappers that preserve the original parameters and state-dict paths. At eager execution or `torch.compile` tracing, wrappers resolve a registered implementation from an operator descriptor and execution context; package profiles provide exact-SM and generic NVIDIA rules, while the separate profiling subsystem benchmarks isolated candidates and whole models to generate the same schema.

**Tech Stack:** Python 3.10+, PyTorch 2.6+, Triton through `torch.library.triton_op`, PyYAML 6+, pytest, Ruff, JSON package data, argparse console entry point.

**Spec:** `docs/superpowers/specs/2026-09-21-profile-driven-optimization-design.md`

## Global Constraints

- `optimization="auto"` is the default for every public MedNeXt factory.
- `optimization="reference"` is the only public reference-mode name; remove `torch`, `conservative`, and runtime `autotune` policies.
- Public configuration inputs must be YAML-compatible strings or mappings.
- Model weights, parameter objects, tensor layouts, and state-dict keys must remain compatible with the reference and official checkpoint mappings.
- Model construction and first forward must never benchmark or write a cache.
- Unknown batches and spatial shapes must resolve through interval, formula, or family-default rules rather than an empty selection.
- Unknown NVIDIA SM architectures use `generic-nvidia` and emit one warning.
- Cross-SM external profiles load and execute after one warning.
- Only correctness constraints may force an unsupported-operation reference guard.
- Approximate implementations require explicit `allow_approximate: true`.
- Checkpointing and batch size remain user choices.
- `torch.jit.trace` and ONNX export must resolve export-safe native branches.
- GPU-less CI validates schema, resolution, model identity, and export; bundled GPU profiles require separate CUDA evidence.

## File Map

Create these focused modules:

- `src/mednext_accel/optimization/descriptors.py`: stable operator and execution-context values.
- `src/mednext_accel/optimization/implementations.py`: implementation metadata and registry.
- `src/mednext_accel/optimization/schema.py`: versioned profile dataclasses, primitive parsing, and validation.
- `src/mednext_accel/optimization/profiles.py`: bundled/external loading and SM selection.
- `src/mednext_accel/optimization/resolver.py`: precedence, matching, formulas, warnings, and decisions.
- `src/mednext_accel/optimization/report.py`: per-operator decisions and model inspection reports.
- `src/mednext_accel/profiles/generic-nvidia.json`: generic RTX-5090-derived defaults.
- `src/mednext_accel/profiles/sm120.json`: measured SM 12.0 rules and overrides.
- `src/mednext_accel/ops/adaptive.py`: adaptive pointwise/depthwise wrappers and model installation.
- `src/mednext_accel/profiling/campaign.py`: preset and YAML campaign parsing.
- `src/mednext_accel/profiling/batch_search.py`: VRAM-aware batch probing strategy.
- `src/mednext_accel/profiling/discovery.py`: model/operator enumeration.
- `src/mednext_accel/profiling/benchmark.py`: isolated subprocess benchmarks.
- `src/mednext_accel/profiling/whole_model.py`: compiled end-to-end validation.
- `src/mednext_accel/profiling/synthesize.py`: candidate selection and interval/formula generation.
- `src/mednext_accel/profiling/api.py`: Python `profile()` API and result persistence.
- `src/mednext_accel/profiling/cli.py`: `mednext-accel profile` command.

Modify these integration points:

- `src/mednext_accel/models/mednext_v1.py`: factory argument and inspection method.
- `src/mednext_accel/models/__init__.py` and `src/mednext_accel/__init__.py`: public exports.
- `src/mednext_accel/ops/pointwise.py`: resolver-driven GEMM dispatch.
- `src/mednext_accel/ops/_triton/depthwise.py`: profile-supplied phase parameters and generalized split formulas.
- `src/mednext_accel/export.py`: explicit export-safe context.
- `pyproject.toml` and `requirements.txt`: PyYAML dependency, package data, and console script.
- `README.md`, `docs/optimization.md`, and `docs/development/package-design.md`: new default behavior and profiling workflow.

Remove after migration:

- `src/mednext_accel/optimization/api.py`
- `src/mednext_accel/optimization/policy.py`
- `src/mednext_accel/optimization/autotune.py`
- `tests/optimization/test_policy.py`
- `tests/optimization/test_autotune.py`
- `benchmarks/autotune.py`

## Task 1: Define Stable Operator Descriptors and the Implementation Registry

**Files:**

- Create: `src/mednext_accel/optimization/descriptors.py`
- Create: `src/mednext_accel/optimization/implementations.py`
- Replace: `src/mednext_accel/optimization/report.py`
- Create: `tests/optimization/test_descriptors.py`
- Create: `tests/optimization/test_implementations.py`

**Interfaces:**

- Produces `OperatorDescriptor`, `ExecutionContext`, `ImplementationSpec`, `ImplementationRegistry`, `Decision`, and `OptimizationReport`.
- Later tasks consume stable string identifiers from these types; no module path or Python class name enters a profile document.

- [ ] **Step 1: Write descriptor canonicalization tests**

```python
from mednext_accel.optimization.descriptors import ExecutionContext, OperatorDescriptor


def test_descriptor_has_a_stable_primitive_identity() -> None:
    descriptor = OperatorDescriptor(
        family="depthwise_conv3d",
        direction="regular",
        in_channels=32,
        out_channels=32,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        dilation=(1, 1, 1),
        groups=32,
    )

    assert descriptor.to_primitive()["family"] == "depthwise_conv3d"
    assert descriptor.signature == ("depthwise_conv3d", "regular", 32, 32, (3, 3, 3), (1, 1, 1))


def test_execution_context_records_runtime_resolution_inputs() -> None:
    context = ExecutionContext(
        phase="training",
        device_type="cuda",
        sm=(12, 0),
        total_vram_bytes=32 * 2**30,
        dtype="bfloat16",
        batch_size=3,
        spatial_shape=(128, 128, 128),
        model_family="mednext_v1",
        variant="base",
        checkpointing="all-expansion",
    )

    assert context.spatial_volume == 128**3
    assert context.total_vram_gib == 32
```

- [ ] **Step 2: Run the descriptor tests and confirm they fail**

Run: `pytest tests/optimization/test_descriptors.py -q`

Expected: import failure because `descriptors.py` does not exist.

- [ ] **Step 3: Implement frozen descriptor and context values**

Use frozen, slotted dataclasses and string literals. Normalize tuple-like inputs in `__post_init__`, reject nonpositive channels/batches/spatial values, and expose only JSON-compatible output from `to_primitive()`.

```python
@dataclass(frozen=True, slots=True)
class OperatorDescriptor:
    family: OperatorFamily
    direction: str
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int, int]
    stride: tuple[int, int, int]
    padding: tuple[int, int, int]
    dilation: tuple[int, int, int]
    groups: int
    role: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    phase: ExecutionPhase
    device_type: str
    sm: tuple[int, int] | None
    total_vram_bytes: int
    dtype: str
    batch_size: int
    spatial_shape: tuple[int, int, int]
    model_family: str
    variant: str
    checkpointing: str
    export: bool = False
    allow_approximate: bool = False
```

- [ ] **Step 4: Write implementation registry tests**

```python
import pytest

from mednext_accel.optimization.implementations import (
    ImplementationRegistry,
    ImplementationSpec,
)


def test_registry_rejects_duplicate_identifiers() -> None:
    registry = ImplementationRegistry()
    spec = ImplementationSpec(
        identifier="reference",
        version=1,
        families=("pointwise_conv3d",),
        phases=("training", "inference", "export"),
        equivalence="exact",
        export_safe=True,
    )
    registry.register(spec)

    with pytest.raises(ValueError, match="reference"):
        registry.register(spec)


def test_approximate_implementation_requires_opt_in() -> None:
    spec = ImplementationSpec(
        identifier="gelu_tanh",
        version=1,
        families=("gelu",),
        phases=("inference",),
        equivalence="approximate",
        export_safe=True,
    )

    assert not spec.is_allowed(allow_approximate=False)
    assert spec.is_allowed(allow_approximate=True)


def test_registry_accepts_a_future_mednext_v2_operator_family() -> None:
    registry = ImplementationRegistry()
    registry.register(
        ImplementationSpec(
            identifier="reference_grn",
            version=1,
            families=("global_response_norm",),
            phases=("training", "inference", "export"),
            equivalence="exact",
            export_safe=True,
        )
    )

    assert registry.resolve_metadata("reference_grn").families == ("global_response_norm",)
```

- [ ] **Step 5: Implement registry metadata and report types**

`ImplementationRegistry.resolve_metadata(identifier)` must raise a profile-facing error naming an unknown implementation. Register these initial identifiers without importing Triton eagerly:

- `reference`
- `pointwise_gemm_per_sample`
- `triton_depthwise_dx`
- `triton_split_dw`
- `triton_transpose_split_dw`
- `triton_downsample_dx`
- `gelu_tanh`

Replace the old shape-table `BackendSelections` report with:

```python
@dataclass(frozen=True, slots=True)
class Decision:
    descriptor: OperatorDescriptor
    phase: str
    implementation: str
    parameters: Mapping[str, JsonValue]
    profile: str
    rule: str
    confidence: Literal["measured", "interpolated", "extrapolated", "default", "guard"]
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    profile: str
    context: ExecutionContext
    decisions: tuple[Decision, ...]
    warnings: tuple[str, ...] = ()
```

- [ ] **Step 6: Verify Task 1**

Run: `pytest tests/optimization/test_descriptors.py tests/optimization/test_implementations.py -q`

Expected: all tests pass.

Run: `ruff check src/mednext_accel/optimization tests/optimization/test_descriptors.py tests/optimization/test_implementations.py`

Expected: no errors.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/mednext_accel/optimization/descriptors.py \
  src/mednext_accel/optimization/implementations.py \
  src/mednext_accel/optimization/report.py \
  tests/optimization/test_descriptors.py \
  tests/optimization/test_implementations.py
git commit -m "feat: add optimization implementation registry"
```

## Task 2: Add the Versioned Profile Schema and Loaders

**Files:**

- Create: `src/mednext_accel/optimization/schema.py`
- Create: `src/mednext_accel/optimization/profiles.py`
- Create: `tests/optimization/test_schema.py`
- Create: `tests/optimization/test_profiles.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

**Interfaces:**

- Consumes implementation identifiers from Task 1.
- Produces `OptimizationProfile`, `ProfileRule`, `ProfileOverride`, `load_profile()`, and `select_profile()` for the resolver.

- [ ] **Step 1: Add failing schema parsing and validation tests**

```python
from mednext_accel.optimization.schema import parse_profile


def test_profile_parses_json_compatible_mapping() -> None:
    profile = parse_profile(
        {
            "schema_version": 1,
            "profile": {
                "name": "test-sm120",
                "target": {"vendor": "nvidia", "sm": [12, 0]},
                "provenance": {"gpu": "test", "torch": "test", "cuda": "test"},
            },
            "defaults": {
                "pointwise_conv3d": {"training": {"implementation": "reference", "parameters": {}}}
            },
            "rules": [],
            "overrides": [],
        }
    )

    assert profile.name == "test-sm120"
    assert profile.target_sm == (12, 0)


def test_unknown_schema_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        parse_profile({"schema_version": 2})
```

- [ ] **Step 2: Run schema tests and confirm they fail**

Run: `pytest tests/optimization/test_schema.py -q`

Expected: import failure because `schema.py` does not exist.

- [ ] **Step 3: Implement strict primitive parsing**

The parser must:

- reject missing required keys with a path such as `rules[2].match.batch.min`;
- reject unknown implementation identifiers;
- normalize SM, shapes, ranges, and parameters to immutable values;
- require unique rule and override IDs;
- validate `min <= max` for numeric ranges;
- validate semantic and export declarations through the registry;
- keep raw measurements optional and opaque to runtime resolution.

Use these core values:

```python
@dataclass(frozen=True, slots=True)
class NumericRange:
    minimum: int | float | None = None
    maximum: int | float | None = None


@dataclass(frozen=True, slots=True)
class ProfileSelection:
    implementation: str
    parameters: FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class ProfileRule:
    identifier: str
    family: str
    direction: str | None
    phases: FrozenJsonMapping
    batch: NumericRange
    spatial_volume: NumericRange
    in_channels: int | None
    out_channels: int | None
    dtype: str | None
    total_vram_gib: NumericRange
    confidence: str
```

- [ ] **Step 4: Add file and mapping loader tests**

Test JSON and YAML paths, primitive mappings, invalid suffixes, missing files, and cross-SM metadata. Use `tmp_path`; do not depend on package profiles yet.

```python
def test_yaml_path_and_mapping_produce_the_same_profile(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(PROFILE_MAPPING))

    assert load_profile(path) == load_profile(PROFILE_MAPPING)
```

- [ ] **Step 5: Implement profile loading and dependency metadata**

`load_profile(source)` accepts `str | PathLike[str] | Mapping[str, object]`. A string equal to `auto` or `reference` is handled by factory integration later; other strings are file paths. JSON output is canonical; YAML is accepted through `PyYAML>=6`.

Add `PyYAML>=6` to project dependencies and `requirements.txt`. Do not import CUDA or Triton while loading a profile.

- [ ] **Step 6: Verify Task 2**

Run: `pytest tests/optimization/test_schema.py tests/optimization/test_profiles.py -q`

Expected: all tests pass.

Run: `python -m build && python -m twine check dist/*`

Expected: wheel and source distribution build and metadata validation pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add pyproject.toml requirements.txt \
  src/mednext_accel/optimization/schema.py \
  src/mednext_accel/optimization/profiles.py \
  tests/optimization/test_schema.py \
  tests/optimization/test_profiles.py
git commit -m "feat: add versioned optimization profiles"
```

## Task 3: Implement Deterministic Resolution and Bootstrap Profiles

**Files:**

- Create: `src/mednext_accel/optimization/resolver.py`
- Create: `src/mednext_accel/profiles/generic-nvidia.json`
- Create: `src/mednext_accel/profiles/sm120.json`
- Create: `src/mednext_accel/profiles/__init__.py`
- Modify: `src/mednext_accel/optimization/profiles.py`
- Create: `tests/optimization/test_resolver.py`
- Create: `tests/optimization/test_bundled_profiles.py`

**Interfaces:**

- Consumes parsed profiles and registered implementations.
- Produces `OptimizationResolver.resolve(descriptor, context, phase) -> Decision` and `resolve_all()`.
- The adaptive wrappers in Task 4 receive one immutable resolver.

- [ ] **Step 1: Write precedence, interpolation, and warning tests**

```python
def test_batch_three_uses_the_batch_two_to_four_rule(sm120_resolver) -> None:
    decision = sm120_resolver.resolve(
        pointwise_descriptor(32, 64),
        cuda_context(batch=3, spatial=(128, 128, 128)),
        phase="training",
    )

    assert decision.implementation == "pointwise_gemm_per_sample"
    assert decision.confidence == "interpolated"


def test_unknown_sm_uses_generic_profile_and_warns_once(profile_registry) -> None:
    resolver = profile_registry.resolver_for(sm=(9, 0))
    first = resolver.resolve(pointwise_descriptor(32, 64), cuda_context(sm=(9, 0)), "training")
    second = resolver.resolve(pointwise_descriptor(64, 32), cuda_context(sm=(9, 0)), "training")

    assert first.profile == "generic-nvidia"
    assert first.warning is not None
    assert second.profile == "generic-nvidia"
```

Also test:

- external override beats external rule;
- external rule beats bundled exact-SM rule;
- exact-SM rule beats generic family default;
- cross-SM external profile warns but executes;
- approximate decisions are rejected without opt-in;
- export requests reject non-export-safe implementations;
- every supported CUDA BF16 descriptor receives a decision rather than `None`.

- [ ] **Step 2: Run resolver tests and confirm they fail**

Run: `pytest tests/optimization/test_resolver.py tests/optimization/test_bundled_profiles.py -q`

Expected: import or file-not-found failures.

- [ ] **Step 3: Implement ordered deterministic matching**

`OptimizationResolver` evaluates exact overrides, ordered rules, and family defaults. Matching must be pure: no device query, timing, file access, mutation, or warning emission inside `resolve()`. Return warnings in decisions; `ProfileRegistry.resolver_for()` is responsible for emitting a process-once Python warning.

Implement split formulas as schema data:

```python
def scale_work(
    anchor_value: int,
    anchor_work: int,
    context_work: int,
    *,
    minimum: int,
    maximum: int,
    multiple: int,
) -> int:
    raw = round(anchor_value * context_work / anchor_work)
    clamped = min(max(raw, minimum), maximum)
    return max(multiple, round(clamped / multiple) * multiple)
```

Formula output must be included in `Decision.parameters` and validated against implementation constraints.

- [ ] **Step 4: Encode the generic and SM120 bootstrap profiles**

Translate the current RTX 5090 selections into schema data, with these behavior changes:

- pointwise `32->64 @ 128^3` and `64->32 @ 128^3` use one batch interval from 2 through 4;
- batch 3 therefore uses GEMM without an exact entry;
- batch 6 keeps the measured lower-memory reference selection on 32 GiB and permits a higher-VRAM rule to choose GEMM;
- regular depthwise dW splits use work-scaling anchors instead of exact batch lookup;
- regular, transpose, and downsample family defaults always name an implementation;
- unsupported implementation constraints produce a reported correctness guard;
- inference and export have explicit decisions rather than falling through training rules.

The generic profile inherits safe SM120 rules and family defaults but contains no SM-specific exact override. The SM120 profile contains current measured shapes, batch intervals, and RTX 5090 full-model overrides.

- [ ] **Step 5: Verify package-data loading**

Build a wheel, install it into the existing release-test virtual environment pattern, and load `auto` profiles outside the source tree. Assert both bundled JSON files are present and parse successfully.

- [ ] **Step 6: Verify Task 3**

Run: `pytest tests/optimization/test_resolver.py tests/optimization/test_bundled_profiles.py tests/test_installed_package.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/mednext_accel/optimization/resolver.py \
  src/mednext_accel/optimization/profiles.py \
  src/mednext_accel/profiles \
  tests/optimization/test_resolver.py \
  tests/optimization/test_bundled_profiles.py \
  tests/test_installed_package.py
git commit -m "feat: resolve optimization profiles across shapes"
```

## Task 4: Replace Exact Shape Wrappers with Adaptive Registered Operators

**Files:**

- Create: `src/mednext_accel/ops/adaptive.py`
- Modify: `src/mednext_accel/ops/pointwise.py`
- Modify: `src/mednext_accel/ops/_triton/depthwise.py`
- Modify: `src/mednext_accel/ops/depthwise.py`
- Modify: `src/mednext_accel/ops/__init__.py`
- Create: `tests/ops/test_adaptive.py`
- Modify: `tests/ops/test_pointwise.py`
- Modify: `tests/ops/test_depthwise.py`

**Interfaces:**

- Consumes `OptimizationResolver`, `OperatorDescriptor`, and `ExecutionContext`.
- Produces `install_adaptive_operators(model, resolver, model_context) -> int` and wrappers whose parameter keys match the replaced module.

- [ ] **Step 1: Write CPU reference, parameter identity, and state-dict tests**

```python
def test_installation_preserves_parameters_and_state_dict_keys() -> None:
    model = nn.Sequential(
        nn.Conv3d(32, 32, 3, padding=1, groups=32),
        nn.Conv3d(32, 64, 1),
    )
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())

    count = install_adaptive_operators(model, resolver=TEST_RESOLVER, model_context=TEST_MODEL)

    assert count == 2
    assert tuple(model.parameters()) == parameters
    assert tuple(model.state_dict()) == keys
```

Add a CPU forward/backward comparison where the resolver selects `reference`, and an inspection test showing batch 3 selects GEMM for the high-resolution pointwise descriptor without executing CUDA.

- [ ] **Step 2: Run adaptive tests and confirm they fail**

Run: `pytest tests/ops/test_adaptive.py -q`

Expected: import failure because `adaptive.py` does not exist.

- [ ] **Step 3: Implement shared execution-context construction**

Create a small pure helper that derives context from `x`, module training state, model metadata, checkpoint metadata, and resolver profile. CUDA capability and total VRAM are read only when the tensor is CUDA; CPU receives `sm=None`. Keep returned values primitive and stable for Dynamo specialization.

- [ ] **Step 4: Refactor pointwise GEMM dispatch**

Replace `selected_shapes` and `selected_batch_size` with a descriptor and resolver. The wrapper asks for a `training` decision and dispatches by implementation identifier:

```python
decision = self.resolver.resolve(self.descriptor, context, phase="training")
if decision.implementation == "pointwise_gemm_per_sample":
    return self._gemm(x)
return F.conv3d(x, self.weight, self.bias)
```

Keep evaluation and export decisions explicit. Do not silently treat `not self.training` as an unsupported fallback.

- [ ] **Step 5: Refactor depthwise phase parameters**

Move exact `_REGULAR_BATCH_SPLITS` policy data out of the Triton module. The adaptive depthwise wrapper resolves forward/dX/dW/dB decisions and passes profile parameters into the existing custom autograd operator. Generalize `regular_config()` to accept resolved `dw_splits`, `dw_block`, and `dx_block` after constraint validation.

Keep the kernel's real correctness constraints: supported dimensionality, odd cubic kernels, supported contiguous layout, same padding, dilation one, and CUDA/Triton availability. A failed constraint returns a `guard` decision and native execution.

- [ ] **Step 6: Add CUDA numerical tests for registered implementations**

Parameterize small BF16/FP32 tests over:

- regular depthwise forward/dX/dW/dB;
- transpose depthwise forward/dW;
- downsample depthwise dX;
- pointwise GEMM forward/dX/dW/dB;
- batch sizes 1 and 3;
- kernel sizes 3 and 5 where the implementation declares support.

Compare against native operators with implementation-specific tolerances declared in registry metadata.

- [ ] **Step 7: Verify Task 4**

Run: `pytest tests/ops -q`

Expected: CPU tests pass; CUDA tests pass on the workstation GPU.

Run: `ruff check src/mednext_accel/ops tests/ops`

Expected: no errors.

- [ ] **Step 8: Commit Task 4**

```bash
git add src/mednext_accel/ops tests/ops
git commit -m "feat: dispatch adaptive registered operators"
```

## Task 5: Integrate Optimized-by-Default Factories and Remove Legacy Policies

**Files:**

- Modify: `src/mednext_accel/models/mednext_v1.py`
- Modify: `src/mednext_accel/models/__init__.py`
- Modify: `src/mednext_accel/__init__.py`
- Modify: `src/mednext_accel/export.py`
- Modify: `tests/models/test_mednext_v1.py`
- Modify: `tests/export/test_torchscript.py`
- Modify: `tests/export/test_torch_export.py`
- Modify: `tests/export/test_onnx.py`
- Modify: `tests/test_public_api.py`
- Remove: `src/mednext_accel/optimization/api.py`
- Remove: `src/mednext_accel/optimization/policy.py`
- Remove: `src/mednext_accel/optimization/autotune.py`
- Remove: `tests/optimization/test_policy.py`
- Remove: `tests/optimization/test_autotune.py`

**Interfaces:**

- Consumes profile loading, resolver, and adaptive operator installation.
- Produces final factory signatures and `MedNeXtV1.explain_optimization()`.

- [ ] **Step 1: Write factory default and reference-mode tests**

```python
def test_factory_defaults_to_auto_and_reference_is_explicit() -> None:
    auto = mednext_base(in_channels=1, out_channels=3)
    reference = mednext_base(in_channels=1, out_channels=3, optimization="reference")

    assert auto.optimization_source == "auto"
    assert any(isinstance(module, AdaptivePointwise3d) for module in auto.modules())
    assert not any(isinstance(module, AdaptivePointwise3d) for module in reference.modules())
    assert tuple(auto.state_dict()) == tuple(reference.state_dict())
```

Also test mapping/path inputs, invalid `torch`/`conservative`/`autotune` strings, exact parameter identity after loading the same state dict, and the inspection report for batch 3.

- [ ] **Step 2: Run factory tests and confirm they fail**

Run: `pytest tests/models/test_mednext_v1.py tests/test_public_api.py -q`

Expected: factory rejects the new `optimization` argument.

- [ ] **Step 3: Add factory integration**

Add this YAML-compatible alias:

```python
OptimizationSource = Literal["auto", "reference"] | str | os.PathLike[str] | Mapping[str, object]
```

Thread `optimization: OptimizationSource = "auto"` through `_factory()` and all four public factories. After constructing the reference model, load the profile and install adaptive wrappers unless mode is `reference`. Store only immutable resolver/model metadata outside `state_dict`.

Add:

```python
def explain_optimization(
    self,
    *,
    input_shape: tuple[int, int, int, int, int],
    dtype: str | torch.dtype,
    device: str | torch.device,
) -> OptimizationReport:
    context = execution_context_for_shape(
        self,
        input_shape=input_shape,
        dtype=dtype,
        device=device,
    )
    return explain_model(self, resolver=self._optimization_resolver, context=context)
```

- [ ] **Step 4: Make export context explicit**

The trace and ONNX helpers enter an export-resolution context before tracing. Assert generated graphs contain no custom MedNeXt operator domains and match the reference output. Direct `torch.jit.trace(model.eval(), example)` must continue to work because adaptive wrappers resolve eval/export-safe native forward operations.

- [ ] **Step 5: Remove legacy policy modules and migrate scripts**

Delete the old API, policy, autotune implementation, and tests. Update internal benchmark scripts that still need reference execution to construct models with `optimization="reference"`. Move reusable kernel benchmark functions needed by the new profiler into Task 7 modules rather than retaining runtime autotune code.

- [ ] **Step 6: Verify state-dict and checkpoint compatibility**

Run official and MONAI checkpoint conversion tests for both auto and reference models. Add an assertion that an official checkpoint loaded into each mode produces exact eval output equality because eval resolves native implementations.

- [ ] **Step 7: Verify Task 5**

Run: `pytest tests/models tests/checkpoints tests/export tests/test_public_api.py -q`

Expected: all tests pass; ONNX may skip when optional dependencies are absent.

Run: `pytest tests/ops tests/optimization -q`

Expected: all new registry/resolver tests pass and no legacy policy tests remain.

- [ ] **Step 8: Commit Task 5**

```bash
git add -A src/mednext_accel tests benchmarks tools
git commit -m "feat: enable profile optimization in model factories"
```

## Task 6: Add Campaign Configuration and VRAM-Aware Batch Search

**Files:**

- Create: `src/mednext_accel/profiling/__init__.py`
- Create: `src/mednext_accel/profiling/campaign.py`
- Create: `src/mednext_accel/profiling/batch_search.py`
- Create: `tests/profiling/test_campaign.py`
- Create: `tests/profiling/test_batch_search.py`

**Interfaces:**

- Produces `Campaign`, `Workload`, `BatchSearch`, `ProbeResult`, `load_campaign()`, and built-in preset expansion.
- Task 7 supplies the subprocess probe callback.

- [ ] **Step 1: Write zero-configuration and minimal-YAML campaign tests**

```python
def test_default_campaign_covers_installed_models() -> None:
    campaign = load_campaign(None)

    assert campaign.preset == "all"
    assert {workload.variant for workload in campaign.workloads} == {
        "small",
        "base",
        "medium",
        "large",
    }
    assert campaign.batch_search.strategy == "auto"
    assert campaign.batch_search.memory_fraction == 0.90
    assert campaign.batch_search.dense_until == 8


def test_minimal_yaml_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "workload.yaml"
    path.write_text(
        "preset: mednext-v1\nworkloads:\n  - variant: base\n    spatial: [128, 128, 128]\n"
    )

    campaign = load_campaign(path)

    assert len(campaign.workloads) == 1
    assert campaign.workloads[0].dtypes == ("bfloat16",)
    assert campaign.workloads[0].phases == ("training", "inference")
```

- [ ] **Step 2: Implement strict campaign parsing and presets**

Support `mednext-v1`, future `mednext-v2`, and `all`. Reject unavailable presets with an error listing installed model families. Expand `checkpointing: auto` into `none`, `all-expansion`, and `whole-block` contexts for profiling only; it does not alter runtime model choices.

- [ ] **Step 3: Write adaptive batch-search tests using a fake probe**

```python
def test_auto_search_is_dense_then_refines_to_vram_limit() -> None:
    def probe(batch: int) -> ProbeResult:
        return ProbeResult(batch=batch, feasible=batch <= 13, peak_bytes=batch * 100)

    result = search_batches(
        BatchSearch(strategy="auto", memory_fraction=0.90, dense_until=8),
        total_vram_bytes=1500,
        probe=probe,
    )

    assert set(range(1, 9)).issubset(result.probed_batches)
    assert result.maximum_feasible == 13
    assert 13 in result.probed_batches
    assert 14 in result.probed_batches
```

Also test batch-1 OOM, explicit maximum, 90% headroom, exponential growth, binary refinement, and insertion of neighbors around a reported implementation transition.

- [ ] **Step 4: Implement deterministic adaptive search**

The search sequence is:

1. probe batch 1 and 2;
2. measure or estimate memory slope;
3. probe every integer through `dense_until` while feasible;
4. probe powers of two above the dense range;
5. binary-refine the feasible/OOM boundary;
6. add maximum-feasible and backend-transition neighbors;
7. return ordered unique batches and every probe result.

The search module contains no CUDA calls. OOM isolation belongs to Task 7.

- [ ] **Step 5: Verify Task 6**

Run: `pytest tests/profiling/test_campaign.py tests/profiling/test_batch_search.py -q`

Expected: all tests pass without CUDA.

- [ ] **Step 6: Commit Task 6**

```bash
git add src/mednext_accel/profiling/__init__.py \
  src/mednext_accel/profiling/campaign.py \
  src/mednext_accel/profiling/batch_search.py \
  tests/profiling/test_campaign.py \
  tests/profiling/test_batch_search.py
git commit -m "feat: add profiling campaigns and batch search"
```

## Task 7: Implement Isolated Benchmarks, Whole-Model Validation, and Rule Synthesis

**Files:**

- Create: `src/mednext_accel/profiling/discovery.py`
- Create: `src/mednext_accel/profiling/benchmark.py`
- Create: `src/mednext_accel/profiling/whole_model.py`
- Create: `src/mednext_accel/profiling/synthesize.py`
- Create: `tests/profiling/test_discovery.py`
- Create: `tests/profiling/test_benchmark.py`
- Create: `tests/profiling/test_synthesize.py`
- Create: `tests/profiling/test_whole_model.py`

**Interfaces:**

- Consumes campaign workloads, adaptive model descriptors, and registry candidates.
- Produces raw `Measurement` records and one complete `OptimizationProfile`.

- [ ] **Step 1: Write descriptor discovery tests**

Construct a tiny Base-like model with reference operators and assert discovery deduplicates identical descriptors while retaining occurrence count and model roles. Test regular, downsample, transpose depthwise, pointwise expansion/projection, GroupNorm, and GELU families.

- [ ] **Step 2: Implement model discovery without executing kernels**

Use model metadata and a lightweight shape-propagation pass. The pass may run reference operations on meta/fake tensors; it must not allocate full CUDA activations. Return descriptor/context templates and stable architecture fingerprints.

- [ ] **Step 3: Write subprocess result and OOM isolation tests**

Use a fake child module invoked through `sys.executable` to return success, numerical failure, timeout, and OOM JSON records. Assert the parent campaign continues after OOM and records the failed batch.

- [ ] **Step 4: Implement isolated subprocess benchmarking**

Move the useful pointwise/depthwise benchmark logic from the old autotuner into implementation-specific benchmark adapters. Each child:

- imports CUDA/Triton only after argument parsing;
- validates forward and every declared backward phase;
- measures median latency after warmup;
- records peak allocated and reserved memory;
- emits one JSON record on stdout;
- converts CUDA OOM to a structured result;
- exits nonzero for infrastructure or schema errors.

- [ ] **Step 5: Write automatic synthesis tests**

```python
def test_adjacent_batches_with_same_winner_merge_into_interval() -> None:
    measurements = [
        measured(batch=2, reference_ms=4.0, candidate_ms=3.0),
        measured(batch=3, reference_ms=6.0, candidate_ms=4.5),
        measured(batch=4, reference_ms=8.0, candidate_ms=6.2),
    ]

    rules = synthesize_rules(measurements, objective=BALANCED)

    assert len(rules) == 1
    assert rules[0].batch.minimum == 2
    assert rules[0].batch.maximum == 4
    assert rules[0].selection.implementation == "pointwise_gemm_per_sample"
```

Also test:

- numerical failure always rejects a candidate;
- throughput objective selects the fastest valid candidate;
- memory objective selects the lowest-peak candidate inside latency tolerance;
- balanced objective rejects a small speedup with excessive memory;
- a whole-model memory regression produces a VRAM-qualified override;
- measured, interpolated, extrapolated, and default confidence is preserved.

- [ ] **Step 6: Implement rule and formula synthesis**

Group measurements by operator signature, dtype, phase, and checkpoint context. Rank candidates by the campaign objective. Merge adjacent batches with the same winner. Fit depthwise work-scaling anchors only when residual error stays within the campaign tolerance; otherwise retain piecewise intervals. Keep raw summaries and synthesis thresholds in provenance.

- [ ] **Step 7: Implement whole-model validation**

For each candidate profile slice, create the requested model with the user-selected checkpoint context, compile when requested by the campaign, and run a complete forward/loss/backward/optimizer step. Record median step time and peak memory. Compare against the reference profile and the operator-only candidate.

When isolated results fail to predict graph-level memory, emit an exact or VRAM-qualified override. The batch-6 `32<->64 @ 128^3` case must be representable by this test fixture.

- [ ] **Step 8: Verify Task 7**

Run: `pytest tests/profiling -q`

Expected: all CPU tests pass; tests marked `cuda` pass on the workstation GPU.

Run: `ruff check src/mednext_accel/profiling tests/profiling`

Expected: no errors.

- [ ] **Step 9: Commit Task 7**

```bash
git add src/mednext_accel/profiling tests/profiling
git commit -m "feat: synthesize profiles from isolated and model benchmarks"
```

## Task 8: Add the Zero-Argument CLI and Python Profiling API

**Files:**

- Create: `src/mednext_accel/profiling/api.py`
- Create: `src/mednext_accel/profiling/cli.py`
- Modify: `src/mednext_accel/profiling/__init__.py`
- Modify: `pyproject.toml`
- Modify: `tests/tools/test_cli.py`
- Create: `tests/profiling/test_api.py`
- Remove: `benchmarks/autotune.py`

**Interfaces:**

- Produces `profile(source=None) -> ProfilingResult`, `ProfilingResult.save()`, and the `mednext-accel profile` command.

- [ ] **Step 1: Write CLI parsing tests**

Test these commands without importing CUDA or Triton:

```text
mednext-accel --help
mednext-accel profile --help
mednext-accel profile
mednext-accel profile workload.yaml
mednext-accel profile --preset mednext-v1
```

Monkeypatch the profiling API for the execution tests. Assert zero-argument `profile` loads the `all` preset and automatic output path.

- [ ] **Step 2: Implement the console entry point**

Add:

```toml
[project.scripts]
mednext-accel = "mednext_accel.profiling.cli:main"
```

The root parser has a required `profile` subcommand. CUDA/Triton imports occur inside campaign execution, not parser construction. A workload path and `--preset` are mutually exclusive.

- [ ] **Step 3: Implement Python API and atomic persistence**

```python
@dataclass(frozen=True, slots=True)
class ProfilingResult:
    profile: OptimizationProfile
    measurements: tuple[Measurement, ...]
    output_path: Path

    def save(self, path: str | PathLike[str] | None = None) -> Path:
        destination = self.output_path if path is None else Path(path)
        return write_profile_atomic(destination, self.profile, self.measurements)


def profile(source: CampaignSource | None = None) -> ProfilingResult:
    campaign = load_campaign(source)
    measurements = run_campaign(campaign)
    generated_profile = synthesize_profile(campaign, measurements)
    result = ProfilingResult(
        profile=generated_profile,
        measurements=measurements,
        output_path=default_profile_path(generated_profile),
    )
    result.save()
    return result
```

Default output is `~/.cache/mednext_accel/profiles/sm<major><minor>-local.json`. Write to a sibling temporary file, `fsync`, and atomically replace the destination. Preserve raw measurement summaries in the output profile.

- [ ] **Step 4: Add user-facing progress and final summary**

Progress output reports campaign/workload/batch progress, OOM boundaries, candidate rejection, and elapsed estimates. The final summary reports the single output path, tested coverage, warnings, and profile identity. It does not ask users to choose among experiment files.

- [ ] **Step 5: Verify Task 8**

Run: `pytest tests/tools/test_cli.py tests/profiling/test_api.py -q`

Expected: all tests pass without CUDA.

Run: `python -m build`

Install the wheel in a temporary environment and run `mednext-accel profile --help` outside the repository.

- [ ] **Step 6: Commit Task 8**

```bash
git add pyproject.toml src/mednext_accel/profiling tests/tools/test_cli.py \
  tests/profiling/test_api.py benchmarks/autotune.py
git commit -m "feat: add automatic profiling command"
```

## Task 9: Generate and Validate the First Production Profiles

**Files:**

- Regenerate: `src/mednext_accel/profiles/generic-nvidia.json`
- Regenerate: `src/mednext_accel/profiles/sm120.json`
- Create: `artifacts/profile-campaigns/sm120/` locally; keep ignored by Git.
- Create: `docs/benchmarks/sm120-profile.md`
- Modify: `tests/optimization/test_bundled_profiles.py`

**Interfaces:**

- Consumes the profiler delivered in Tasks 6-8.
- Produces the first generated bundled profile and an auditable benchmark summary.

- [ ] **Step 1: Run the targeted RTX 5090 campaign**

Create an ignored campaign file covering MedNeXt v1 Base, BF16, `128^3`, training/inference, and automatic batches for all checkpoint contexts. Run:

```bash
mednext-accel profile artifacts/profile-campaigns/sm120/base-128.yaml
```

Confirm the search includes every batch 1-8 that fits and records OOM boundaries in isolated subprocesses.

- [ ] **Step 2: Compare generated decisions with current production evidence**

Assert generated rules retain or improve the measured current behavior for:

- batch 1 native high-resolution pointwise;
- batch 2-4 high-resolution pointwise GEMM;
- batch 6 VRAM-qualified high-resolution pointwise;
- batch-aware regular depthwise splits;
- regular depthwise dX;
- transpose depthwise dW;
- export-safe native inference paths.

Investigate any changed winner by rerunning the affected isolated shape and whole-model context before accepting the profile.

- [ ] **Step 3: Publish generated SM120 and generic profiles**

Copy the synthesized profile into `sm120.json`. Derive `generic-nvidia.json` by removing exact-SM-only overrides while retaining family defaults and general formulas. Preserve provenance and measurement summary hashes. Do not commit raw traces or compiled artifacts.

- [ ] **Step 4: Add release coverage assertions**

Test batch sizes 1-8, spatial shapes `96^3`, `128^3`, `160^3`, and `192^3`, and unknown SM 9.0 through resolver-only tests. Every supported descriptor must return a decision; the test records whether it is measured, interpolated, extrapolated, default, or guard.

- [ ] **Step 5: Document the evidence**

Write `docs/benchmarks/sm120-profile.md` with environment, campaign file, coverage, whole-model step time/VRAM tables, known extrapolations, and reproduction command. Link the generated profile and state that RTX 5090 evidence seeds generic NVIDIA behavior.

- [ ] **Step 6: Commit Task 9**

```bash
git add src/mednext_accel/profiles docs/benchmarks/sm120-profile.md \
  tests/optimization/test_bundled_profiles.py
git commit -m "perf: publish generated SM120 optimization profile"
```

## Task 10: Update Documentation, Migration Guidance, and Full Verification

**Files:**

- Modify: `README.md`
- Replace: `docs/optimization.md`
- Modify: `docs/development/package-design.md`
- Modify: `benchmarks/README.md`
- Modify: `tools/README.md`
- Modify: `.gitignore`
- Modify: `tests/test_installed_package.py`

**Interfaces:**

- Documents the final public contract and validates the distributable package.

- [ ] **Step 1: Rewrite installation and model examples around auto optimization**

The first model example uses no optimization call:

```python
from mednext_accel import mednext_base

model = mednext_base(in_channels=1, out_channels=3)
```

Document `optimization="reference"`, profile path/mapping input, explicit checkpointing, compilation recommendations, inspection reports, cross-SM warnings, and approximate semantic opt-in.

- [ ] **Step 2: Document the profiling workflow**

Lead with:

```bash
mednext-accel profile
```

Then document presets, the minimal workload YAML, automatic VRAM batch expansion, isolated OOM handling, output location, and how to pass the generated path into a factory. Do not present internal kernel benchmark flags as the standard workflow.

- [ ] **Step 3: Update project structure and artifact exclusions**

Document registry/profile/profiling directories. Ensure raw profiler outputs, traces, TorchInductor caches, Nsight captures, temporary campaign files, build directories, and local generated profiles remain ignored. Bundled JSON profiles and benchmark summaries stay tracked.

- [ ] **Step 4: Run complete verification**

Run:

```bash
pytest -q
ruff check .
python -m build
python -m twine check dist/*
git diff --check
```

Expected:

- all CPU tests pass;
- CUDA tests pass on the RTX 5090;
- ONNX tests pass when export extras are installed, otherwise skip explicitly;
- wheel and source distribution pass Twine checks;
- no ignored artifact is staged;
- repository status contains only intended source, profile, test, and documentation changes.

- [ ] **Step 5: Install and test the wheel outside the source tree**

In the release-test virtual environment:

```python
import torch
import mednext_accel

model = mednext_accel.mednext_small(in_channels=1, out_channels=3).eval()
output = model(torch.randn(1, 1, 32, 32, 32))
assert output.shape == (1, 3, 32, 32, 32)
assert model.explain_optimization(
    input_shape=(3, 1, 128, 128, 128),
    dtype="bfloat16",
    device="cpu",
).decisions
```

- [ ] **Step 6: Commit Task 10**

```bash
git add README.md docs benchmarks/README.md tools/README.md .gitignore tests/test_installed_package.py
git commit -m "docs: publish automatic optimization workflow"
```

## Final Review Gate

- [ ] Confirm `git log --oneline` shows one reviewable commit per task.
- [ ] Confirm no old `policy="torch"`, `policy="conservative"`, or runtime `policy="autotune"` examples remain.
- [ ] Confirm batch 3 resolves the high-resolution pointwise interval.
- [ ] Confirm unknown SM resolves `generic-nvidia` with one warning.
- [ ] Confirm cross-SM external profiles warn and execute.
- [ ] Confirm state-dict keys match `optimization="reference"` exactly.
- [ ] Confirm `torch.compile(fullgraph=True)`, direct `torch.jit.trace`, helper tracing, and ONNX export pass.
- [ ] Confirm `mednext-accel profile` runs with no workload flags and writes one merged profile.
- [ ] Confirm a simulated 96 GiB batch-search probe extends beyond the same workload's simulated 32 GiB maximum; retain real-device maximum batches as profile provenance rather than a release blocker.
- [ ] Confirm Git status is clean and push the completed branch only after all checks pass.
