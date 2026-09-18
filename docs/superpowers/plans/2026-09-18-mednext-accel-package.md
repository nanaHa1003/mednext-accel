# MedNeXt-Accel Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat experimental codebase with an installable `mednext-accel` package that provides an official-compatible MedNeXt v1 model, explicit MONAI compatibility, lossless checkpoint import, eval tracing/export, and optional per-shape acceleration.

**Architecture:** A pure-PyTorch reference model under `src/mednext_accel` owns all parameters and is always usable without Triton. Checkpoint adapters translate external schemas into a stable native schema. Optional optimization changes module execution policy while preserving parameters and makes backward-only custom paths fall back to standard PyTorch during evaluation.

**Tech Stack:** Python 3.10+, PyTorch 2.6+, optional Triton, pytest, Ruff, ONNX/ONNX Script/ONNX Runtime, Hatchling.

**Spec:** `docs/superpowers/specs/2026-09-18-mednext-accel-package-design.md`

## Global Constraints

- Distribution name is `mednext-accel`; import namespace is `mednext_accel`.
- Core runtime depends on PyTorch only; CUDA is never a Python dependency and Triton is optional.
- Do not pin an exact PyTorch, Triton, or CUDA version.
- Model construction performs no benchmarking, compilation, device transfer, or cache write.
- Official MedNeXt v1 S/B/M/L checkpoint import is lossless for compatible model configurations.
- MONAI B/M/L checkpoints load only into explicit MONAI-compatible architectures.
- Evaluation always returns one primary logits tensor.
- Direct `torch.jit.trace`, `torch.export`, and ONNX export must work in evaluation mode without custom `mednext_accel::*` nodes in the portable inference graph.
- Generated artifacts, profiles, archives, caches, and environments remain ignored and absent from packages.
- Use tests before production changes and commit each independently reviewable task.

---

### Task 1: Packaging scaffold and public configuration

**Files:**
- Create: `pyproject.toml`
- Create: `LICENSE`
- Create: `NOTICE`
- Create: `README.md`
- Create: `src/mednext_accel/__init__.py`
- Create: `src/mednext_accel/models/__init__.py`
- Create: `src/mednext_accel/models/config.py`
- Create: `src/mednext_accel/checkpointing.py`
- Create: `tests/test_public_api.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces `CheckpointConfig(expansion: bool = True, stages: tuple[int, ...] | None = None)`.
- Produces immutable `MedNeXtV1Config` and `get_mednext_v1_config(variant, ..., compatibility="official")`.
- Establishes top-level names that later tasks fill: `MedNeXtV1`, `mednext_small`, `mednext_base`, `mednext_medium`, `mednext_large`.

- [ ] **Step 1: Write failing public configuration tests**

```python
def test_base_config_matches_official_v1():
    config = get_mednext_v1_config("base", in_channels=1, out_channels=3)
    assert config.block_counts == (2,) * 9
    assert config.expansion_ratios == (2, 3, 4, 4, 4, 4, 4, 3, 2)
    assert config.downsample_expansion_ratios == (3, 4, 4, 4)

def test_checkpoint_stages_are_validated():
    with pytest.raises(ValueError, match="stages"):
        CheckpointConfig(stages=(0, 0))
```

- [ ] **Step 2: Run `pytest tests/test_public_api.py -q` and verify import/configuration failures**
- [ ] **Step 3: Add `pyproject.toml`, license files, src package, immutable configs, and validation**
- [ ] **Step 4: Install with `python -m pip install -e '.[test]'` and run the test again**
- [ ] **Step 5: Build wheel/sdist with `python -m build` and inspect contents for only package/docs metadata**
- [ ] **Step 6: Commit with `git commit -m "build: scaffold mednext-accel package"`**

### Task 2: Pure-PyTorch MedNeXt v1 reference model

**Files:**
- Create: `src/mednext_accel/models/blocks.py`
- Create: `src/mednext_accel/models/mednext_v1.py`
- Create: `tests/models/test_mednext_v1.py`
- Modify: `src/mednext_accel/__init__.py`
- Modify: `src/mednext_accel/models/__init__.py`

**Interfaces:**
- Produces `MedNeXtV1(config, *, checkpointing=None, deep_supervision_output="tuple", approximate_gelu_eval=False)`.
- Produces factories with keyword-only public arguments.
- `forward(x)` returns a tensor in eval and tensor/tuple/list according to training deep-supervision settings.

- [ ] **Step 1: Write failing shape, parameter-count, eval-output, and configuration tests**

```python
@pytest.mark.parametrize("factory", [mednext_small, mednext_base, mednext_medium, mednext_large])
def test_factory_eval_returns_logits(factory):
    model = factory(in_channels=1, out_channels=3).eval()
    assert model(torch.randn(1, 1, 32, 32, 32)).shape == (1, 3, 32, 32, 32)

def test_base_parameter_count_matches_official_without_dummy_tensor():
    model = mednext_base(in_channels=1, out_channels=3, deep_supervision=True)
    assert sum(p.numel() for p in model.parameters()) == 10_529_231
```

- [ ] **Step 2: Run focused tests and verify missing model failures**
- [ ] **Step 3: Implement blocks using standard PyTorch modules and stable native names**
- [ ] **Step 4: Implement encoder, bottleneck, decoder, heads, and factories**
- [ ] **Step 5: Run focused tests and correct only implementation defects**
- [ ] **Step 6: Add failing checkpoint-policy and deep-supervision output tests**
- [ ] **Step 7: Implement expansion checkpoint selection and tuple/list/stacked training outputs**
- [ ] **Step 8: Run `pytest tests/models/test_mednext_v1.py -q` and the legacy suite**
- [ ] **Step 9: Commit with `git commit -m "feat: add MedNeXt v1 reference models"`**

### Task 3: Official and MONAI checkpoint interoperability

**Files:**
- Create: `src/mednext_accel/checkpoints/__init__.py`
- Create: `src/mednext_accel/checkpoints/api.py`
- Create: `src/mednext_accel/checkpoints/official_v1.py`
- Create: `src/mednext_accel/checkpoints/monai.py`
- Create: `src/mednext_accel/compat/__init__.py`
- Create: `src/mednext_accel/compat/monai.py`
- Create: `tests/checkpoints/test_official_v1.py`
- Create: `tests/checkpoints/test_monai.py`

**Interfaces:**
- Produces `load_checkpoint(model, checkpoint, *, source="auto", strict=True) -> CheckpointLoadReport`.
- Produces pure `convert_state_dict(state_dict, *, source, target_config)` helpers.
- Produces explicit MONAI-compatible S/B/M/L factories.

- [ ] **Step 1: Write a failing official-key conversion test including `dummy_tensor` and deep-supervision heads**
- [ ] **Step 2: Verify failure, then implement deterministic official-to-native key conversion**
- [ ] **Step 3: Write a failing exact-output test using converted official-compatible weights**
- [ ] **Step 4: Implement loading reports, wrapper-prefix removal, raw/wrapped state-dict extraction, and strict diagnostics**
- [ ] **Step 5: Write a failing MONAI Small lossless conversion test**
- [ ] **Step 6: Implement MONAI key mapping and verify exact output against installed MONAI when available**
- [ ] **Step 7: Write a failing MONAI Base mismatch test that names `down_0` and `down_1`**
- [ ] **Step 8: Implement `compat.monai` factories with MONAI downsample expansion ratios and lossless loading**
- [ ] **Step 9: Run all checkpoint tests and commit with `git commit -m "feat: add checkpoint compatibility adapters"`**

### Task 4: Stable evaluation export

**Files:**
- Create: `src/mednext_accel/export.py`
- Create: `tests/export/test_torchscript.py`
- Create: `tests/export/test_torch_export.py`
- Create: `tests/export/test_onnx.py`
- Create: `docs/export.md`

**Interfaces:**
- Direct PyTorch trace/export remains supported.
- Convenience `trace(model, example, *, check_trace=True)` validates eval mode and returns `torch.jit.ScriptModule`.
- Convenience `export_onnx(model, example, *, opset_version=18, dynamic_batch=False)` returns the modern ONNX program.

- [ ] **Step 1: Write a failing test for trace, save/load, and absence of `mednext_accel::` graph nodes**
- [ ] **Step 2: Verify failure and make eval paths trace-safe without special user wrappers**
- [ ] **Step 3: Write a failing strict `torch.export` equality test and make it pass**
- [ ] **Step 4: Write ONNX checker/runtime tests guarded by the export extra**
- [ ] **Step 5: Verify ONNX output against eager FP32 with explicit tolerances**
- [ ] **Step 6: Document static spatial and dynamic-batch limits**
- [ ] **Step 7: Run all export tests and commit with `git commit -m "feat: support eval model export"`**

### Task 5: Optional accelerated operators

**Files:**
- Create: `src/mednext_accel/ops/__init__.py`
- Create: `src/mednext_accel/ops/pointwise.py`
- Create: `src/mednext_accel/ops/depthwise.py`
- Create: `src/mednext_accel/ops/_triton/__init__.py`
- Create: `src/mednext_accel/ops/_triton/depthwise.py`
- Create: `tests/ops/test_pointwise.py`
- Create: `tests/ops/test_depthwise.py`

**Interfaces:**
- Produces replacement/dispatch modules that preserve `weight` and `bias` parameter identity and state-dict paths.
- Triton modules are imported only when acceleration is requested.
- Eval or disabled-grad execution of backward-only depthwise optimizations uses standard PyTorch operations.

- [ ] **Step 1: Port the existing pointwise behavior tests to the package namespace and observe import failures**
- [ ] **Step 2: Implement traceable GEMM pointwise execution with parameter identity preservation**
- [ ] **Step 3: Port depthwise forward/gradient/replacement tests and observe import failures**
- [ ] **Step 4: Split dispatch modules from Triton kernels and rename custom-op namespace to `mednext_accel`**
- [ ] **Step 5: Register fake/autograd behavior and validate with `torch.library.opcheck` and gradient tests**
- [ ] **Step 6: Add eval fallback tests proving trace graphs contain standard ATen operations only**
- [ ] **Step 7: Run CPU and available CUDA tests, then commit with `git commit -m "feat: package accelerated convolution operators"`**

### Task 6: Optimization policy and per-shape autotuning

**Files:**
- Create: `src/mednext_accel/optimization/__init__.py`
- Create: `src/mednext_accel/optimization/api.py`
- Create: `src/mednext_accel/optimization/policy.py`
- Create: `src/mednext_accel/optimization/report.py`
- Create: `src/mednext_accel/optimization/autotune.py`
- Create: `tests/optimization/test_policy.py`
- Create: `tests/optimization/test_autotune.py`

**Interfaces:**
- Produces `optimize(model, *, input_shape, dtype, policy) -> OptimizationReport`.
- Produces `use_backend(model, "torch")` context manager with policy restoration.
- Cache keys include hardware and software identity plus all benchmark-relevant shapes.

- [ ] **Step 1: Write failing `torch` and `conservative` policy report tests**
- [ ] **Step 2: Implement immutable reports and in-place execution-policy application**
- [ ] **Step 3: Add failing tests for validated transpose-dW and stride2-dX selections**
- [ ] **Step 4: Implement conservative tables and native fallback for every unlisted shape**
- [ ] **Step 5: Port shape capture and benchmark tests for one benchmark per unique shape**
- [ ] **Step 6: Implement versioned JSON cache with atomic replace and corruption fallback**
- [ ] **Step 7: Test backend context restoration on success and exception**
- [ ] **Step 8: Run optimization tests and commit with `git commit -m "feat: add model optimization policies"`**

### Task 7: Repository tooling, documentation, and experiment cleanup

**Files:**
- Create: `benchmarks/README.md`
- Move/adapt: kernel and model benchmark scripts under `benchmarks/`
- Move/adapt: profiling collection and summary scripts under `tools/`
- Create: `docs/checkpoints.md`
- Create: `docs/optimization.md`
- Modify: `README.md`
- Modify: `.gitignore`
- Remove: obsolete flat production modules and duplicate flat tests after migration

**Interfaces:**
- CLI tools import installed `mednext_accel`; none rely on repository-root import behavior.
- Benchmark output directories are user-selectable and ignored by default.

- [ ] **Step 1: Add smoke tests for benchmark/profile CLI `--help` without Triton import**
- [ ] **Step 2: Move and adapt scripts, then make smoke tests pass**
- [ ] **Step 3: Document installation, basic model use, checkpoint import, optimization order, export, and limitations**
- [ ] **Step 4: Remove duplicate root modules only after all replacement tests pass**
- [ ] **Step 5: Run repository tests and inspect `git status --ignored` for artifact coverage**
- [ ] **Step 6: Commit with `git commit -m "docs: organize package tools and usage"`**

### Task 8: CI and release verification

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_installed_package.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- CPU CI runs lint, unit tests, export tests, build, metadata check, and wheel-install smoke test.
- CUDA tests remain marked and runnable with documented commands on supported hardware.

- [ ] **Step 1: Add a failing wheel-install smoke test that imports outside the repository**
- [ ] **Step 2: Configure build metadata and package data until wheel/sdist checks pass**
- [ ] **Step 3: Add Ruff and pytest configuration plus CPU CI workflow**
- [ ] **Step 4: Run `ruff check .`, the full pytest suite, package build, and wheel smoke test**
- [ ] **Step 5: Run available CUDA correctness and compile tests**
- [ ] **Step 6: Re-run 128-cubed Base profiling for reference and conservative policies when memory permits**
- [ ] **Step 7: Review the spec line by line, record any deferred items explicitly, and commit with `git commit -m "ci: validate mednext-accel releases"`**
