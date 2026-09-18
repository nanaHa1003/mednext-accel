# Resolution-Aware Selective Checkpointing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow expanded-branch checkpointing at an explicit tuple of static resolution levels and measure the time/VRAM Pareto curve.

**Architecture:** Keep `checkpoint_style` as the recomputation target and add `checkpoint_levels` as a static model-construction selector. Canonicalize and validate the tuple in the shared policy module, then pass the existing `checkpoint_expanded` Boolean to each block according to the expansion branch's architectural level. Extend the profiler without adding runtime tensor-shape decisions.

**Tech Stack:** Python 3.10+, PyTorch 2.x, `torch.utils.checkpoint`, `unittest`, `torch.compile`, existing profiling matrix.

**Spec:** [`docs/superpowers/specs/2026-09-18-resolution-aware-checkpointing-design.md`](../specs/2026-09-18-resolution-aware-checkpointing-design.md)

## Global Constraints

- `checkpoint_levels` has type `Optional[Tuple[int, ...]]`; `None` means every level for `expanded`.
- Explicit levels are valid only with `checkpoint_style="expanded"`.
- Reject empty tuples, duplicate values, Booleans, non-integers, negative values, and values greater than model depth.
- Sort accepted tuples for deterministic introspection.
- Assign levels from static architecture topology, never runtime tensor shapes.
- Preserve every Parameter name, state-dict key, output contract, and existing checkpoint mode.
- Keep `use_grad_checkpoint=True` mapped to whole-block checkpointing.
- Keep profiling artifacts under ignored `profile_results/`; commit only source, tests, and summarized documentation.

---

### Task 1: Validate and canonicalize level tuples

**Files:**
- Modify: `activation_checkpoint.py`
- Modify: `test_activation_checkpoint.py`

**Interfaces:**
- Consumes: `CheckpointStyle` and `CHECKPOINT_STYLES`.
- Produces: `CheckpointLevels` and `resolve_checkpoint_levels(style, checkpoint_levels, depth)`.

- [ ] **Step 1: Add failing resolver tests**

Add tests asserting:

```python
self.assertIsNone(resolve_checkpoint_levels("expanded", None, depth=4))
self.assertEqual(
    resolve_checkpoint_levels("expanded", (2, 0, 1), depth=4),
    (0, 1, 2),
)
```

Use `subTest` to reject `()`, `(0, 0)`, `(-1,)`, `(5,)`, `(True,)`, `(1.0,)`,
and `[0, 1]`. Assert explicit `(0,)` with `none` or `block` raises a message
containing `only valid with checkpoint_style='expanded'`.

- [ ] **Step 2: Run the resolver tests and observe the missing API failure**

Run: `python -m unittest test_activation_checkpoint.CheckpointPolicyTests -v`

Expected: import failure for `resolve_checkpoint_levels`.

- [ ] **Step 3: Implement the resolver**

Add:

```python
CheckpointLevels = Optional[Tuple[int, ...]]


def resolve_checkpoint_levels(
    checkpoint_style: CheckpointStyle,
    checkpoint_levels: CheckpointLevels,
    depth: int,
) -> CheckpointLevels:
    if checkpoint_levels is None:
        return None
    if checkpoint_style != "expanded":
        raise ValueError(
            "checkpoint_levels is only valid with checkpoint_style='expanded'"
        )
    if not isinstance(checkpoint_levels, tuple) or not checkpoint_levels:
        raise ValueError("checkpoint_levels must be a non-empty tuple of integers")
    if any(type(level) is not int for level in checkpoint_levels):
        raise ValueError("checkpoint_levels must contain only integers")
    if len(set(checkpoint_levels)) != len(checkpoint_levels):
        raise ValueError("checkpoint_levels must not contain duplicates")
    if any(level < 0 or level > depth for level in checkpoint_levels):
        raise ValueError(f"checkpoint_levels must be between 0 and {depth}")
    return tuple(sorted(checkpoint_levels))
```

- [ ] **Step 4: Run the focused and policy suites**

Run: `python -m unittest test_activation_checkpoint.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add activation_checkpoint.py test_activation_checkpoint.py
git commit -m "feat: validate checkpoint resolution levels"
```

### Task 2: Assign checkpointing by expansion resolution

**Files:**
- Modify: `mednext.py`
- Modify: `test_mednext.py`

**Interfaces:**
- Consumes: `resolve_checkpoint_levels(style, levels, depth)`.
- Produces: `checkpoint_levels` on `MedNeXt` and every factory, plus static block selection.

- [ ] **Step 1: Add failing model assignment tests**

For a Base model, collect every `(module_name, module.checkpoint_expanded)` for
instances of `MedNeXtBlock`. Assert:

```python
self.assertEqual(sum(flags_for_all_levels), 26)
self.assertEqual(sum(flags_for_levels_0_1), 11)
self.assertEqual(sum(flags_for_level_0), 5)
```

Also assert the exact level-0 names:

```python
{
    "enc_blocks.0.0",
    "enc_blocks.0.1",
    "up_blocks.3",
    "dec_blocks.3.0",
    "dec_blocks.3.1",
}
```

Assert the model stores `(0, 1)` when constructed with `(1, 0)`, and all four
factories accept `checkpoint_levels`.

- [ ] **Step 2: Add failing validation tests at the model boundary**

Assert `ValueError` for levels with `none`, `block`, legacy
`use_grad_checkpoint=True`, empty/duplicate/out-of-range tuples, and non-tuple
inputs. This verifies errors are raised before training begins.

- [ ] **Step 3: Run the focused tests and observe the missing argument failure**

Run: `python -m unittest test_mednext.MedNeXtCheckpointPolicyTests -v`

Expected: `unexpected keyword argument 'checkpoint_levels'`.

- [ ] **Step 4: Add the model and factory argument**

Append to `MedNeXt` and all factories:

```python
checkpoint_levels: CheckpointLevels = None
```

After validating `num_blocks` and setting `self.depth`, resolve:

```python
self.checkpoint_levels = resolve_checkpoint_levels(
    self.checkpoint_style, checkpoint_levels, self.depth
)

def checkpoint_expanded_at(level: int) -> bool:
    return self.checkpoint_style == "expanded" and (
        self.checkpoint_levels is None or level in self.checkpoint_levels
    )
```

Do not store the nested helper on the module.

- [ ] **Step 5: Pass the correct Boolean to each block**

Use these expressions at construction sites:

```python
# encoder normal block
checkpoint_expanded=checkpoint_expanded_at(i)

# down block
checkpoint_expanded=checkpoint_expanded_at(i + 1)

# bottleneck
checkpoint_expanded=checkpoint_expanded_at(self.depth)

# up and decoder blocks inside reversed(range(self.depth))
checkpoint_expanded=checkpoint_expanded_at(i)
```

- [ ] **Step 6: Extend gradient parity to partial levels**

Run the existing double-precision whole-model comparison for
`checkpoint_levels=(0,)` and `(0, 1)` with deep supervision both disabled and
enabled. Require exact output, input-gradient, and named parameter-gradient
parity.

- [ ] **Step 7: Prove only selected blocks invoke checkpoint**

Patch `mednext.grad_ckpt`, run a training forward on a depth-one direct model
with `checkpoint_levels=(0,)`, and record the bound module for each checkpoint
call. Assert the calls come only from level-zero encoder, up, and decoder
blocks. The level-one down and bottleneck blocks must execute directly.

- [ ] **Step 8: Run all model tests**

Run: `python -m unittest test_mednext.py -v`

Expected: all tests pass, including current all-level and whole-block behavior.

- [ ] **Step 9: Commit**

```bash
git add mednext.py test_mednext.py
git commit -m "feat: select checkpointing by resolution level"
```

### Task 3: Expose level selection in profiling tools

**Files:**
- Modify: `profile_mednext.py`
- Modify: `run_profile_matrix.py`
- Modify: `summarize_profiles.py`
- Modify: `test_profiling.py`

**Interfaces:**
- Consumes: `--checkpoint-style expanded` and a tuple of architectural levels.
- Produces: `--checkpoint-levels`, canonical summary metadata, and matrix cases for levels 0, 0--1, and 0--2.

- [ ] **Step 1: Add failing profiler CLI tests**

Run the CPU smoke profiler with:

```text
--checkpoint-style expanded --checkpoint-levels 1 0
```

Assert `settings.effective_checkpoint_levels == [0, 1]`. Add subprocess cases
asserting that `--no-checkpoint --checkpoint-levels 0` and
`--checkpoint --checkpoint-levels 0` exit nonzero with the validation message.

- [ ] **Step 2: Add a failing matrix dry-run test**

Request cases `compile_expanded_l0`, `compile_expanded_l01`, and
`compile_expanded_l012`. Assert emitted commands contain respectively:

```text
--checkpoint-levels 0
--checkpoint-levels 0 1
--checkpoint-levels 0 1 2
```

- [ ] **Step 3: Run profiling tests and observe argparse failures**

Run: `python -m unittest test_profiling.py -v`

Expected: failures because the option and matrix cases do not exist.

- [ ] **Step 4: Implement profiler parsing and metadata**

Add:

```python
parser.add_argument('--checkpoint-levels', type=int, nargs='+')
```

Before model creation, reject levels unless the effective style is `expanded`.
Convert the parser list to a tuple for the model. After constructing the model,
store:

```python
args.effective_checkpoint_levels = (
    None if model.checkpoint_levels is None else list(model.checkpoint_levels)
)
```

- [ ] **Step 5: Add matrix cases**

Add:

```python
'compile_expanded_l0': [
    '--checkpoint-style', 'expanded', '--checkpoint-levels', '0',
    '--cudnn-benchmark', '--compile'],
'compile_expanded_l01': [
    '--checkpoint-style', 'expanded', '--checkpoint-levels', '0', '1',
    '--cudnn-benchmark', '--compile'],
'compile_expanded_l012': [
    '--checkpoint-style', 'expanded', '--checkpoint-levels', '0', '1', '2',
    '--cudnn-benchmark', '--compile'],
```

- [ ] **Step 6: Make summary aggregation hashable and backward compatible**

Add a helper that returns `"all"` for absent/`None` metadata and a compact
comma-separated string such as `"0,1"` for a list. Include
`effective_checkpoint_levels` in output rows and aggregation keys. Existing
results must continue to summarize.

- [ ] **Step 7: Run profiler and full tests**

Run:

```bash
python -m unittest test_profiling.py -v
python -m unittest discover -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add profile_mednext.py run_profile_matrix.py summarize_profiles.py test_profiling.py
git commit -m "feat: profile checkpoint resolution levels"
```

### Task 4: Measure and document the Pareto curve

**Files:**
- Modify: `ARCHITECTURE_MEMORY.md`
- Ignored artifacts: `profile_results/resolution-checkpoint-*`

**Interfaces:**
- Consumes: compiled matrix cases from Task 3.
- Produces: reproducible RTX 5090 native/custom VRAM tables and a keep/drop decision for each sampled tuple.

- [ ] **Step 1: Run the native RTX 5090 matrix**

```bash
python run_profile_matrix.py \
  --output profile_results/resolution-checkpoint-native-$(hostname) \
  --variant base --shape 1 1 128 128 128 --classes 3 8 \
  --precisions bf16 \
  --cases compile compile_expanded_l0 compile_expanded_l01 \
          compile_expanded_l012 compile_expanded compile_ckpt \
  --deep-supervision --no-include-fp32-baseline \
  --repeats 3 --warmup 20 --steps 50 --profile-steps 3
```

- [ ] **Step 2: Run the custom-operator RTX 5090 matrix**

Run the same command with a distinct output directory and:

```text
--pointwise-gemm --depthwise-split
```

- [ ] **Step 3: Calculate decision columns**

For every class count and policy calculate median step time, sample standard
deviation, maximum peak allocation, time versus `none`, memory saved versus
`none`, and time/memory relative to all-level `expanded`. Include the three raw
step times and median forward/backward GPU times.

- [ ] **Step 4: Apply the Pareto rule**

A sampled tuple is retained as evidence when no other measured policy is both
faster and lower-memory. Do not create model-level preset strings. Record any
dominated sampled tuple and keep tuple selection available for advanced users.

- [ ] **Step 5: Update the architecture document**

Add native and custom tables in the format defined by the spec, the exact
software/hardware versions, and the Pareto interpretation. Mark resolution-aware
checkpointing implemented. State that cluster portability remains unverified
until L40S, RTX A6000, and RTX 8000 results are supplied.

- [ ] **Step 6: Run final verification**

```bash
python -m unittest discover -v
python -m compileall -q activation_checkpoint.py mednext.py \
  profile_mednext.py run_profile_matrix.py summarize_profiles.py
git diff --check
```

Verify `git status --ignored` shows raw results under `!! profile_results/` and
`git ls-files` contains no profiling outputs.

- [ ] **Step 7: Commit documentation only**

```bash
git add ARCHITECTURE_MEMORY.md
git commit -m "docs: record checkpoint level Pareto curve"
```

## Completion Criteria

- `checkpoint_levels=(...)` has precise static level semantics and validation.
- Existing `expanded`, `block`, and legacy Boolean calls retain their behavior.
- Base `(0, 1)` selects exactly 11 of 26 expansion branches.
- Partial policies preserve outputs, gradients, parameters, and state dicts.
- CUDA BF16 full-graph compilation and custom operators remain compatible.
- Profiler metadata and aggregation distinguish all sampled tuples.
- RTX 5090 native and custom tables report time and VRAM for every requested policy.
- Raw profiler artifacts remain excluded from Git.
