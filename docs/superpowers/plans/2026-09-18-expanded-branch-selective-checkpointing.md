# Expanded-Branch Selective Checkpointing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-quality `expanded` activation-checkpoint policy that recomputes only `conv2 -> GELU -> conv3`, preserves existing checkpoints and model outputs, and can be benchmarked against `none` and whole-block checkpointing.

**Architecture:** Introduce one shared policy type and compatibility resolver, then pass a resolved Boolean into every MedNeXt block. The block keeps its existing `layers.*` module tree and state-dict keys, but executes the depthwise and normalization prefix directly and places a non-reentrant PyTorch checkpoint around the expansion branch during gradient-enabled training. Whole-block checkpointing remains at the existing model traversal boundary.

**Tech Stack:** Python 3.10+, PyTorch 2.x, `torch.utils.checkpoint`, `unittest`, `torch.compile`, existing profiling scripts.

**Spec:** [`ARCHITECTURE_MEMORY.md`](../../../ARCHITECTURE_MEMORY.md), especially section “1. Selective expanded-branch checkpointing”.

## Global Constraints

- Preserve every existing parameter name and tensor shape. A strict v1 checkpoint load must succeed in all three policies.
- Preserve the meaning of `use_grad_checkpoint=True`: it continues to select whole-block checkpointing.
- Add `checkpoint_style` after all existing constructor arguments so positional callers do not change meaning.
- Supported policy names are exactly `none`, `expanded`, and `block`.
- Reject `use_grad_checkpoint=True` together with an explicit `checkpoint_style`; do not silently choose one.
- Use `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
- Activate selective checkpointing only when both `module.training` and `torch.is_grad_enabled()` are true.
- Do not change deep-supervision output semantics in this work.
- Do not change pointwise or depthwise operators in this work. They are composition targets for validation.
- Treat numerical compatibility, state-dict compatibility, eager/compiled execution, and profiler reproducibility as release gates.
- The current workspace is not a Git repository. The commit steps below describe the intended review units; initialize or restore Git before executing them if commit history is required.

## File Map

| File | Action | Responsibility |
|---|---|---|
| `activation_checkpoint.py` | Create | Shared policy type, valid values, and legacy-option resolver. |
| `test_activation_checkpoint.py` | Create | Exhaustive policy resolution and error tests. |
| `mednext.py` | Modify | Expansion-branch boundary, policy propagation, and legacy whole-block dispatch. |
| `test_mednext.py` | Modify | Block/model parity, lifecycle, state-dict, deep-supervision, and factory tests. |
| `profile_mednext.py` | Modify | Accept and record `none`, `expanded`, and `block` while retaining old CLI flags. |
| `run_profile_matrix.py` | Modify | Add eager and compiled expanded-policy cases. |
| `summarize_profiles.py` | Modify | Group and report by effective checkpoint policy. |
| `ARCHITECTURE_MEMORY.md` | Modify | Record the implemented contract and final repeated measurements. |

## Task 1: Define the policy contract and legacy mapping

**Files:**
- Create: `activation_checkpoint.py`
- Create: `test_activation_checkpoint.py`

- [ ] **Step 1: Write the failing policy tests**

Create `test_activation_checkpoint.py` with tests for all supported mappings and invalid combinations:

```python
"""Tests for activation-checkpoint policy resolution."""
import unittest

from activation_checkpoint import CHECKPOINT_STYLES, resolve_checkpoint_style


class CheckpointPolicyTests(unittest.TestCase):
    def test_legacy_flag_maps_to_existing_behavior(self):
        self.assertEqual(resolve_checkpoint_style(False, None), "none")
        self.assertEqual(resolve_checkpoint_style(True, None), "block")

    def test_explicit_policy_is_returned(self):
        for style in CHECKPOINT_STYLES:
            with self.subTest(style=style):
                self.assertEqual(resolve_checkpoint_style(False, style), style)

    def test_rejects_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_style"):
            resolve_checkpoint_style(False, "stage")  # type: ignore[arg-type]

    def test_rejects_legacy_true_with_explicit_policy(self):
        with self.assertRaisesRegex(ValueError, "use_grad_checkpoint"):
            resolve_checkpoint_style(True, "expanded")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and verify that import fails**

Run: `python -m unittest test_activation_checkpoint.py -v`

Expected: `ModuleNotFoundError: No module named 'activation_checkpoint'`.

- [ ] **Step 3: Implement the typed resolver**

Create `activation_checkpoint.py`:

```python
"""Activation-checkpoint execution policies shared by model architectures."""
from typing import Literal, Optional, Tuple, cast

CheckpointStyle = Literal["none", "expanded", "block"]
CHECKPOINT_STYLES: Tuple[CheckpointStyle, ...] = ("none", "expanded", "block")


def resolve_checkpoint_style(
    use_grad_checkpoint: bool,
    checkpoint_style: Optional[CheckpointStyle],
) -> CheckpointStyle:
    if checkpoint_style is None:
        return "block" if use_grad_checkpoint else "none"
    if checkpoint_style not in CHECKPOINT_STYLES:
        raise ValueError(
            f"checkpoint_style must be one of {CHECKPOINT_STYLES}, got {checkpoint_style!r}"
        )
    if use_grad_checkpoint:
        raise ValueError(
            "use_grad_checkpoint=True cannot be combined with checkpoint_style; "
            "use checkpoint_style='block' instead"
        )
    return cast(CheckpointStyle, checkpoint_style)
```

- [ ] **Step 4: Run the policy tests**

Run: `python -m unittest test_activation_checkpoint.py -v`

Expected: four tests pass.

- [ ] **Step 5: Commit the policy unit**

```bash
git add activation_checkpoint.py test_activation_checkpoint.py
git commit -m "feat: define activation checkpoint policies"
```

## Task 2: Add the checkpoint boundary to every MedNeXt block type

**Files:**
- Modify: `mednext.py`
- Modify: `test_mednext.py`

- [ ] **Step 1: Add a block-construction helper and parity assertions to the tests**

Add tests that cover `MedNeXtBlock`, `MedNeXtDownBlock`, and `MedNeXtUpBlock` in 3D. For each class:

1. Build a reference block with `checkpoint_expanded=False` and a candidate with `checkpoint_expanded=True`.
2. Copy the state dict with `strict=True`.
3. Use independent cloned `torch.float64` inputs with `requires_grad=True`.
4. Run both blocks in training mode.
5. Backpropagate the same dense random output gradient.
6. Assert output, input gradient, and every parameter gradient with `rtol=0, atol=0`.

Use these small shapes so the test remains quick:

```python
cases = (
    (mednext.MedNeXtBlock, (1, 2, 5, 6, 7), 2),
    (mednext.MedNeXtDownBlock, (1, 2, 6, 8, 10), 4),
    (mednext.MedNeXtUpBlock, (1, 4, 3, 4, 5), 2),
)
```

Construct with `spatial_dims=3`, `expand_ratio=2`, `kernel_size=3`, and `res_block=True`.

- [ ] **Step 2: Add lifecycle tests by patching the imported checkpoint function**

Patch `mednext.grad_ckpt` with a wrapper that increments a counter and directly invokes its function argument. Assert:

- training plus enabled gradients calls it once per block invocation;
- evaluation calls it zero times;
- `torch.no_grad()` calls it zero times;
- `checkpoint_expanded=False` calls it zero times.

The wrapper must accept `use_reentrant` and assert that it is `False`.

- [ ] **Step 3: Run the block tests and verify constructor failures**

Run: `python -m unittest test_mednext.ExpandedBranchCheckpointBlockTests -v`

Expected: failures reporting that `checkpoint_expanded` is not accepted.

- [ ] **Step 4: Split the block forward without changing the module tree**

Append `checkpoint_expanded: bool = False` to all three block constructors. Forward it from down/up blocks into `super().__init__`. Store it as a plain Boolean attribute.

Keep `self.layers` exactly as it is. Add this method to `MedNeXtBlock`:

```python
def _expanded_forward(self, x: Tensor) -> Tensor:
    x = self.layers.conv2(x)
    x = self.layers.act(x)
    return self.layers.conv3(x)
```

Replace `MedNeXtBlock.forward` with:

```python
def forward(self, x: Tensor) -> Tensor:
    residual = x
    x = self.layers.conv1(x)
    x = self.layers.norm(x)
    if self.checkpoint_expanded and self.training and torch.is_grad_enabled():
        x = grad_ckpt(self._expanded_forward, x, use_reentrant=False)
    else:
        x = self._expanded_forward(x)
    if self.res_block:
        x = x + residual
    return x
```

This preserves keys such as `layers.conv1.weight`, `layers.conv2.weight`, and `layers.conv3.weight`. Down/up blocks continue to call `super().forward`, so they receive the same selective boundary without duplicating execution code.

- [ ] **Step 5: Run the focused and existing tests**

Run:

```bash
python -m unittest test_mednext.ExpandedBranchCheckpointBlockTests -v
python -m unittest test_mednext.EvalApproximateGELUTests -v
```

Expected: all tests pass with exact block parity.

- [ ] **Step 6: Commit the block execution change**

```bash
git add mednext.py test_mednext.py
git commit -m "feat: checkpoint MedNeXt expanded branches"
```

## Task 3: Expose the model policy while preserving the old API

**Files:**
- Modify: `mednext.py`
- Modify: `test_mednext.py`

- [ ] **Step 1: Write model-level policy tests**

Build a tiny direct `MedNeXt` fixture with `num_blocks=[1, 1, 1]`, `expand_ratio=[2, 2, 2]`, `filters=2`, `kernel_size=3`, and 3D input `[1, 1, 16, 16, 16]`. Test:

- default and `checkpoint_style="none"` resolve to `model.checkpoint_style == "none"`;
- legacy `use_grad_checkpoint=True` resolves to `"block"` and keeps `model.use_grad_checkpoint is True`;
- `checkpoint_style="expanded"` sets every `MedNeXtBlock`, including down/up subclasses, to `checkpoint_expanded=True` and leaves `model.use_grad_checkpoint is False`;
- `checkpoint_style="block"` leaves every block’s internal selective flag false and sets `model.use_grad_checkpoint is True`;
- explicit policy plus legacy `True` raises `ValueError`;
- all factories accept and propagate the new option.

- [ ] **Step 2: Add state-dict and evaluation compatibility tests**

Construct `none`, `expanded`, and `block` versions from identical arguments. Load the `none` state dict strictly into the other two, then assert:

```python
self.assertEqual(list(reference.state_dict()), list(candidate.state_dict()))
candidate.eval()
reference.eval()
with torch.no_grad():
    torch.testing.assert_close(reference(x), candidate(x), rtol=0, atol=0)
```

Run this once with deep supervision disabled and once with it enabled. Evaluation must still return only the normal full-resolution output, matching current behavior.

- [ ] **Step 3: Add training-gradient parity tests**

Compare `none` against `expanded` on the tiny model in `float64`. Exercise both `deep_supervision=False` and `True`. Use the sum of squared outputs as the scalar loss and compare:

- output values;
- input gradient;
- all named parameter gradients.

Require `rtol=0, atol=0` first. If a supported PyTorch version changes reduction order, relax only gradient checks to `rtol=1e-12, atol=1e-12` and document the observed maximum error in the test comment.

- [ ] **Step 4: Run tests and verify the public option is absent**

Run: `python -m unittest test_mednext.MedNeXtCheckpointPolicyTests -v`

Expected: constructor errors for the new `checkpoint_style` argument.

- [ ] **Step 5: Wire the resolved policy into model construction**

Import `CheckpointStyle` and `resolve_checkpoint_style`. Append this argument to `MedNeXt` and each factory:

```python
checkpoint_style: Optional[CheckpointStyle] = None
```

Resolve once near the start of `MedNeXt.__init__`:

```python
self.checkpoint_style = resolve_checkpoint_style(
    use_grad_checkpoint, checkpoint_style
)
self.use_grad_checkpoint = self.checkpoint_style == "block"
checkpoint_expanded = self.checkpoint_style == "expanded"
```

Pass `checkpoint_expanded=checkpoint_expanded` to every normal, downsample, bottleneck, upsample, and decoder block. Keep `encode_with_ckpt`, `decode_with_ckpt`, and `decode_with_ds_ckpt` unchanged; only the `block` policy enters them through the existing `self.use_grad_checkpoint` condition.

Append the factory argument after `approximate_gelu_eval` and pass it by keyword into `MedNeXt`.

- [ ] **Step 6: Run all model tests**

Run: `python -m unittest test_mednext.py -v`

Expected: all tests pass, including exact state-dict and evaluation parity.

- [ ] **Step 7: Commit the public policy API**

```bash
git add mednext.py test_mednext.py
git commit -m "feat: expose MedNeXt checkpoint styles"
```

## Task 4: Make the profiler compare all three policies

**Files:**
- Modify: `profile_mednext.py`
- Modify: `run_profile_matrix.py`
- Modify: `summarize_profiles.py`

- [ ] **Step 1: Replace the required Boolean CLI with a backward-compatible mutually exclusive group**

In `profile_mednext.py`, import `CHECKPOINT_STYLES` and create one required group containing:

```python
checkpoint_group = parser.add_mutually_exclusive_group(required=True)
checkpoint_group.add_argument(
    "--checkpoint", dest="legacy_checkpoint", action="store_const", const=True
)
checkpoint_group.add_argument(
    "--no-checkpoint", dest="legacy_checkpoint", action="store_const", const=False
)
checkpoint_group.add_argument("--checkpoint-style", choices=CHECKPOINT_STYLES)
```

Resolve the CLI values before model construction:

```python
if args.checkpoint_style is None:
    effective_checkpoint_style = "block" if args.legacy_checkpoint else "none"
else:
    effective_checkpoint_style = args.checkpoint_style
args.checkpoint = effective_checkpoint_style == "block"
```

Pass only `checkpoint_style=effective_checkpoint_style` to the model. Add `effective_checkpoint_style` to `result["settings"]`. Keep the derived `settings.checkpoint` Boolean so existing summary consumers remain usable.

- [ ] **Step 2: Add expanded cases to the matrix runner**

Add:

```python
"eager_expanded": ["--checkpoint-style", "expanded"],
"compile_expanded": [
    "--checkpoint-style", "expanded", "--cudnn-benchmark", "--compile"
],
```

Keep the existing `eager_ckpt` and `compile_ckpt` names and commands. They remain whole-block compatibility cases.

- [ ] **Step 3: Group summaries by the effective policy**

Add `effective_checkpoint_style` to the selected settings columns and aggregation keys in `summarize_profiles.py`. Retain the old `checkpoint` column for old result directories.

For older summaries, derive the policy with:

```python
effective_style = settings.get(
    "effective_checkpoint_style",
    "block" if settings.get("checkpoint") else "none",
)
```

- [ ] **Step 4: Run CLI smoke checks**

Run:

```bash
python profile_mednext.py --help
python run_profile_matrix.py \
  --output /tmp/mednext-checkpoint-dry-run \
  --variant base --shape 1 1 128 128 128 --classes 3 \
  --precisions bf16 --cases compile compile_expanded compile_ckpt \
  --repeats 1 --dry-run
```

Expected: the dry run emits three commands using `none`, `expanded`, and whole-block policies, and existing `--checkpoint`/`--no-checkpoint` invocations remain valid.

- [ ] **Step 5: Run a CPU profiler smoke test**

Run:

```bash
python profile_mednext.py \
  --variant small --shape 1 1 16 16 --classes 3 --filters 2 \
  --precision fp32 --no-deep-supervision --checkpoint-style expanded \
  --device cpu --warmup 1 --steps 1 --profile-steps 0 \
  --output /tmp/mednext-checkpoint-profiler-smoke
```

Expected: exit code zero, `valid_training_timing: true`, and `effective_checkpoint_style: expanded` in `summary.json`.

- [ ] **Step 6: Commit profiler support**

```bash
git add profile_mednext.py run_profile_matrix.py summarize_profiles.py
git commit -m "feat: profile selective checkpoint policies"
```

## Task 5: Verify compile, AMP, deep supervision, and optimized operators

**Files:**
- Modify: `test_mednext.py`

- [ ] **Step 1: Add a CUDA-gated compiled training smoke test**

Guard with `@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")`. Build the tiny 3D model with `checkpoint_style="expanded"`, move it to CUDA, call `torch.compile(model, fullgraph=True)`, and run forward/backward under BF16 autocast. Assert finite output and finite gradients.

Use deep supervision in one invocation so the test exercises the existing stacked-head route together with internal block checkpointing.

- [ ] **Step 2: Test operator replacement composition**

In a CUDA-gated test, apply `replace_pointwise_convs` and `replace_highres_depthwise_convs` to a model configured with `checkpoint_style="expanded"`. Assert at least one replacement of each requested type, then run a finite forward/backward step.

Do not assert speed in unit tests. Performance belongs to the repeated matrix in Task 6.

- [ ] **Step 3: Run the complete suite in eager mode**

Run: `python -m unittest discover -v`

Expected: every CPU test passes; CUDA/Triton tests either pass or report an explicit skip when their requirements are unavailable.

- [ ] **Step 4: Run the compiled tests on each supported GPU environment**

Run: `python -m unittest test_mednext.MedNeXtCheckpointCudaTests -v`

Required environments:

- RTX 5090 with the current PyTorch/CUDA environment;
- L40S;
- RTX A6000;
- RTX 8000 using FP16 where BF16 is unsupported.

Record PyTorch, CUDA, cuDNN, Triton, driver, and GPU names from test/profiler output. A skip caused by a missing optional custom-kernel dependency is acceptable only for the composition test; the native selective checkpoint test must pass.

- [ ] **Step 5: Commit integration coverage**

```bash
git add test_mednext.py
git commit -m "test: cover selective checkpoint integration"
```

## Task 6: Run the decision benchmark and update the architecture record

**Files:**
- Modify: `ARCHITECTURE_MEMORY.md`
- Generated: `profile_results/checkpoint-*`

- [ ] **Step 1: Measure the native compiled Base model**

On RTX 5090, L40S, and RTX A6000 run:

```bash
python run_profile_matrix.py \
  --output "profile_results/checkpoint-native-$(hostname)" \
  --variant base --shape 1 1 128 128 128 --classes 3 8 \
  --precisions bf16 --cases compile compile_expanded compile_ckpt \
  --deep-supervision --repeats 3 --warmup 20 --steps 50 --profile-steps 3
```

On RTX 8000 run the same command with `--precisions fp16`.

- [ ] **Step 2: Measure composition with the current custom operators**

Run the same matrix in a separate directory with:

```bash
--pointwise-gemm --depthwise-split
```

Do not mix native and custom-operator runs in one output directory because the manifest intentionally rejects changed source/settings.

- [ ] **Step 3: Evaluate the release gates from medians**

For every GPU and class count, report:

- median full-step, forward, backward, loss, and optimizer time;
- maximum peak allocated memory;
- `expanded / none` time ratio;
- `expanded / none` memory ratio;
- `expanded / block` time ratio;
- `expanded / block` memory ratio;
- standard deviation and all three raw repeat values.

Hard correctness gates:

- all runs report finite outputs/gradients and the expected optimizer update count;
- no state-dict key differences;
- no compiled full-graph failure;
- no deep-supervision output change.

Performance decision gates on the RTX 5090 reference workload:

- expanded peak allocation is at most 60% of `none`;
- expanded median step time is at most 115% of `none`;
- expanded is faster than whole-block checkpointing.

Treat the other GPUs as portability data rather than forcing identical ratios. Keep the implementation if correctness holds and at least one supported architecture obtains a useful memory/time Pareto point; document architectures where whole-block or no checkpoint remains preferable.

- [ ] **Step 4: Update the roadmap with implementation status and final data**

In `ARCHITECTURE_MEMORY.md`:

- mark the policy contract as implemented;
- distinguish preliminary two-warmup/five-step measurements from the repeated results;
- add one table per GPU architecture for native and custom-operator configurations;
- state the exact torch/CUDA/cuDNN/Triton versions;
- record any compile graph failures or unsupported combinations;
- leave resolution-aware checkpointing as a separate future experiment.

- [ ] **Step 5: Run final verification**

Run:

```bash
python -m unittest discover -v
python run_profile_matrix.py \
  --output /tmp/mednext-final-dry-run \
  --variant base --shape 1 1 128 128 128 --classes 3 8 \
  --precisions bf16 --cases compile compile_expanded compile_ckpt \
  --repeats 3 --dry-run
```

Inspect `git diff --check` and verify that the model source contains no changes to existing parameter-bearing module paths.

- [ ] **Step 6: Commit the validated result record**

```bash
git add ARCHITECTURE_MEMORY.md profile_results/checkpoint-*
git commit -m "docs: record selective checkpoint benchmarks"
```

## Completion Criteria

The work is complete when:

- existing callers and checkpoints behave exactly as before;
- `checkpoint_style="expanded"` works through `MedNeXt` and every factory;
- only the expansion branch is recomputed in gradient-enabled training;
- block and model parity tests pass;
- eager, compiled, AMP, deep-supervision, pointwise-GEMM, and split-depthwise combinations are covered;
- profiler output identifies the effective policy without breaking old result summaries;
- repeated Base `128^3`, 3-class and 8-class results are recorded for the available GPU architectures;
- the measured data supports retaining or removing the public `expanded` policy under the stated gates.
