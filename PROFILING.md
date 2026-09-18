# MedNeXt training profiling package

SSH into your GPU host and use an existing CUDA-enabled PyTorch environment.
The scripts install no packages. By default they preserve all model layers;
the optional `--pointwise-gemm` prototype replaces eligible pointwise layers.
Deep-supervision outputs are preserved in either case.
The scripts were tested with Python 3.11, PyTorch 2.12.0/2.14.0+cu132 and an RTX 5090.
Other GPU/software combinations must be measured; compilation failures are recorded.

## Files

- `mednext.py`: original model, required alongside the scripts.
- `profile_mednext.py`: one training configuration with timing and optional trace.
- `run_profile_matrix.py`: sequential comparisons in separate child processes.
- `summarize_profiles.py`: per-run and repeated-run CSVs and kernel attribution.
- `KERNEL_NOTES.md`: measured normalization and depthwise-backward findings.
- `pointwise_gemm.py`: optional dense stride-one 1x1x1 Conv3d GEMM prototype.
- `test_pointwise_gemm.py`: CPU output, gradient, and checkpoint compatibility tests.
- `validate_pointwise_gemm.py`: CUDA FP32/BF16 eager/compiled numerical comparisons.
- `depthwise_split.py`: opt-in Triton kernels for regular and transpose depthwise dW/dX.
- `test_depthwise_split.py`: CUDA boundary, dtype, forward, and gradient checks.
- `benchmark_depthwise_split.py`: isolated dW tuning against ATen.
- `benchmark_depthwise_stride2.py`: native dX/dW/dB timing for stride-two depthwise layers.
- `kernel_autotune.py`: per-shape implementation selection with optional caching.
- `test_kernel_autotune.py`: checkpoint and native-fallback policy checks.
- `requirements.txt`: unpinned Python dependencies; choose the PyTorch/CUDA build separately.
- `test_mednext.py`: eval-only approximate-GELU and checkpoint compatibility tests.

## Experimental regular depthwise dW + dX

Pass `--depthwise-split` to either profiling CLI. The optimized runtime dispatch is
deliberately narrow: contiguous batch-one cubic input with `(channels, spatial size)`
of `(32,128)`, `(64,64)`, `(128,32)`, `(256,16)`, or `(512,8)`, a positive odd
cubic kernel, stride/dilation 1, and symmetric same padding. All 18 matching Base
modules are wrapped. Downsampling, unmeasured transpose shapes, and other input
shapes retain PyTorch's convolution path. The measured C64, 64-cubed transpose-depthwise block
is also wrapped; its dW uses an FP32 split reduction while dX and dB remain native.
Forward and bias-gradient computation use PyTorch; regular dW uses an FP32 split
reduction and regular dX uses a direct Triton stencil kernel.

The stride-two downsample dX prototype is available separately with
`--depthwise-stride2-dx`. It is opt-in because its isolated kernel is faster but
the current end-to-end compiled graph was slower after the native dW/custom dX
split. Keep this flag off for the default measured configuration.
Kernel sizes 1, 3, 5, and 7 have correctness coverage; 3 and 5 have been benchmarked
end to end on the RTX 5090.

```bash
python -m unittest test_depthwise_split
python benchmark_depthwise_split.py \
  --output results/depthwise_standalone.json --dtype bf16 --kernel-size 5
python run_profile_matrix.py --output results/depthwise \
  --precisions bf16 --cases autotune compile --no-include-fp32-baseline \
  --repeats 3 --warmup 20 --steps 50 --profile-steps 3 \
  --kernel-size 5 --depthwise-split
```

The tuning result is architecture-specific. Retune `splits`, `block`, and precision
on each GPU. The altered parallel reduction order changes rounding. Validate real
training convergence before adopting it.

## Per-shape implementation selection

`kernel_autotune.py` benchmarks one representative layer for every unique eligible
shape in a selected model. It compares native Conv3d/ConvTranspose3d with the
pointwise GEMM, regular depthwise, transpose dW, and optional downsample dX paths.
The benchmark includes the forward and relevant backward components. Results can be
cached by GPU, software version, dtype, input shape, model signature, and selector
source hash.

```python
import torch
import mednext
from kernel_autotune import benchmark_kernel_choices, apply_kernel_policy

model = mednext.mednext_base(
    spatial_dims=3, in_channels=1, out_channels=3,
    deep_supervision=True).cuda()
report = benchmark_kernel_choices(
    model, (1, 1, 128, 128, 128), dtype=torch.bfloat16,
    cache_path='results/base_rtx5090_bf16.json')
apply_kernel_policy(model, report)
```

The same operation is available from the command line:

```bash
python kernel_autotune.py --output results/base_kernel_policy.json \
  --dtype bf16 --warmup 10 --repetitions 50
```

The model must be on CUDA and the current selector requires batch one. Policy
application preserves parameter objects and state-dict keys. The downsample dX
choice is disabled unless `include_stride2_input_grad=True` is passed to the helper;
the command-line profiling equivalent is `--depthwise-stride2-dx`.

On RTX 5090 with PyTorch 2.14, pointwise GEMM, compilation, regular dW+dX, and
transpose dW, full-model medians were 59.76 ms for 3 classes and 62.70 ms for
8 classes. The matching original-Conv3d medians were 96.18 and 97.61 ms.

Measure stride-two depthwise headroom before implementing another kernel:

```bash
python benchmark_depthwise_stride2.py \
  --output results/stride2_components.json --dtype bf16
```

Its defaults match the largest Base downsample (`C=32`, input `128^3`) and
transpose-depthwise layer (`C=64`, input `64^3`). It reports native dX, dW, dB,
and combined backward timings independently.

To reproduce the downsample dX integration experiment, add
`--depthwise-stride2-dx` to the profiling command. On the RTX 5090 it raised the
3-class median from about 59.1 ms to 60.4 ms, despite the isolated C32/128³ dX
kernel improving from 1.253 to 0.578 ms.

## Eval-only approximate GELU

All model factories accept `approximate_gelu_eval=True`. Training continues to use
exact GELU; after `model.eval()`, GELU uses PyTorch's tanh approximation. The option
adds no parameters and preserves checkpoint keys:

```python
model = mednext_base(
    spatial_dims=3, in_channels=1, out_channels=3,
    approximate_gelu_eval=True)
```

On RTX 5090/PyTorch 2.14, three matched compiled BF16 inference comparisons found
no repeatable speedup: median changes were +0.90%, -0.22%, and -0.66%. Keep the
option disabled unless the target backend measures a benefit. The tested final
logits differed by 0.673% relative L2, so task-level validation is still required.

## Experimental pointwise GEMM

Pass `--pointwise-gemm` to either profiling CLI. This preserves parameter
objects and state-dict keys and replaces 53 eligible Conv3d layers in Base.
Depthwise, strided, grouped, and transposed convolutions retain their original
implementation. The module uses addmm for batch size one and batched matmul
otherwise. No handwritten CUDA or autograd is required. Apply the converter
before compilation, distributed wrapping, and optimizer construction:

```python
from pointwise_gemm import replace_pointwise_convs
count = replace_pointwise_convs(model)
```

```bash
python -m unittest test_pointwise_gemm
python validate_pointwise_gemm.py --output results/gemm_validation.json
python run_profile_matrix.py --output results/gemm \
  --precisions bf16 --cases autotune compile --no-include-fp32-baseline \
  --repeats 3 --warmup 20 --steps 50 --profile-steps 3 --pointwise-gemm
```

Use a separate output directory without the flag for the Conv3d control.
Floating-point rounding can differ, especially with autocast; numerical checks
do not establish convergence equivalence. Results on other GPUs are unmeasured.

## Quick start over SSH

In a fresh environment, install the unpinned Python dependencies first. Select
the appropriate PyTorch wheel or package index for the target GPU separately;
the requirements file deliberately does not choose a CUDA or PyTorch version:

```bash
python -m pip install -r requirements.txt
```

```bash
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name())'
python run_profile_matrix.py --output results/this_gpu
```

Defaults match MedNeXt Base, input `[1,1,128,128,128]`, 32 initial filters, kernel
size 3, 3 and 8 classes, and deep supervision enabled. Change the batch size or
input channels by changing the first two numbers of `--shape`.

The default matrix includes:

1. A checkpointed FP32 baseline (cuDNN TF32 follows PyTorch's default).
2. Eager mixed precision with checkpointing.
3. Eager mixed precision without checkpointing.
4. No checkpointing plus cuDNN autotuning.
5. The same plus channels-last memory format.
6. Autotuning plus default `torch.compile`, contiguous layout, no checkpointing.
7. Autotuning plus default `torch.compile`, contiguous layout, with checkpointing.

`--precisions auto` selects BF16 and FP16 on compute capability >= 8, or FP16
on older devices. FP16 uses GradScaler. Explicit BF16 on older devices is rejected.
This selects supported precision candidates, not a predicted best configuration.
L40S and RTX A6000 get both mixed precisions; Quadro RTX 8000 gets FP16.

Each configuration has three repeats in independent processes, 20 warmup steps,
and 50 timed steps. Only the first repeat captures a three-step trace. Defaults
produce 78 child runs on a BF16-capable GPU and 42 on Turing. Compilation time is
excluded from timings but counts toward the 1800-second child timeout. Shared
compiler caches can reduce startup time for later repeats. Expect substantial
runtime and potentially gigabytes of trace files; `--profile-steps 0` disables
traces while retaining timing and memory results.

For a shorter first pass:

```bash
python run_profile_matrix.py --output results/quick \
  --classes 3 --repeats 1 --precisions auto \
  --cases eager_ckpt autotune compile
```

For one precision, a different batch size, and longer timing windows:

```bash
python run_profile_matrix.py --output results/batch2 \
  --shape 2 1 128 128 128 --classes 3 8 --precisions bf16 \
  --repeats 3 --warmup 30 --steps 100
```

To preview commands without accessing a GPU or writing results:

```bash
python run_profile_matrix.py --output results/preview --dry-run --dry-run-cc 7.5
```

`--dry-run-cc` is only a simulated capability for planning. Actual execution always
detects the visible GPU. `--tf32 off` gives a stricter FP32 baseline if desired;
record this distinction when comparing results. `--seed` and `--lr` are exposed.

## Select a GPU and collect results

Copy `mednext_profiling.tar.gz` to the remote host, extract it, and activate your
existing PyTorch environment. For example, replacing `GPU_HOST` with your SSH host:

```bash
scp mednext_profiling.tar.gz GPU_HOST:~/
ssh GPU_HOST
mkdir -p ~/mednext-profiling
tar -xzf ~/mednext_profiling.tar.gz -C ~/mednext-profiling
cd ~/mednext-profiling
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python run_profile_matrix.py --output results/l40s
```

Use `results/a6000` or `results/rtx8000` on those hosts. The output label is only
for organization: actual GPU identity and native precision support are detected.
Select another GPU by changing `CUDA_VISIBLE_DEVICES` to its index or UUID. One
process uses logical CUDA device 0 from the visible devices. Avoid overlapping
two matrices on the same GPU. For a long SSH session, run inside `tmux` if desired.

Copy each host's results directory back to a separate subdirectory locally, then
run `python summarize_profiles.py results` to combine the reports. This is a
single-GPU study; it does not measure distributed training.

## Outputs, failures, and resume

Every child directory has `run.log` and `status.json`. Successful timed runs have
`summary.json`; the first repeat also has `trace.json`, `operators.txt`, and
`kernel_breakdown.json` when trace collection is enabled.

- `manifest.json`: commands, GPU/driver/software environment, and source hashes.
- `comparison.csv`: all available individual summaries, including validity/status.
- `aggregate.csv`: median/min/max/stdev of per-run mean step time and peak memory
  across matching, successful, valid repeats. Hardware, host, software, model hash,
  shapes, precision, TF32 settings, and timing settings are kept separate.
- `kernel_breakdown.json`: kernel durations attributed once to launching CPU
  operators and input shapes via trace External id. GPU annotation ranges are
  excluded. Includes a clearly labeled GroupNorm fused-kernel name-matching proxy.

Open traces in a local Chrome tracing-compatible viewer such as Perfetto. Raw
operator tables include hierarchical ranges and kernels; do not sum all rows.
Compiler kernel names are version-dependent, and fused kernels cannot be uniquely
split into separate normalization, cast, and neighboring-operation costs.

OOMs, nonzero exits, and compilation errors are logged, and the matrix continues.
A timeout kills the child's process group to avoid leaving compiler subprocesses.
The runner exits nonzero if any case fails. Results with nonfinite final
loss/gradients or skipped timed optimizer updates are invalid and excluded from
aggregates. Inspect failed FP16 runs: increase warmup if scaling is still settling;
do not accept skipped-step timing as an optimization.

Resume with exactly the same command plus `--resume`. Successful children are
skipped; failed children are retried. Commands, source hashes, and environment
must match, including the host/GPU inventory. On a different host, use
a new output directory so measurements are not silently mixed.

Regenerate or combine summaries after copying results back:

```bash
python summarize_profiles.py results
```

## Measurement limits

The model trains on fixed synthetic resident inputs and targets. The loss is the
mean cross-entropy over all five full-resolution supervision heads; AdamW uses
learning rate 0.001. Replace `loss_fn` and optimizer setup with your actual training
logic for representative loss/optimizer costs. Changing these files changes the
manifest hash, so start a new result directory.

No disk I/O, augmentation, H2D input transfer, or distributed communication is
included. Wall step timing uses synchronization only around the full window;
CUDA events measure phases separately. Instrumented traces are collected after
timing. Trace kernel totals are not identical to uninstrumented wall time.
Compilation/autotune startup and final numerical checks are outside the timer.
Memory is PyTorch peak allocated/reserved memory, not total `nvidia-smi` usage.

The final numerical checks detect gross failures, not model equivalence or
convergence. Precision, layout, compilation, and backend selection may alter
rounding and training trajectories. Compare outputs/gradients and validation
quality before selecting a production configuration. CPU mode on the single-run
script is only a smoke test; it cannot establish GPU optimization priorities.
