# Optimization and compilation

Model construction is pure PyTorch and has no benchmark or cache side effects.
Apply an execution policy after moving the model to its training device, and
before constructing an optimizer, compiling, or wrapping with DDP:

```python
import torch
from mednext_accel import mednext_base
from mednext_accel.optimization import optimize

model = mednext_base(in_channels=1, out_channels=3).cuda()
report = optimize(
    model,
    input_shape=(1, 1, 128, 128, 128),
    dtype=torch.bfloat16,
    policy="autotune",
    compile_mode="default",
)
model.compile(mode=report.compile_mode, fullgraph=True)
```

`torch` disables all installed optional backends. `conservative` selects only
the BF16, batch-one, 128-cubed depthwise configurations validated end to end on
an SM 12.0 GPU. Other hardware and shapes stay on PyTorch. `autotune` measures
each unique eligible shape on the current device. Its cache key includes model
topology, device identity and capability, dtype, input shape, PyTorch, CUDA,
cuDNN, Triton, package and kernel versions, and the planned compile mode.

The default cache is `~/.cache/mednext_accel/autotune-v1.json`, or beneath
`XDG_CACHE_HOME` when set. Pass `cache_path=None` to disable it. Writes use an
atomic replacement, and a corrupt or incompatible cache is ignored and rebuilt.

Stride-two downsample dX remains excluded unless
`include_stride2_input_grad=True`; its isolated kernel was faster on RTX 5090,
but the extra integration boundary made the complete training step slower.

## torch.compile modes

`compile_mode` records the mode intended for the subsequent `model.compile`
call. It does not compile the model. Supported values match PyTorch:

- `default` balances compile time, steady-state speed, and memory.
- `reduce-overhead` uses CUDA Graphs where possible and may retain more device
  memory.
- `max-autotune` benchmarks more generated kernels and enables CUDA Graphs on
  GPU. It has the largest cold compile cost.
- `max-autotune-no-cudagraphs` performs the broader kernel search without CUDA
  Graphs.

Use the same mode for every row in a performance comparison, warm up compiled
graphs before timing, and report cold compilation separately. `fullgraph=True`
controls graph-break handling and is independent of these modes.

## Checkpoints

Compilation does not change parameter values or shapes. Prefer the in-place
module API and save the ordinary model state dict:

```python
model.compile(mode="default", fullgraph=True)
torch.save(model.state_dict(), "weights.pt")
```

The wrapper returned by `compiled = torch.compile(model)` may prefix state-dict
keys with `_orig_mod.`. `mednext_accel.checkpoints.load_checkpoint` removes that
prefix when reading and unwraps a compiled target when writing, so eager and
compiled-wrapper checkpoints load in either direction. For portable application
code, save and load the original model rather than depending directly on the
wrapper's private `_orig_mod` attribute. Compiled kernels live in compiler caches
and are not part of the checkpoint.
