# Benchmarks

The standard hardware-tuning workflow is:

```bash
mednext-accel profile
```

It produces one reusable profile and isolates CUDA OOM in child processes. Use a
campaign YAML to restrict variants, input shapes, checkpoint contexts, or batch
search. Generated profiles and raw output belong under `artifacts/` or the user
cache and are ignored by Git.

Focused developer benchmarks remain available:

```bash
python benchmarks/depthwise.py --kind regular --phase backward_weight \
  --batch-size 2 --channels 32 --spatial 128
python benchmarks/pointwise.py --batch-size 2 --in-channels 32 \
  --out-channels 96 --spatial 64 64 64
python benchmarks/batch_memory.py --batch-sizes 1 2 4 8 \
  --optimization auto --output artifacts/batch-memory.json
```

Run benchmarks on an otherwise idle GPU and record GPU, driver, PyTorch, CUDA,
cuDNN, Triton, dtype, compile mode, shape, warmup, and repetitions.
