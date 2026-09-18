# Profiling tools

Profile one forward/backward configuration from an installed checkout:

```bash
python tools/profile_train_step.py \
  --output artifacts/rtx5090-base-default \
  --variant base --shape 1 1 128 128 128 --classes 3 \
  --dtype bf16 --policy conservative --compile-mode default \
  --warmup 20 --steps 50 --profile-steps 3
```

Use a separate process and output directory for every policy, dtype, compile
mode, class count, and repeat. `profile_matrix.py` creates that matrix:

```bash
python tools/profile_matrix.py \
  --output artifacts/rtx5090-matrix \
  --policies torch conservative autotune \
  --compile-modes default max-autotune-no-cudagraphs max-autotune
```

The per-run output contains `summary.json`,
`operators.txt`, and `trace.json`. Summarize several runs with:

```bash
python tools/summarize_profiles.py artifacts/ --output artifacts/summary.csv
```

`--compile-mode none` measures eager PyTorch. The other values are passed to
`model.compile(mode=..., fullgraph=...)`. Compilation and graph warmup happen
before timing. The summary records allocated and reserved device memory; it does
not claim total board memory or memory used by unrelated processes.
