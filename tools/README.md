# Profiling tools

Use `mednext-accel profile` to generate a runtime optimization profile. The
repository tools collect deeper traces for development and published evidence.

```bash
python tools/profile_train_step.py \
  --output artifacts/rtx5090-base-auto \
  --variant base --shape 1 1 128 128 128 --classes 3 \
  --dtype bf16 --optimization auto --compile-mode default

python tools/profile_matrix.py \
  --output artifacts/rtx5090-matrix \
  --optimizations reference auto \
  --compile-modes default max-autotune-no-cudagraphs
```

Each run writes `summary.json`, `operators.txt`, and optionally `trace.json`.
`summarize_profiles.py` combines summaries into CSV. These files belong under
`artifacts/`, which Git ignores.
