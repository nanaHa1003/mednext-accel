# Benchmarks

These scripts operate on the installed `mednext_accel` package and write no
files unless an output path is supplied.

```bash
python benchmarks/autotune.py \
  --variant base --shape 1 1 128 128 128 --classes 3 \
  --dtype bf16 --compile-mode default \
  --output artifacts/autotune-rtx5090.json

python benchmarks/depthwise.py \
  --kind regular --batch-size 2 --channels 32 --spatial 128

python benchmarks/pointwise.py \
  --batch-size 2 --in-channels 32 --out-channels 96 --spatial 64 64 64

# Direct official/MONAI/package comparison (requires an official checkout)
python benchmarks/implementations.py \
  --official-root /tmp/MedNeXt-official \
  --output artifacts/implementation-comparison.json

# Batch-size memory scaling across checkpoint policies
python benchmarks/batch_memory.py \
  --batch-sizes 1 2 4 8 \
  --policy torch \
  --warmup 5 --steps 20 \
  --output artifacts/batch-memory.json
```

Use `--policy torch` to isolate batch and checkpoint scaling with one backend.
Use `--policy conservative` to measure production dispatch, including its safe
fallback for shapes outside the validated custom-kernel envelope. Results include
median step time, samples per second, milliseconds per sample, and allocated and
reserved CUDA memory.

Run benchmarks on an otherwise idle GPU. Record the GPU, driver, PyTorch, CUDA,
cuDNN, Triton, dtype, compile mode, input shape, warmup, and repetitions with
every result. Generated output belongs under `artifacts/`, which Git ignores.
