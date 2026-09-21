# Direct implementation comparison on RTX 5090

## Scope

This benchmark compares the model implementations directly, without including
nnU-Net or MONAI data pipelines. It uses:

- official MedNeXt commit `0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba`;
- MONAI 1.5.2;
- MedNeXt-Accel commit containing the production optimized profile;
- PyTorch 2.12.0+cu132, CUDA 13.2, cuDNN 9.20;
- NVIDIA GeForce RTX 5090, compute capability 12.0;
- Base variant, batch one, input `[1, 1, 128, 128, 128]`, three classes;
- BF16 autocast, deep supervision, mean cross entropy across five output heads,
  AdamW with learning rate `1e-3`, and resident synthetic input/target tensors;
- 20 warmup steps followed by 50 measured optimizer steps per run;
- three independently constructed model runs per configuration;
- `torch.compile(mode="default", fullgraph=True)` for compiled rows;
- `torch.backends.cudnn.benchmark = True`.

Step time is the median CUDA-event duration of a complete zero-grad, forward,
loss, backward, and optimizer step. The reported result is the median of the
three run medians. Peak memory is the maximum PyTorch allocation across the
three runs after warmup. Compiler cold-start time is excluded.

## Direct results

| Implementation | Policy | Compiled | Checkpoint | Run medians (ms) | Median ± run SD | Peak allocated |
|---|---|---:|---|---|---:|---:|
| Official | Native | No | None | 100.109, 100.007, 99.893 | 100.007 ± 0.108 ms | 9,358 MiB |
| MONAI | Native | No | None | 99.773, 99.816, 99.778 | 99.778 ± 0.023 ms | 9,318 MiB |
| MedNeXt-Accel | PyTorch reference | No | None | 99.873, 100.104, 99.959 | 99.959 ± 0.117 ms | 9,355 MiB |
| MedNeXt-Accel | Conservative | No | None | 85.661, 85.987, 85.629 | 85.661 ± 0.198 ms | 9,355 MiB |
| Official | Native | Yes | None | 73.751, 73.977, 74.138 | 73.977 ± 0.194 ms | 8,339 MiB |
| MONAI | Native | Yes | None | 73.676, 73.979, 73.836 | 73.836 ± 0.152 ms | 8,298 MiB |
| MedNeXt-Accel | PyTorch reference | Yes | None | 74.141, 74.059, 73.949 | 74.059 ± 0.096 ms | 8,339 MiB |
| MedNeXt-Accel | Conservative | Yes | None | 59.343, 59.464, 59.584 | 59.464 ± 0.121 ms | 8,339 MiB |
| Official | Native | Yes | Whole block | 84.919, 84.933, 84.847 | 84.919 ± 0.046 ms | 3,236 MiB |
| MedNeXt-Accel | Conservative | Yes | Stage `(0,)` | 62.488, 62.726, 62.544 | 62.544 ± 0.125 ms | 5,795 MiB |
| MedNeXt-Accel | Conservative | Yes | Stages `(0, 1)` | 63.808, 63.777, 63.677 | 63.777 ± 0.069 ms | 4,358 MiB |
| MedNeXt-Accel | Conservative | Yes | All expansion stages | 64.261, 64.225, 64.261 | 64.261 ± 0.021 ms | 3,762 MiB |
| MedNeXt-Accel | Conservative | Yes | Whole block | 72.538, 72.589, 72.553 | 72.553 ± 0.026 ms | 3,235 MiB |

Whole-block checkpointing is exposed as `CheckpointConfig(style="block")`. It
matches the official policy's peak allocation within 1 MiB while taking 14.6%
less step time. All-expansion checkpointing remains the faster intermediate
point: it is 11.5% faster than MedNeXt-Accel whole-block at a 527 MiB memory
cost.

The official and MedNeXt-Accel reference implementations differ by only 0.1%
when compiled. This is a useful control: the 19.6% reduction from 73.977 to
59.464 ms appears only after selecting the custom depthwise path. MONAI is 0.2%
faster than official in this run but contains 15,457 fewer parameters because
its Base down-block expansion ratios differ.

The official Base factory does not enable checkpointing, although its underlying
model class accepts `checkpoint_style="outside_block"`; that explicit setting is
used for the whole-block row. The published Medium and Large factories enable
this policy. MONAI 1.5.2 has no activation-checkpointing argument or branch in
its MedNeXt model. MedNeXt-Accel supports expansion-branch and whole-block
policies, and can select resolution stages for either one.

## Reproduce

Clone the official implementation outside this repository, then run:

```bash
git clone --depth 1 https://github.com/MIC-DKFZ/MedNeXt /tmp/MedNeXt-official
python benchmarks/implementations.py \
  --official-root /tmp/MedNeXt-official \
  --output artifacts/implementation-comparison.json \
  --warmup 20 --steps 50 --repeats 3
```

The script records the exact official commit, installed MONAI/PyTorch/CUDA/cuDNN
versions, raw per-run timings, memory, compile settings, and workload. Generated
JSON remains under `artifacts/`, which Git ignores.

## Interpretation limits

These are synthetic model-level training measurements, not end-to-end
application throughput. Data loading, augmentation, distributed communication,
losses used by a real task, and host-to-device transfer are excluded. MONAI's
architectural difference makes its timing informative but not an exact
parameter-for-parameter comparison. Results should be repeated on other GPU
architectures; the standalone profiler exists because the best backend can
change with hardware and software versions.
