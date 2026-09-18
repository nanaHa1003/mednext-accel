# Direct implementation comparison on RTX 5090

## Scope

This benchmark compares the model implementations directly, without including
nnU-Net or MONAI data pipelines. It uses:

- official MedNeXt commit `0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba`;
- MONAI 1.5.2;
- MedNeXt-Accel commit containing the production conservative policy;
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
| Official | Native | Yes | Whole block | 84.772, 85.156, 85.381 | 85.156 ± 0.308 ms | 3,236 MiB |
| MedNeXt-Accel | Conservative | Yes | Stage `(0,)` | 62.833, 62.818, 63.025 | 62.833 ± 0.115 ms | 5,795 MiB |
| MedNeXt-Accel | Conservative | Yes | Stages `(0, 1)` | 63.584, 63.981, 64.077 | 63.981 ± 0.261 ms | 4,359 MiB |
| MedNeXt-Accel | Conservative | Yes | All expansion stages | 64.528, 64.287, 64.279 | 64.287 ± 0.142 ms | 3,763 MiB |

The official and MedNeXt-Accel reference implementations differ by only 0.1%
when compiled. This is a useful control: the 19.6% reduction from 73.977 to
59.464 ms appears only after selecting the custom depthwise path. MONAI is 0.2%
faster than official in this run but contains 15,457 fewer parameters because
its Base down-block expansion ratios differ.

The official Base factory does not enable checkpointing, although its underlying
model class accepts `checkpoint_style="outside_block"`; that explicit setting is
used for the whole-block row. The published Medium and Large factories enable
this policy. MONAI 1.5.2 has no activation-checkpointing argument or branch in
its MedNeXt model. MedNeXt-Accel checkpoints only the expanded branch and can
select resolution stages independently.

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
architectures; the per-shape autotuner exists because the best backend can
change with hardware and software versions.
