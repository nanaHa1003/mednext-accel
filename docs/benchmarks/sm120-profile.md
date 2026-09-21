# SM120 bootstrap optimization profile

The bundled `sm120` profile translates the repository's validated RTX 5090
measurements into the versioned operator/phase schema. It covers MedNeXt Base,
BF16 training at 128³ and preserves native inference and export paths. The
generic NVIDIA profile carries the same choices as explicit extrapolations so
unknown batches and GPU generations still receive a complete decision.

The source measurements used PyTorch 2.12 and 2.14 development builds, CUDA
13.x, BF16 autocast, AdamW, deep supervision, and compiled full-graph training.
For batch one, custom regular dW+dX, transpose dW, and pointwise selections
reduced the three-class Base step from 96.176 ms with original Conv3d operators
to 59.765 ms. With all-expansion checkpointing, batch-aware pointwise choices
measured 64.02, 149.11, 306.12, and 465.77 ms at batches 1, 2, 4, and 6, with
peak allocations of 3,762, 8,459, 16,729, and 21,992 MiB respectively.

The profile records measured regular depthwise split counts for batches 2, 4,
and 6. Other positive batches use per-shape work-scaling formulas. Pointwise
rules merge choices measured at batches 2 and 4 into closed intervals, so batch
3 no longer falls back merely because it was absent from a lookup table. The
two 128³ high-resolution pointwise shapes remain native above batch 4 on a
32-GiB card; a VRAM-qualified extrapolation permits GEMM at 48 GiB and above.

These results are synthetic model-only measurements. A local profile should be
generated when software versions, GPU architecture, input geometry, or memory
capacity differ materially:

```bash
mednext-accel profile
```

The command probes complete compiled steps in isolated processes to locate the
VRAM boundary, then validates and times candidate pointwise and depthwise
operators. It writes one merged profile under
`~/.cache/mednext_accel/profiles/`.
