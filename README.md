# MedNeXt-Accel

Accelerated and production-ready MedNeXt model architectures for PyTorch.

The package is under active development. Its core is a pure-PyTorch MedNeXt v1
implementation with optional Triton acceleration, historical checkpoint import,
selective activation checkpointing, and portable evaluation export.

```python
from mednext_accel import mednext_base

model = mednext_base(in_channels=1, out_channels=3)
```

MedNeXt-Accel provides model architecture code only. It does not include a
trainer, dataset pipeline, loss functions, or an nnU-Net dependency.

