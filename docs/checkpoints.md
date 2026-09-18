# Checkpoint compatibility

The native state dict uses stable names such as
`encoder_stages.0.0.depthwise.weight`. Optimization and activation
checkpointing preserve the same parameter objects and keys.

```python
from mednext_accel import mednext_base
from mednext_accel.checkpoints import load_checkpoint

model = mednext_base(in_channels=1, out_channels=3, deep_supervision=True)
report = load_checkpoint(model, "checkpoint.pt", source="auto", strict=True)
print(report)
```

The loader accepts a path, a raw state dict, or a mapping under `state_dict`,
`model_state_dict`, `network_weights`, or `model`. It removes common wrapper
prefixes including `module.` and `_orig_mod.`. File loading uses
`torch.load(..., weights_only=True)`.

| Source | Target | Result |
|---|---|---|
| Native MedNeXt-Accel | Matching native model | Lossless |
| Official MedNeXt v1 S/B/M/L | Matching official-compatible factory | Lossless key conversion; `dummy_tensor` ignored |
| MONAI Small | `monai_mednext_small` or native Small | Lossless |
| MONAI Base/Medium/Large | Factory in `mednext_accel.compat.monai` | Lossless |
| MONAI Base/Medium/Large | Default official-compatible factory | Rejected before loading |

MONAI Base, Medium, and Large use the current encoder expansion ratio in the
downsampling blocks. The official architecture uses the next stage's ratio.
This changes parameter shapes in early downsampling blocks, so key renaming
alone cannot make those checkpoints compatible.

```python
from mednext_accel.compat.monai import monai_mednext_base

model = monai_mednext_base(in_channels=1, out_channels=3)
load_checkpoint(model, monai_state_dict, source="monai")
```

Strict loading never pads, truncates, or silently initializes tensors. Configure
the same input channels, output classes, kernel size, base channels, variant,
and deep-supervision heads as the checkpoint.

Compilation does not change checkpoint tensors. Prefer `model.compile()` and
save `model.state_dict()`. A state dict obtained from the wrapper returned by
`torch.compile(model)` may contain `_orig_mod.` prefixes; the loader accepts it.
