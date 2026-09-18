"""Activation-checkpoint execution policies shared by model architectures."""
from typing import Literal, Optional, Tuple, cast


CheckpointStyle = Literal["none", "expanded", "block"]
CHECKPOINT_STYLES: Tuple[CheckpointStyle, ...] = ("none", "expanded", "block")


def resolve_checkpoint_style(
    use_grad_checkpoint: bool,
    checkpoint_style: Optional[CheckpointStyle],
) -> CheckpointStyle:
    """Resolve the current policy while preserving the legacy Boolean API."""
    if checkpoint_style is None:
        return "block" if use_grad_checkpoint else "none"
    if checkpoint_style not in CHECKPOINT_STYLES:
        raise ValueError(
            f"checkpoint_style must be one of {CHECKPOINT_STYLES}, "
            f"got {checkpoint_style!r}"
        )
    if use_grad_checkpoint:
        raise ValueError(
            "use_grad_checkpoint=True cannot be combined with checkpoint_style; "
            "use checkpoint_style='block' instead"
        )
    return cast(CheckpointStyle, checkpoint_style)
