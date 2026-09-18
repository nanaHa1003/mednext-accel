"""Activation-checkpoint execution policies shared by model architectures."""
from typing import Literal, Optional, Tuple, cast


CheckpointStyle = Literal["none", "expanded", "block"]
CHECKPOINT_STYLES: Tuple[CheckpointStyle, ...] = ("none", "expanded", "block")
CheckpointLevels = Optional[Tuple[int, ...]]


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


def resolve_checkpoint_levels(
    checkpoint_style: CheckpointStyle,
    checkpoint_levels: CheckpointLevels,
    depth: int,
) -> CheckpointLevels:
    """Validate and canonicalize static resolution-level selection."""
    if checkpoint_levels is None:
        return None
    if checkpoint_style != "expanded":
        raise ValueError(
            "checkpoint_levels is only valid with checkpoint_style='expanded'"
        )
    if not isinstance(checkpoint_levels, tuple) or not checkpoint_levels:
        raise ValueError("checkpoint_levels must be a non-empty tuple of integers")
    if any(type(level) is not int for level in checkpoint_levels):
        raise ValueError("checkpoint_levels must contain only integers")
    if len(set(checkpoint_levels)) != len(checkpoint_levels):
        raise ValueError("checkpoint_levels must not contain duplicates")
    if any(level < 0 or level > depth for level in checkpoint_levels):
        raise ValueError(f"checkpoint_levels must be between 0 and {depth}")
    return tuple(sorted(checkpoint_levels))
