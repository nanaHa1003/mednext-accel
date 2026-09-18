"""Activation-checkpoint configuration shared by MedNeXt models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CheckpointStyle = Literal["expansion", "block"]


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Select a checkpointing style and the resolution stages it covers."""

    style: CheckpointStyle = "expansion"
    stages: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.style not in ("expansion", "block"):
            raise ValueError("style must be 'expansion' or 'block'")
        if self.stages is None:
            return
        if not self.stages:
            raise ValueError("stages must be a non-empty tuple or None")
        if any(type(stage) is not int for stage in self.stages):
            raise ValueError("stages must contain only integers")
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("stages must not contain duplicates")
        if any(stage < 0 or stage > 4 for stage in self.stages):
            raise ValueError("stages must be between 0 and 4")
        object.__setattr__(self, "stages", tuple(sorted(self.stages)))

    def includes(self, stage: int) -> bool:
        """Return whether the selected policy covers ``stage``."""

        return self.stages is None or stage in self.stages

    @property
    def checkpoints_expansion(self) -> bool:
        """Return whether expanded branches are checkpointed."""

        return self.style == "expansion"

    @property
    def checkpoints_blocks(self) -> bool:
        """Return whether complete blocks are checkpointed."""

        return self.style == "block"
