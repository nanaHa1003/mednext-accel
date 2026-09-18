"""Activation-checkpoint configuration shared by MedNeXt models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Select expansion-branch checkpointing by resolution stage."""

    expansion: bool = True
    stages: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.stages is None:
            return
        if not self.expansion:
            raise ValueError("stages require expansion checkpointing to be enabled")
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
        """Return whether the expansion branch at ``stage`` is checkpointed."""

        return self.expansion and (self.stages is None or stage in self.stages)
