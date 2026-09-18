"""Public checkpoint conversion and loading API."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor, nn

from ..models.config import MedNeXtV1Config
from . import monai, official_v1

CheckpointSource = Literal["auto", "native", "official-v1", "official_v1", "monai"]


@dataclass(frozen=True, slots=True)
class CheckpointLoadReport:
    """A deterministic summary of a checkpoint load operation."""

    source: Literal["native", "official_v1", "monai"]
    loaded_keys: tuple[str, ...]
    ignored_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _strip_wrapper_prefix(key: str) -> str:
    prefixes = ("module.", "_orig_mod.", "model.", "network.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def _detect_source(keys: list[str]) -> Literal["native", "official_v1", "monai"]:
    if any(key.startswith(("enc_block_", "down_", "up_", "dec_block_")) for key in keys):
        return "official_v1"
    if any(key.startswith(("enc_stages.", "down_blocks.", "up_blocks.")) for key in keys):
        return "monai"
    if any(key.startswith(("encoder_stages.", "downsamples.", "upsamples.")) for key in keys):
        return "native"
    raise ValueError("could not detect checkpoint source from its parameter names")


def _convert(
    state_dict: Mapping[str, Tensor],
    *,
    source: CheckpointSource,
) -> tuple[OrderedDict[str, Tensor], str, tuple[str, ...]]:
    stripped = OrderedDict((_strip_wrapper_prefix(key), value) for key, value in state_dict.items())
    resolved = _detect_source(list(stripped)) if source == "auto" else source
    if resolved == "official-v1":
        resolved = "official_v1"
    if resolved not in ("native", "official_v1", "monai"):
        raise ValueError(
            "source must be 'auto', 'native', 'official-v1', 'official_v1', or 'monai'"
        )

    converted: OrderedDict[str, Tensor] = OrderedDict()
    ignored: list[str] = []
    for key, value in stripped.items():
        if resolved == "official_v1":
            destination = official_v1.convert_key(key)
            if destination is None:
                ignored.append(key)
                continue
        elif resolved == "monai":
            destination = monai.convert_key(key)
        else:
            destination = key
        if destination in converted:
            raise ValueError(f"multiple checkpoint keys map to {destination!r}")
        converted[destination] = value
    return converted, resolved, tuple(ignored)


def convert_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    source: CheckpointSource,
    target_config: MedNeXtV1Config | None = None,
) -> OrderedDict[str, Tensor]:
    """Return a state dict translated to the stable native parameter schema."""

    converted, resolved, _ = _convert(state_dict, source=source)
    _validate_compatibility(resolved, target_config)
    return converted


def _validate_compatibility(
    source: str,
    target_config: MedNeXtV1Config | None,
) -> None:
    if target_config is None or target_config.variant == "small":
        return
    if source == "monai" and target_config.compatibility != "monai":
        raise ValueError(
            "MONAI Base/Medium/Large checkpoints require a MONAI-compatible model; "
            "down_0 and down_1 expansion widths differ from official MedNeXt v1"
        )
    if source == "official_v1" and target_config.compatibility == "monai":
        raise ValueError(
            "official Base/Medium/Large checkpoints require an official-compatible "
            "model; down_0 and down_1 expansion widths differ from MONAI"
        )


def _extract_state_dict(
    checkpoint: Mapping[str, object],
) -> Mapping[str, Tensor]:
    if checkpoint and all(isinstance(value, Tensor) for value in checkpoint.values()):
        return cast(Mapping[str, Tensor], checkpoint)
    for key in ("state_dict", "model_state_dict", "network_weights", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and all(isinstance(item, Tensor) for item in value.values()):
            return cast(Mapping[str, Tensor], value)
    raise ValueError("checkpoint does not contain a recognizable tensor state dict")


def _unwrap_compile_target(model: nn.Module) -> nn.Module:
    """Return the original module behind a ``torch.compile`` wrapper."""

    original = getattr(model, "_orig_mod", None)
    if not isinstance(original, nn.Module):
        return model
    keys = tuple(model.state_dict())
    if keys and all(key.startswith("_orig_mod.") for key in keys):
        return original
    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint: Mapping[str, object] | str | PathLike[str],
    *,
    source: CheckpointSource = "auto",
    strict: bool = True,
) -> CheckpointLoadReport:
    """Load a native, official-v1, or MONAI checkpoint into ``model``."""

    if isinstance(checkpoint, (str, PathLike)):
        loaded = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if not isinstance(loaded, Mapping):
            raise ValueError("checkpoint file must contain a mapping")
        checkpoint = loaded
    state_dict = _extract_state_dict(checkpoint)
    converted, resolved, ignored = _convert(state_dict, source=source)
    target = _unwrap_compile_target(model)
    config = getattr(target, "config", None)
    _validate_compatibility(
        resolved,
        config if isinstance(config, MedNeXtV1Config) else None,
    )
    incompatible = target.load_state_dict(converted, strict=strict)
    return CheckpointLoadReport(
        source=cast(Literal["native", "official_v1", "monai"], resolved),
        loaded_keys=tuple(converted),
        ignored_keys=ignored,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )
