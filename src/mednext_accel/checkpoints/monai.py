"""Key translation for MONAI's MedNeXt implementation."""

from __future__ import annotations

import re


_COMPONENTS = {
    "conv1": "depthwise",
    "conv2": "expand",
    "conv3": "project",
    "res_conv": "residual",
}


def convert_key(key: str) -> str:
    """Convert one MONAI state-dict key to the native schema."""

    replacements = (
        ("enc_stages.", "encoder_stages."),
        ("down_blocks.", "downsamples."),
        ("up_blocks.", "upsamples."),
        ("dec_stages.", "decoder_stages."),
    )
    for old, new in replacements:
        if key.startswith(old):
            key = new + key[len(old) :]
            break

    match = re.fullmatch(r"out_0\.conv_out\.(.+)", key)
    if match:
        return f"head.conv.{match.group(1)}"
    match = re.fullmatch(r"out_blocks\.([0-3])\.conv_out\.(.+)", key)
    if match:
        return f"deep_supervision_heads.{match.group(1)}.conv.{match.group(2)}"

    parts = key.split(".")
    parts = [_COMPONENTS.get(part, part) for part in parts]
    return ".".join(parts)
