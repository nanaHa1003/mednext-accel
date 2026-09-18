"""Key translation for the original MedNeXt v1 implementation."""

from __future__ import annotations

import re

_COMPONENTS = {
    "conv1": "depthwise",
    "conv2": "expand",
    "conv3": "project",
    "res_conv": "residual",
}


def convert_key(key: str) -> str | None:
    """Convert one official state-dict key to the native schema."""

    if key == "dummy_tensor":
        return None

    match = re.match(r"enc_block_([0-3])\.(.+)", key)
    if match:
        key = f"encoder_stages.{match.group(1)}.{match.group(2)}"
    else:
        match = re.match(r"down_([0-3])\.(.+)", key)
        if match:
            key = f"downsamples.{match.group(1)}.{match.group(2)}"
        else:
            match = re.match(r"up_([0-3])\.(.+)", key)
            if match:
                key = f"upsamples.{3 - int(match.group(1))}.{match.group(2)}"
            else:
                match = re.match(r"dec_block_([0-3])\.(.+)", key)
                if match:
                    key = f"decoder_stages.{3 - int(match.group(1))}.{match.group(2)}"
                else:
                    match = re.match(r"out_([0-4])\.conv_out\.(.+)", key)
                    if match:
                        level = int(match.group(1))
                        key = (
                            f"head.conv.{match.group(2)}"
                            if level == 0
                            else f"deep_supervision_heads.{4 - level}.conv.{match.group(2)}"
                        )

    parts = key.split(".")
    parts = [_COMPONENTS.get(part, part) for part in parts]
    return ".".join(parts)
