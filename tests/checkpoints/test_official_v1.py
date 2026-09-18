from __future__ import annotations

from collections import OrderedDict

import torch

from mednext_accel import mednext_small
from mednext_accel.checkpoints import convert_state_dict, load_checkpoint

_BLOCK_PARTS = {
    "depthwise": "conv1",
    "expand": "conv2",
    "project": "conv3",
    "residual": "res_conv",
}


def _native_to_official(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        parts = key.split(".")
        if parts[0] == "encoder_stages":
            prefix, rest = f"enc_block_{parts[1]}", parts[2:]
        elif parts[0] == "downsamples":
            prefix, rest = f"down_{parts[1]}", parts[2:]
        elif parts[0] == "upsamples":
            prefix, rest = f"up_{3 - int(parts[1])}", parts[2:]
        elif parts[0] == "decoder_stages":
            prefix, rest = f"dec_block_{3 - int(parts[1])}", parts[2:]
        elif parts[0] == "head":
            prefix, rest = "out_0", ["conv_out", *parts[2:]]
        elif parts[0] == "deep_supervision_heads":
            prefix, rest = f"out_{4 - int(parts[1])}", ["conv_out", *parts[3:]]
        else:
            prefix, rest = parts[0], parts[1:]
        if rest and rest[0] in _BLOCK_PARTS:
            rest[0] = _BLOCK_PARTS[rest[0]]
        converted[f"module.{prefix}.{'.'.join(rest)}"] = value.clone()
    converted["module.dummy_tensor"] = torch.ones(1)
    return converted


def test_official_conversion_maps_all_keys_and_ignores_dummy_tensor() -> None:
    model = mednext_small(
        in_channels=1,
        out_channels=3,
        base_channels=2,
        deep_supervision=True,
    )
    official = _native_to_official(model.state_dict())

    converted = convert_state_dict(official, source="official_v1", target_config=model.config)

    assert converted.keys() == model.state_dict().keys()
    for key, expected in model.state_dict().items():
        torch.testing.assert_close(converted[key], expected)


def test_official_source_accepts_hyphenated_public_spelling() -> None:
    model = mednext_small(in_channels=1, out_channels=3, base_channels=2)
    official = _native_to_official(model.state_dict())

    converted = convert_state_dict(official, source="official-v1", target_config=model.config)

    assert converted.keys() == model.state_dict().keys()


def test_wrapped_official_checkpoint_loads_with_exact_output() -> None:
    torch.manual_seed(12)
    source = mednext_small(in_channels=1, out_channels=3, base_channels=2).eval()
    target = mednext_small(in_channels=1, out_channels=3, base_channels=2).eval()
    checkpoint = {"state_dict": _native_to_official(source.state_dict()), "epoch": 42}

    report = load_checkpoint(target, checkpoint, source="auto")
    sample = torch.randn(1, 1, 32, 32, 32)

    assert report.source == "official_v1"
    assert report.ignored_keys == ("dummy_tensor",)
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    with torch.no_grad():
        torch.testing.assert_close(target(sample), source(sample), rtol=0, atol=0)


def test_torch_compile_wrapper_checkpoint_loads_into_eager_model() -> None:
    source = mednext_small(in_channels=1, out_channels=3, base_channels=2).eval()
    compiled = torch.compile(source)
    target = mednext_small(in_channels=1, out_channels=3, base_channels=2).eval()

    report = load_checkpoint(target, compiled.state_dict(), source="auto")

    assert report.source == "native"
    for actual, expected in zip(target.parameters(), source.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_eager_checkpoint_loads_into_torch_compile_wrapper() -> None:
    source = mednext_small(in_channels=1, out_channels=3, base_channels=2).eval()
    target = mednext_small(in_channels=1, out_channels=3, base_channels=2).eval()
    compiled_target = torch.compile(target)

    report = load_checkpoint(compiled_target, source.state_dict(), source="native")

    assert report.source == "native"
    for actual, expected in zip(target.parameters(), source.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
