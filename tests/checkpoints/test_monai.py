from __future__ import annotations

import inspect

import pytest
import torch

from mednext_accel import mednext_base
from mednext_accel.checkpoints import load_checkpoint
from mednext_accel.compat.monai import monai_mednext_base, monai_mednext_small

MonaiMedNeXt = pytest.importorskip("monai.networks.nets").MedNeXt


def _monai_small(*, deep_supervision: bool = False) -> MonaiMedNeXt:
    return MonaiMedNeXt(
        spatial_dims=3,
        init_filters=2,
        in_channels=1,
        out_channels=3,
        encoder_expansion_ratio=2,
        decoder_expansion_ratio=2,
        bottleneck_expansion_ratio=2,
        kernel_size=3,
        deep_supervision=deep_supervision,
        use_residual_connection=True,
        blocks_down=(2, 2, 2, 2),
        blocks_bottleneck=2,
        blocks_up=(2, 2, 2, 2),
    )


def test_monai_small_loads_losslessly_and_has_exact_eval_output() -> None:
    torch.manual_seed(21)
    source = _monai_small().eval()
    target = monai_mednext_small(
        in_channels=1,
        out_channels=3,
        base_channels=2,
    ).eval()

    report = load_checkpoint(target, source.state_dict(), source="monai")
    sample = torch.randn(1, 1, 32, 32, 32)

    assert target.config.compatibility == "monai"
    assert report.missing_keys == ()
    with torch.no_grad():
        torch.testing.assert_close(target(sample), source(sample), rtol=0, atol=0)


def test_monai_deep_supervision_heads_map_by_resolution() -> None:
    source = _monai_small(deep_supervision=True)
    target = monai_mednext_small(
        in_channels=1,
        out_channels=3,
        base_channels=2,
        deep_supervision=True,
    )

    load_checkpoint(target, source.state_dict(), source="monai")

    for monai_index, native_index in enumerate((0, 1, 2, 3)):
        torch.testing.assert_close(
            target.deep_supervision_heads[native_index].conv.weight,
            source.out_blocks[monai_index].conv_out.weight,
        )


def test_monai_base_checkpoint_is_rejected_by_official_base_architecture() -> None:
    monai_compatible = monai_mednext_base(
        in_channels=1,
        out_channels=3,
        base_channels=2,
    )
    official = mednext_base(in_channels=1, out_channels=3, base_channels=2)

    with pytest.raises(ValueError, match=r"down_0.*down_1"):
        load_checkpoint(official, monai_compatible.state_dict(), source="monai")


def test_compatibility_factory_has_discoverable_keyword_signature() -> None:
    signature = inspect.signature(monai_mednext_base)

    assert list(signature.parameters)[:2] == ["in_channels", "out_channels"]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
