from __future__ import annotations

import torch

from mednext_accel import mednext_small


def test_strict_torch_export_matches_eager() -> None:
    torch.manual_seed(2)
    model = mednext_small(
        in_channels=1,
        out_channels=3,
        base_channels=2,
        deep_supervision=True,
    ).eval()
    example = torch.randn(1, 1, 32, 32, 32)

    exported = torch.export.export(model, (example,), strict=True)

    with torch.no_grad():
        torch.testing.assert_close(exported.module()(example), model(example), rtol=0, atol=0)
    assert "mednext_accel::" not in str(exported.graph_module.graph)
