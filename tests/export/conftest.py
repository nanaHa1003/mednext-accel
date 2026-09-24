from __future__ import annotations

import builtins
import importlib.util
import sys

import pytest
import torch

from mednext_accel import MedNeXtV2Config, mednext_v2_base
from mednext_accel.models import mednext_v2 as v2


@pytest.fixture
def reduced_v2_export_model(monkeypatch: pytest.MonkeyPatch):
    """Build the default v2 Base factory with a compact architecture for export tests."""

    config = MedNeXtV2Config(
        variant="base",
        in_channels=1,
        out_channels=3,
        base_channels=2,
        kernel_size=3,
        block_counts=(1,) * 9,
        expansion_ratios=(2,) * 9,
        downsample_expansion_ratios=(2,) * 4,
        upsample_expansion_ratios=(2,) * 4,
    )
    monkeypatch.setattr(v2, "get_mednext_v2_config", lambda *args, **kwargs: config)
    model = mednext_v2_base(in_channels=1, out_channels=3)
    assert model.optimization_source == "auto"
    return model.eval()


@pytest.fixture
def v2_export_example() -> torch.Tensor:
    return torch.randn(1, 1, 32, 32, 32)


@pytest.fixture
def no_v2_grn_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if evaluation tries to import the optional Triton GRN backend."""

    backend = "mednext_accel.ops._triton.grn"
    monkeypatch.setitem(sys.modules, backend, None)
    original_import = builtins.__import__

    def reject_grn_backend_import(name, globals=None, locals=None, fromlist=(), level=0):
        package = globals.get("__package__", "") if globals is not None else ""
        resolved = importlib.util.resolve_name("." * level + name, package) if level else name
        if resolved == backend:
            raise AssertionError("evaluation attempted to import the Triton GRN backend")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_grn_backend_import)
