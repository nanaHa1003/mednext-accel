"""Production-oriented MedNeXt model architectures for PyTorch."""

from .checkpointing import CheckpointConfig
from .models import (
    GlobalResponseNorm3d,
    MedNeXtV1,
    MedNeXtV1Config,
    MedNeXtV2,
    MedNeXtV2Config,
    OptimizationSource,
    get_mednext_v1_config,
    get_mednext_v2_config,
    mednext_base,
    mednext_large,
    mednext_medium,
    mednext_small,
    mednext_v2_base,
    mednext_v2_wide,
)
from .optimization import OptimizationPolicy, OptimizationReport, PolicyDecision, load_policy

__all__ = [
    "__version__",
    "CheckpointConfig",
    "MedNeXtV1",
    "MedNeXtV1Config",
    "MedNeXtV2",
    "MedNeXtV2Config",
    "GlobalResponseNorm3d",
    "OptimizationSource",
    "OptimizationPolicy",
    "PolicyDecision",
    "OptimizationReport",
    "get_mednext_v1_config",
    "get_mednext_v2_config",
    "mednext_base",
    "mednext_large",
    "mednext_medium",
    "mednext_small",
    "mednext_v2_base",
    "mednext_v2_wide",
    "load_policy",
]

try:
    from ._version import __version__
except ModuleNotFoundError:
    __version__ = "0+unknown"
