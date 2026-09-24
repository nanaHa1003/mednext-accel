"""Production-oriented MedNeXt model architectures for PyTorch."""

from .checkpointing import CheckpointConfig
from .models import (
    MedNeXtV1,
    MedNeXtV1Config,
    OptimizationSource,
    get_mednext_v1_config,
    mednext_base,
    mednext_large,
    mednext_medium,
    mednext_small,
)
from .optimization import OptimizationPolicy, OptimizationReport, PolicyDecision, load_policy

__all__ = [
    "__version__",
    "CheckpointConfig",
    "MedNeXtV1",
    "MedNeXtV1Config",
    "OptimizationSource",
    "OptimizationPolicy",
    "PolicyDecision",
    "OptimizationReport",
    "get_mednext_v1_config",
    "mednext_base",
    "mednext_large",
    "mednext_medium",
    "mednext_small",
    "load_policy",
]

try:
    from ._version import __version__
except ModuleNotFoundError:
    __version__ = "0+unknown"
