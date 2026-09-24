"""Adaptive execution backends that preserve native model parameters."""

from .adaptive import AdaptiveDepthwise3d, AdaptivePointwise3d, install_adaptive_operators
from .grn import (
    AdaptiveGlobalResponseNorm3d,
    GlobalResponseNorm3d,
    global_response_norm3d_reference,
    install_adaptive_grn,
)

__all__ = [
    "AdaptiveDepthwise3d",
    "AdaptivePointwise3d",
    "AdaptiveGlobalResponseNorm3d",
    "GlobalResponseNorm3d",
    "global_response_norm3d_reference",
    "install_adaptive_grn",
    "install_adaptive_operators",
]
