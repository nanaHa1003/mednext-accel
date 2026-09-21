"""Optional execution backends that preserve model parameters."""

from .depthwise import replace_depthwise_convs
from .pointwise import GemmPointwise3d, replace_pointwise_convs

__all__ = ["replace_depthwise_convs", "GemmPointwise3d", "replace_pointwise_convs"]
from .adaptive import AdaptiveDepthwise3d, AdaptivePointwise3d, install_adaptive_operators

__all__ = ["AdaptiveDepthwise3d", "AdaptivePointwise3d", "install_adaptive_operators"]
