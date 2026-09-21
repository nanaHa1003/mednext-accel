"""Adaptive execution backends that preserve native model parameters."""

from .adaptive import AdaptiveDepthwise3d, AdaptivePointwise3d, install_adaptive_operators

__all__ = ["AdaptiveDepthwise3d", "AdaptivePointwise3d", "install_adaptive_operators"]
