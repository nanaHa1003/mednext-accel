"""Compatibility architectures for third-party implementations."""

from .monai import (
    monai_mednext_base,
    monai_mednext_large,
    monai_mednext_medium,
    monai_mednext_small,
)

__all__ = [
    "monai_mednext_base",
    "monai_mednext_large",
    "monai_mednext_medium",
    "monai_mednext_small",
]
