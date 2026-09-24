"""Private Triton kernels loaded only when acceleration is requested.

The ``grn`` module provides fused Global Response Normalization forward and
backward operators. Keep imports lazy so reference execution needs no Triton.
"""
