"""Tests for per-shape policy application without requiring CUDA benchmarking."""
import copy
import unittest

import torch
from torch import nn

from kernel_autotune import apply_kernel_policy
from pointwise_gemm import GemmPointwise3d


class KernelPolicyTests(unittest.TestCase):
    def test_policy_preserves_checkpoint_and_native_fallback(self):
        original = nn.Sequential(nn.Conv3d(3, 5, 1), nn.ReLU())
        candidate = copy.deepcopy(original)
        report = {'policy': {'pointwise_gemm': [], 'depthwise_regular': [],
                             'depthwise_transpose': [], 'depthwise_downsample': []}}
        apply_kernel_policy(candidate, report)
        self.assertIsInstance(candidate[0], GemmPointwise3d)
        self.assertEqual(list(original.state_dict()), list(candidate.state_dict()))
        x = torch.randn(1, 3, 4, 5, 6)
        torch.testing.assert_close(original(x), candidate(x))

    def test_selected_pointwise_shape_uses_gemm_wrapper(self):
        model = nn.Sequential(nn.Conv3d(3, 5, 1))
        report = {'policy': {'pointwise_gemm': [[3, 5, 4, 5, 6]],
                             'depthwise_regular': [], 'depthwise_transpose': [],
                             'depthwise_downsample': []}}
        apply_kernel_policy(model, report)
        self.assertIsInstance(model[0], GemmPointwise3d)
        self.assertEqual(model[0].selected_shapes, frozenset({(3, 5, 4, 5, 6)}))


if __name__ == '__main__':
    unittest.main()
