"""Numerical and checkpoint compatibility checks for the opt-in prototype."""
import copy
import unittest

import torch
from torch import nn

from pointwise_gemm import replace_pointwise_convs


class PointwiseTests(unittest.TestCase):
    def test_outputs_gradients_and_parameter_identity(self):
        for batch in (1, 2):
            for bias in (False, True):
                torch.manual_seed(7)
                original = nn.Sequential(nn.Conv3d(3, 7, 1, bias=bias)).double()
                candidate = copy.deepcopy(original)
                parameters = list(candidate.parameters())
                self.assertEqual(replace_pointwise_convs(candidate), 1)
                self.assertTrue(all(a is b for a, b in zip(parameters, candidate.parameters())))
                candidate.load_state_dict(original.state_dict(), strict=True)
                x = torch.randn(batch, 3, 3, 4, 5, dtype=torch.float64, requires_grad=True)
                z = x.detach().clone().requires_grad_()
                y, q = original(x), candidate(z)
                torch.testing.assert_close(y, q, rtol=1e-12, atol=1e-12)
                g = torch.randn_like(y)
                a = torch.autograd.grad(y, (x, *original.parameters()), g)
                b = torch.autograd.grad(q, (z, *candidate.parameters()), g)
                for u, v in zip(a, b):
                    torch.testing.assert_close(u, v, rtol=1e-12, atol=1e-12)

    def test_ineligible_layers_are_preserved(self):
        layers = nn.Sequential(nn.Conv3d(4, 4, 3), nn.Conv3d(4, 4, 1, stride=2),
                               nn.Conv3d(4, 4, 1, groups=4), nn.ConvTranspose3d(4, 4, 1),
                               nn.Conv3d(4, 4, 1, padding=1))
        before = list(layers)
        self.assertEqual(replace_pointwise_convs(layers), 0)
        self.assertEqual(list(layers), before)

    def test_mednext_deep_supervision_outputs_and_gradients(self):
        import mednext
        torch.manual_seed(42)
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(2)
        original = mednext.mednext_base(
            spatial_dims=3, in_channels=1, out_channels=3, filters=2,
            kernel_size=3, deep_supervision=True, use_grad_checkpoint=False)
        candidate = copy.deepcopy(original)
        self.assertEqual(replace_pointwise_convs(candidate), 53)
        self.assertEqual(list(original.state_dict()), list(candidate.state_dict()))
        x = torch.randn(1, 1, 32, 32, 32)
        y, z = original(x), candidate(x)
        torch.testing.assert_close(y, z, rtol=2e-4, atol=2e-5)
        y.square().mean().backward()
        z.square().mean().backward()
        for p, q in zip(original.parameters(), candidate.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-3, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
