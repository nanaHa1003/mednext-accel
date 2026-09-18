"""CUDA correctness checks for the depthwise weight-gradient prototype."""
import unittest
import torch
import torch.nn.functional as F

from depthwise_split import (depthwise_input_grad, depthwise_weight_grad,
                             replace_highres_depthwise_convs)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class WeightGradientTests(unittest.TestCase):
    def test_transpose_weight_gradient_against_double_precision(self):
        from depthwise_split import depthwise_transpose_weight_grad
        torch.manual_seed(41)
        x = torch.randn((2, 3, 4, 5, 6), device='cuda', dtype=torch.float64)
        w = torch.randn((3, 1, 3, 3, 3), device='cuda', dtype=torch.float64,
                        requires_grad=True)
        y = F.conv_transpose3d(x, w, stride=2, padding=1, groups=3)
        g = torch.randn_like(y)
        reference, = torch.autograd.grad(y, w, g)
        actual = depthwise_transpose_weight_grad(x, g, splits=4, block=128)
        torch.testing.assert_close(actual, reference, rtol=2e-12, atol=2e-12)

    def test_input_gradient_against_double_precision(self):
        torch.manual_seed(17)
        for kernel_size in (1, 3, 5):
            shape = (2, 3, 5, 6, 7)
            x = torch.randn(shape, device='cuda', dtype=torch.float64,
                            requires_grad=True)
            g = torch.randn_like(x)
            w = torch.randn(3, 1, kernel_size, kernel_size, kernel_size,
                            device='cuda', dtype=torch.float64)
            ref, = torch.autograd.grad(
                F.conv3d(x, w, padding=kernel_size // 2, groups=3), x, g)
            out = depthwise_input_grad(g, w, block=128)
            torch.testing.assert_close(out, ref, rtol=2e-12, atol=2e-12)

    def test_stride_two_input_gradient_against_double_precision(self):
        from depthwise_split import depthwise_stride2_input_grad
        torch.manual_seed(47)
        x = torch.randn((2, 3, 5, 6, 7), device='cuda', dtype=torch.float64,
                        requires_grad=True)
        w = torch.randn((3, 1, 3, 3, 3), device='cuda', dtype=torch.float64)
        y = F.conv3d(x, w, stride=2, padding=1, groups=3)
        g = torch.randn_like(y)
        reference, = torch.autograd.grad(y, x, g)
        actual = depthwise_stride2_input_grad(g, w, x.shape[2:], block=128)
        torch.testing.assert_close(actual, reference, rtol=2e-12, atol=2e-12)

    def test_module_forward_and_all_gradients(self):
        import copy
        for kernel_size in (1, 3, 5, 7):
            torch.manual_seed(5)
            original = torch.nn.Sequential(torch.nn.Conv3d(
                32, 32, kernel_size, padding=kernel_size // 2,
                groups=32)).cuda().to(torch.bfloat16)
            candidate = copy.deepcopy(original)
            parameters = list(candidate.parameters())
            self.assertEqual(replace_highres_depthwise_convs(candidate), 1)
            self.assertTrue(all(a is b for a, b in zip(parameters, candidate.parameters())))
            x = torch.randn(1, 32, 128, 128, 128, device='cuda',
                            dtype=torch.bfloat16, requires_grad=True)
            z = x.detach().clone().requires_grad_()
            y, q = original(x), candidate(z)
            g = torch.randn_like(y)
            a = (y, *torch.autograd.grad(y, (x, *original.parameters()), g))
            b = (q, *torch.autograd.grad(q, (z, *candidate.parameters()), g))
            for u, v in zip(a, b):
                rel = ((u.float() - v.float()).norm() /
                       u.float().norm().clamp_min(1e-12))
                self.assertLess(rel.item(), .01)

    def test_64_cubed_module_forward_and_all_gradients(self):
        import copy
        torch.manual_seed(11)
        original = torch.nn.Sequential(torch.nn.Conv3d(
            64, 64, 3, padding=1, groups=64)).cuda().to(torch.bfloat16)
        candidate = copy.deepcopy(original)
        self.assertEqual(replace_highres_depthwise_convs(candidate), 1)
        x = torch.randn(1, 64, 64, 64, 64, device='cuda',
                        dtype=torch.bfloat16, requires_grad=True)
        z = x.detach().clone().requires_grad_()
        y, q = original(x), candidate(z)
        g = torch.randn_like(y)
        a = (y, *torch.autograd.grad(y, (x, *original.parameters()), g))
        b = (q, *torch.autograd.grad(q, (z, *candidate.parameters()), g))
        for u, v in zip(a, b):
            rel = ((u.float() - v.float()).norm() /
                   u.float().norm().clamp_min(1e-12))
            self.assertLess(rel.item(), .01)

    def test_transpose_module_forward_and_all_gradients(self):
        import copy
        from depthwise_split import SplitDepthwiseConvTranspose3d
        torch.manual_seed(43)
        original = torch.nn.Sequential(torch.nn.ConvTranspose3d(
            64, 64, 3, stride=2, padding=1, groups=64)).cuda().to(torch.bfloat16)
        candidate = torch.nn.Sequential(
            SplitDepthwiseConvTranspose3d(copy.deepcopy(original[0])))
        x = torch.randn(1, 64, 64, 64, 64, device='cuda', dtype=torch.bfloat16,
                        requires_grad=True)
        z = x.detach().clone().requires_grad_()
        y, q = original(x), candidate(z)
        g = torch.randn_like(y)
        a = (y, *torch.autograd.grad(y, (x, *original.parameters()), g))
        b = (q, *torch.autograd.grad(q, (z, *candidate.parameters()), g))
        for u, v in zip(a, b):
            rel = ((u.float() - v.float()).norm() /
                   u.float().norm().clamp_min(1e-12))
            self.assertLess(rel.item(), .01)

    def test_padding_channels_and_batch_against_double_precision(self):
        torch.manual_seed(19)
        for kernel_size in (1, 3, 5, 7):
            for shape in [(1, 3, 5, 7, 9), (2, 2, 4, 3, 6)]:
                x = torch.randn(shape, device='cuda')
                g = torch.randn_like(x)
                w = torch.randn(shape[1], 1, kernel_size, kernel_size, kernel_size,
                                device='cuda', dtype=torch.float64, requires_grad=True)
                ref, = torch.autograd.grad(
                    F.conv3d(x.double(), w, padding=kernel_size // 2,
                             groups=shape[1]), w, g.double())
                out = depthwise_weight_grad(
                    x, g, 4, 128, kernel_size=kernel_size)
                torch.testing.assert_close(out.double(), ref, rtol=2e-5, atol=2e-5)

    def test_replacement_accepts_cubic_odd_kernels(self):
        for kernel_size in (1, 3, 5, 7):
            model = torch.nn.Sequential(torch.nn.Conv3d(
                32, 32, kernel_size, padding=kernel_size // 2, groups=32))
            self.assertEqual(replace_highres_depthwise_convs(model), 1)

    def test_replacement_accepts_stride_two_transpose_depthwise(self):
        model = torch.nn.Sequential(torch.nn.ConvTranspose3d(
            64, 64, 3, stride=2, padding=1, groups=64))
        self.assertEqual(replace_highres_depthwise_convs(model), 1)

    def test_stride_two_downsample_is_explicitly_opt_in(self):
        model = torch.nn.Sequential(torch.nn.Conv3d(
            32, 32, 3, stride=2, padding=1, groups=32))
        self.assertEqual(replace_highres_depthwise_convs(model), 0)
        self.assertEqual(replace_highres_depthwise_convs(
            torch.nn.Sequential(torch.nn.Conv3d(
                32, 32, 3, stride=2, padding=1, groups=32)),
            include_stride2_input_grad=True), 1)

    def test_high_resolution_mixed_precision(self):
        torch.manual_seed(31)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.randn(1, 32, 128, 128, 128, device='cuda', dtype=dtype)
            g = torch.randn_like(x)
            w = torch.randn(32, 1, 3, 3, 3, device='cuda', dtype=dtype, requires_grad=True)
            ref, = torch.autograd.grad(F.conv3d(x, w, padding=1, groups=32), w, g)
            out = depthwise_weight_grad(x, g, 64, 512)
            rel = (out.float() - ref.float()).norm() / ref.float().norm()
            self.assertTrue(torch.isfinite(out).all())
            self.assertLess(rel.item(), 1e-4 if dtype == torch.float32 else .01)


if __name__ == '__main__':
    unittest.main()
