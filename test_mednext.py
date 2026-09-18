"""Behavior checks for MedNeXt model options."""
import copy
import unittest

import torch
import torch.nn.functional as F

import mednext


class EvalApproximateGELUTests(unittest.TestCase):
    def test_exact_during_training_and_approximate_during_eval(self):
        activation = mednext.EvalModeGELU(approximate_eval=True)
        x = torch.linspace(-4, 4, 1001)

        activation.train()
        torch.testing.assert_close(
            activation(x), F.gelu(x, approximate='none'), rtol=0, atol=0)

        activation.eval()
        torch.testing.assert_close(
            activation(x), F.gelu(x, approximate='tanh'), rtol=0, atol=0)

    def test_factory_option_preserves_training_and_checkpoint_keys(self):
        torch.manual_seed(13)
        exact = mednext.mednext_base(
            spatial_dims=3, in_channels=1, out_channels=3, filters=2)
        candidate = mednext.mednext_base(
            spatial_dims=3, in_channels=1, out_channels=3, filters=2,
            approximate_gelu_eval=True)
        candidate.load_state_dict(copy.deepcopy(exact.state_dict()), strict=True)
        self.assertEqual(list(exact.state_dict()), list(candidate.state_dict()))

        x = torch.randn(1, 1, 32, 32, 32)
        exact.train()
        candidate.train()
        torch.testing.assert_close(exact(x), candidate(x), rtol=0, atol=0)

        candidate.eval()
        activations = [m for m in candidate.modules()
                       if isinstance(m, mednext.EvalModeGELU)]
        self.assertTrue(activations)
        self.assertTrue(all(m.approximate_eval for m in activations))


if __name__ == '__main__':
    unittest.main()
