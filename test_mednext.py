"""Behavior checks for MedNeXt model options."""
import copy
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import mednext


class ExpandedBranchCheckpointBlockTests(unittest.TestCase):
    def _assert_block_parity(self, block_type, shape, out_channels):
        torch.manual_seed(31)
        kwargs = dict(
            spatial_dims=3,
            in_channels=shape[1],
            out_channels=out_channels,
            expand_ratio=2,
            kernel_size=3,
            res_block=True,
        )
        reference = block_type(**kwargs, checkpoint_expanded=False).double().train()
        candidate = block_type(**kwargs, checkpoint_expanded=True).double().train()
        candidate.load_state_dict(copy.deepcopy(reference.state_dict()), strict=True)

        reference_input = torch.randn(shape, dtype=torch.float64, requires_grad=True)
        candidate_input = reference_input.detach().clone().requires_grad_(True)
        reference_output = reference(reference_input)
        candidate_output = candidate(candidate_input)
        output_gradient = torch.randn_like(reference_output)
        reference_output.backward(output_gradient)
        candidate_output.backward(output_gradient)

        torch.testing.assert_close(reference_output, candidate_output, rtol=0, atol=0)
        torch.testing.assert_close(
            reference_input.grad, candidate_input.grad, rtol=0, atol=0)
        reference_parameters = dict(reference.named_parameters())
        candidate_parameters = dict(candidate.named_parameters())
        self.assertEqual(reference_parameters.keys(), candidate_parameters.keys())
        for name in reference_parameters:
            with self.subTest(parameter=name):
                torch.testing.assert_close(
                    reference_parameters[name].grad,
                    candidate_parameters[name].grad,
                    rtol=0,
                    atol=0,
                )

    def test_output_and_gradients_match_for_all_block_geometries(self):
        cases = (
            (mednext.MedNeXtBlock, (1, 2, 5, 6, 7), 2),
            (mednext.MedNeXtDownBlock, (1, 2, 6, 8, 10), 4),
            (mednext.MedNeXtUpBlock, (1, 4, 3, 4, 5), 2),
        )
        for block_type, shape, out_channels in cases:
            with self.subTest(block=block_type.__name__):
                self._assert_block_parity(block_type, shape, out_channels)

    def test_checkpoint_runs_only_during_gradient_enabled_training(self):
        block = mednext.MedNeXtBlock(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            expand_ratio=2,
            kernel_size=3,
            checkpoint_expanded=True,
        )
        calls = []

        def checkpoint(function, *args, use_reentrant):
            self.assertFalse(use_reentrant)
            calls.append(function)
            return function(*args)

        x = torch.randn(1, 2, 4, 4, 4)
        with mock.patch("mednext.grad_ckpt", side_effect=checkpoint):
            block.train()
            block(x)
            self.assertEqual(len(calls), 1)

            block.eval()
            block(x)
            self.assertEqual(len(calls), 1)

            block.train()
            with torch.no_grad():
                block(x)
            self.assertEqual(len(calls), 1)

    def test_disabled_policy_does_not_call_checkpoint(self):
        block = mednext.MedNeXtBlock(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            expand_ratio=2,
            kernel_size=3,
            checkpoint_expanded=False,
        ).train()
        with mock.patch("mednext.grad_ckpt") as checkpoint:
            block(torch.randn(1, 2, 4, 4, 4))
        checkpoint.assert_not_called()


class MedNeXtCheckpointPolicyTests(unittest.TestCase):
    @staticmethod
    def _model(**options):
        return mednext.MedNeXt(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            filters=2,
            num_blocks=[1, 1, 1],
            expand_ratio=[2, 2, 2],
            **options,
        )

    def test_policy_resolution_and_block_propagation(self):
        default = self._model()
        explicit_none = self._model(checkpoint_style="none")
        legacy_block = self._model(use_grad_checkpoint=True)
        explicit_block = self._model(checkpoint_style="block")
        expanded = self._model(checkpoint_style="expanded")

        self.assertEqual(default.checkpoint_style, "none")
        self.assertEqual(explicit_none.checkpoint_style, "none")
        self.assertEqual(legacy_block.checkpoint_style, "block")
        self.assertTrue(legacy_block.use_grad_checkpoint)
        self.assertEqual(explicit_block.checkpoint_style, "block")
        self.assertTrue(explicit_block.use_grad_checkpoint)
        self.assertEqual(expanded.checkpoint_style, "expanded")
        self.assertFalse(expanded.use_grad_checkpoint)

        expanded_blocks = [
            module for module in expanded.modules()
            if isinstance(module, mednext.MedNeXtBlock)
        ]
        block_policy_blocks = [
            module for module in explicit_block.modules()
            if isinstance(module, mednext.MedNeXtBlock)
        ]
        self.assertTrue(expanded_blocks)
        self.assertTrue(all(module.checkpoint_expanded for module in expanded_blocks))
        self.assertTrue(all(
            not module.checkpoint_expanded for module in block_policy_blocks))

    def test_rejects_conflicting_legacy_and_explicit_options(self):
        with self.assertRaisesRegex(ValueError, "use_grad_checkpoint"):
            self._model(use_grad_checkpoint=True, checkpoint_style="expanded")

    def test_all_factories_propagate_expanded_policy(self):
        for factory_name in (
            "mednext_small", "mednext_base", "mednext_medium", "mednext_large"
        ):
            with self.subTest(factory=factory_name):
                model = getattr(mednext, factory_name)(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=3,
                    filters=1,
                    checkpoint_style="expanded",
                )
                blocks = [
                    module for module in model.modules()
                    if isinstance(module, mednext.MedNeXtBlock)
                ]
                self.assertTrue(blocks)
                self.assertTrue(all(module.checkpoint_expanded for module in blocks))

    def test_state_dict_and_eval_output_are_policy_independent(self):
        torch.manual_seed(41)
        x = torch.randn(1, 1, 16, 16, 16)
        for deep_supervision in (False, True):
            with self.subTest(deep_supervision=deep_supervision):
                reference = self._model(
                    deep_supervision=deep_supervision,
                    checkpoint_style="none",
                ).eval()
                for style in ("expanded", "block"):
                    candidate = self._model(
                        deep_supervision=deep_supervision,
                        checkpoint_style=style,
                    ).eval()
                    candidate.load_state_dict(
                        copy.deepcopy(reference.state_dict()), strict=True)
                    self.assertEqual(
                        list(reference.state_dict()), list(candidate.state_dict()))
                    with torch.no_grad():
                        torch.testing.assert_close(
                            reference(x), candidate(x), rtol=0, atol=0)

    def test_training_output_and_gradients_match_expanded_policy(self):
        torch.manual_seed(43)
        for deep_supervision in (False, True):
            with self.subTest(deep_supervision=deep_supervision):
                reference = self._model(
                    deep_supervision=deep_supervision,
                    checkpoint_style="none",
                ).double().train()
                candidate = self._model(
                    deep_supervision=deep_supervision,
                    checkpoint_style="expanded",
                ).double().train()
                candidate.load_state_dict(
                    copy.deepcopy(reference.state_dict()), strict=True)

                reference_input = torch.randn(
                    1, 1, 16, 16, 16, dtype=torch.float64, requires_grad=True)
                candidate_input = reference_input.detach().clone().requires_grad_(True)
                reference_output = reference(reference_input)
                candidate_output = candidate(candidate_input)
                reference_output.square().sum().backward()
                candidate_output.square().sum().backward()

                torch.testing.assert_close(
                    reference_output, candidate_output, rtol=0, atol=0)
                torch.testing.assert_close(
                    reference_input.grad, candidate_input.grad, rtol=0, atol=0)
                reference_parameters = dict(reference.named_parameters())
                candidate_parameters = dict(candidate.named_parameters())
                self.assertEqual(reference_parameters.keys(), candidate_parameters.keys())
                for name in reference_parameters:
                    with self.subTest(parameter=name):
                        torch.testing.assert_close(
                            reference_parameters[name].grad,
                            candidate_parameters[name].grad,
                            rtol=0,
                            atol=0,
                        )


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
