"""Tests for activation-checkpoint policy resolution."""
import unittest

from activation_checkpoint import CHECKPOINT_STYLES, resolve_checkpoint_style


class CheckpointPolicyTests(unittest.TestCase):
    def test_legacy_flag_maps_to_existing_behavior(self):
        self.assertEqual(resolve_checkpoint_style(False, None), "none")
        self.assertEqual(resolve_checkpoint_style(True, None), "block")

    def test_explicit_policy_is_returned(self):
        for style in CHECKPOINT_STYLES:
            with self.subTest(style=style):
                self.assertEqual(resolve_checkpoint_style(False, style), style)

    def test_rejects_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_style"):
            resolve_checkpoint_style(False, "stage")

    def test_rejects_legacy_true_with_explicit_policy(self):
        with self.assertRaisesRegex(ValueError, "use_grad_checkpoint"):
            resolve_checkpoint_style(True, "expanded")


if __name__ == "__main__":
    unittest.main()
