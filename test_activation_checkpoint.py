"""Tests for activation-checkpoint policy resolution."""
import unittest

from activation_checkpoint import (
    CHECKPOINT_STYLES,
    resolve_checkpoint_levels,
    resolve_checkpoint_style,
)


class CheckpointPolicyTests(unittest.TestCase):
    def test_levels_are_optional_and_canonicalized(self):
        self.assertIsNone(
            resolve_checkpoint_levels("expanded", None, depth=4))
        self.assertEqual(
            resolve_checkpoint_levels("expanded", (2, 0, 1), depth=4),
            (0, 1, 2),
        )

    def test_rejects_levels_for_other_checkpoint_styles(self):
        for style in ("none", "block"):
            with self.subTest(style=style), self.assertRaisesRegex(
                ValueError, "only valid with checkpoint_style='expanded'"
            ):
                resolve_checkpoint_levels(style, (0,), depth=4)

    def test_rejects_invalid_level_tuples(self):
        invalid = ((), (0, 0), (-1,), (5,), (True,), (1.0,), [0, 1])
        for levels in invalid:
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                resolve_checkpoint_levels("expanded", levels, depth=4)

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
