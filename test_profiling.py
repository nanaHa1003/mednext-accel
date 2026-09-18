"""CLI compatibility tests for checkpoint-policy profiling."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import summarize_profiles


ROOT = Path(__file__).resolve().parent


class ProfilingCheckpointPolicyTests(unittest.TestCase):
    def test_summary_policy_falls_back_for_legacy_results(self):
        self.assertEqual(
            summarize_profiles.checkpoint_style_from_settings(
                {"checkpoint": True}),
            "block",
        )
        self.assertEqual(
            summarize_profiles.checkpoint_style_from_settings(
                {"checkpoint": False}),
            "none",
        )
        self.assertEqual(
            summarize_profiles.checkpoint_style_from_settings(
                {"checkpoint": False, "effective_checkpoint_style": "expanded"}),
            "expanded",
        )

    def test_profiler_accepts_and_records_expanded_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            command = [
                sys.executable,
                str(ROOT / "profile_mednext.py"),
                "--variant", "small",
                "--shape", "1", "1", "32", "32",
                "--classes", "3",
                "--filters", "2",
                "--precision", "fp32",
                "--no-deep-supervision",
                "--checkpoint-style", "expanded",
                "--device", "cpu",
                "--warmup", "1",
                "--steps", "1",
                "--profile-steps", "0",
                "--output", str(output),
            ]
            result = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            settings = json.loads((output / "summary.json").read_text())["settings"]
            self.assertEqual(settings["effective_checkpoint_style"], "expanded")
            self.assertFalse(settings["checkpoint"])

    def test_profiler_preserves_legacy_checkpoint_flags(self):
        for flag, expected_style in (
            ("--no-checkpoint", "none"),
            ("--checkpoint", "block"),
        ):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                command = [
                    sys.executable,
                    str(ROOT / "profile_mednext.py"),
                    "--variant", "small",
                    "--shape", "1", "1", "32", "32",
                    "--classes", "3",
                    "--filters", "2",
                    "--precision", "fp32",
                    "--no-deep-supervision",
                    flag,
                    "--device", "cpu",
                    "--warmup", "1",
                    "--steps", "1",
                    "--profile-steps", "0",
                    "--output", str(output),
                ]
                result = subprocess.run(
                    command, cwd=ROOT, text=True, capture_output=True, timeout=120)
                self.assertEqual(
                    result.returncode, 0, msg=result.stdout + result.stderr)
                settings = json.loads(
                    (output / "summary.json").read_text())["settings"]
                self.assertEqual(
                    settings["effective_checkpoint_style"], expected_style)
                self.assertEqual(settings["checkpoint"], expected_style == "block")

    def test_matrix_dry_run_emits_all_compiled_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [
                sys.executable,
                str(ROOT / "run_profile_matrix.py"),
                "--output", directory,
                "--variant", "base",
                "--shape", "1", "1", "128", "128", "128",
                "--classes", "3",
                "--precisions", "bf16",
                "--cases", "compile", "compile_expanded", "compile_ckpt",
                "--repeats", "1",
                "--dry-run",
            ]
            result = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("--checkpoint-style expanded", result.stdout)
            self.assertIn("--no-checkpoint", result.stdout)
            self.assertIn("--checkpoint", result.stdout)


if __name__ == "__main__":
    unittest.main()
