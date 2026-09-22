"""Command-line entry point for local profile generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .api import profile
from .progress import create_progress_reporter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mednext-accel")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "profile", help="benchmark this GPU and write policy YAML plus evidence JSON"
    )
    command.add_argument("workload", nargs="?", help="optional YAML campaign")
    command.add_argument("--preset", choices=("mednext-v1", "all"))
    command.add_argument(
        "--progress",
        choices=("auto", "plain", "quiet"),
        default="auto",
        help="progress display (default: auto; plain when output is redirected)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.workload and arguments.preset:
        _parser().error("workload and --preset are mutually exclusive")
    source = arguments.workload
    if arguments.preset:
        source = {"preset": arguments.preset}
    reporter = create_progress_reporter(arguments.progress)
    try:
        result = profile(source, progress=reporter)
    finally:
        reporter.close()
    print(f"Optimization policy written to {result.artifacts.policy_path}")
    print(f"Profiling evidence written to {result.artifacts.evidence_path}")
    print(f"Recorded {len(result.evidence.kernel_measurements)} kernel observations")
    publication = result.evidence.publication
    if publication is not None:
        print(f"Model policy: {publication.reason}")
        if not publication.matches_tested_policy:
            print(
                "Published negative-only overlay; "
                "bundled fallthrough was not validated by this comparison."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
