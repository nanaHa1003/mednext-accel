"""Command-line entry point for local profile generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .api import profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mednext-accel")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("profile", help="benchmark this GPU and write one profile")
    command.add_argument("workload", nargs="?", help="optional YAML campaign")
    command.add_argument("--preset", choices=("mednext-v1", "all"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.workload and arguments.preset:
        _parser().error("workload and --preset are mutually exclusive")
    source = arguments.workload
    if arguments.preset:
        source = {"preset": arguments.preset}
    result = profile(source)
    print(f"Optimization profile written to {result.output_path}")
    print(f"Recorded {len(result.measurements)} validated measurements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
