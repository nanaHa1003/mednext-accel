#!/usr/bin/env python3
"""Run isolated MedNeXt training profiles across optimization and compile modes."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("small", "base", "medium", "large"), default="base")
    parser.add_argument("--shape", type=int, nargs=5, default=(1, 1, 128, 128, 128))
    parser.add_argument("--classes", type=int, nargs="+", default=(3, 8))
    parser.add_argument("--dtypes", choices=("fp32", "bf16", "fp16"), nargs="+", default=("bf16",))
    parser.add_argument(
        "--optimizations",
        choices=("auto", "reference"),
        nargs="+",
        default=("reference", "auto"),
    )
    parser.add_argument(
        "--compile-modes",
        choices=(
            "none",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        nargs="+",
        default=("default",),
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--checkpoint-stages", type=int, nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("repeats must be positive")
    worker = Path(__file__).with_name("profile_train_step.py")
    runs = []
    combinations = itertools.product(
        args.classes,
        args.dtypes,
        args.optimizations,
        args.compile_modes,
        range(1, args.repeats + 1),
    )
    for classes, dtype, optimization, compile_mode, repeat in combinations:
        name = f"c{classes}_{dtype}_{optimization}_{compile_mode}_r{repeat}"
        output = args.output / name
        profile_steps = args.profile_steps if repeat == 1 else 0
        command = [
            sys.executable,
            str(worker),
            "--output",
            str(output),
            "--variant",
            args.variant,
            "--shape",
            *(str(value) for value in args.shape),
            "--classes",
            str(classes),
            "--base-channels",
            str(args.base_channels),
            "--kernel-size",
            str(args.kernel_size),
            "--dtype",
            dtype,
            "--optimization",
            optimization,
            "--compile-mode",
            compile_mode,
            "--warmup",
            str(args.warmup),
            "--steps",
            str(args.steps),
            "--profile-steps",
            str(profile_steps),
        ]
        if args.deep_supervision:
            command.append("--deep-supervision")
        if args.checkpoint_stages is not None:
            command.append("--checkpoint-stages")
            command.extend(str(stage) for stage in args.checkpoint_stages)
        record = {"name": name, "command": command, "status": "planned"}
        runs.append(record)
        print(" ".join(command))
        if args.dry_run:
            continue
        args.output.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, text=True)
        record["status"] = "passed" if completed.returncode == 0 else "failed"
        record["returncode"] = completed.returncode
        (args.output / "manifest.json").write_text(
            json.dumps(
                {
                    "configuration": vars(args) | {"output": str(args.output)},
                    "runs": runs,
                },
                indent=2,
            )
            + "\n"
        )
        if completed.returncode and not args.continue_on_error:
            raise SystemExit(completed.returncode)
    if args.dry_run:
        return
    print(args.output / "manifest.json")


if __name__ == "__main__":
    main()
