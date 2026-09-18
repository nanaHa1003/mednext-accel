#!/usr/bin/env python3
"""Run MedNeXt-Accel per-shape kernel selection on a CUDA device."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("small", "base", "medium", "large"), default="base")
    parser.add_argument("--shape", type=int, nargs=5, default=(1, 1, 128, 128, 128))
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--include-stride2-dx", action="store_true")
    parser.add_argument("--cache", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    import mednext_accel
    from mednext_accel.optimization import optimize

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    factory = getattr(mednext_accel, f"mednext_{args.variant}")
    model = factory(
        in_channels=args.shape[1],
        out_channels=args.classes,
        base_channels=args.base_channels,
        kernel_size=args.kernel_size,
    ).cuda()
    report = optimize(
        model,
        input_shape=tuple(args.shape),
        dtype=dtype,
        policy="autotune",
        compile_mode=args.compile_mode,
        cache_path=args.cache,
        warmup=args.warmup,
        repetitions=args.repetitions,
        include_stride2_input_grad=args.include_stride2_dx,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(report), indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
