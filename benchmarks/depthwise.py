#!/usr/bin/env python3
"""Compare native and custom depthwise forward/backward components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("regular", "transpose", "downsample"), required=True)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--spatial", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from mednext_accel.optimization.autotune import _benchmark_depthwise

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    native_ms, candidate_ms = _benchmark_depthwise(
        args.kind,
        args.channels,
        args.spatial,
        dtype,
        args.warmup,
        args.repetitions,
    )
    result = {
        "kind": args.kind,
        "channels": args.channels,
        "spatial": args.spatial,
        "dtype": str(dtype),
        "native_ms": native_ms,
        "candidate_ms": candidate_ms,
        "speedup": native_ms / candidate_ms,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(args.output)


if __name__ == "__main__":
    main()
