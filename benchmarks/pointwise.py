#!/usr/bin/env python3
"""Compare native Conv3d and GEMM for one pointwise shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-channels", type=int, required=True)
    parser.add_argument("--out-channels", type=int, required=True)
    parser.add_argument("--spatial", type=int, nargs=3, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from mednext_accel.profiling.runner import _invoke

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    result = _invoke({
        "kind": "pointwise", "batch": args.batch_size,
        "in_channels": args.in_channels, "out_channels": args.out_channels,
        "spatial_shape": args.spatial,
    })
    if result.get("status") != "ok":
        raise SystemExit(result.get("message", result["status"]))
    native_ms = float(result["reference_ms"])
    candidate_ms = float(result["candidate_ms"])
    result = {
        "batch_size": args.batch_size,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "spatial": args.spatial,
        "dtype": args.dtype,
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
