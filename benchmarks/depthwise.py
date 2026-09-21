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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--phase", choices=("backward_input", "backward_weight"))
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
    phase = args.phase or ("backward_weight" if args.kind == "transpose" else "backward_input")
    result = _invoke(
        {
            "kind": "depthwise",
            "batch": args.batch_size,
            "in_channels": args.channels,
            "kernel_size": 3,
            "spatial_shape": [args.spatial] * 3,
            "direction": args.kind,
            "phase": phase,
        }
    )
    if result.get("status") != "ok":
        raise SystemExit(result.get("message", result["status"]))
    native_ms = float(result["reference_ms"])
    candidate_ms = float(result["candidate_ms"])
    result = {
        "kind": args.kind,
        "batch_size": args.batch_size,
        "channels": args.channels,
        "spatial": args.spatial,
        "dtype": args.dtype,
        "phase": phase,
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
