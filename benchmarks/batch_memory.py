#!/usr/bin/env python3
"""Measure MedNeXt training memory across batch sizes and checkpoint policies."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

CHECKPOINTS = ("none", "stage-0", "stage-0-1", "all-expansion", "whole-block")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--checkpoints", nargs="+", choices=CHECKPOINTS, default=CHECKPOINTS)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument(
        "--optimization", choices=("auto", "reference"), default="auto"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, help=argparse.SUPPRESS)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _checkpoint_config(name: str):
    from mednext_accel import CheckpointConfig

    if name == "none":
        return None
    if name == "stage-0":
        return CheckpointConfig(stages=(0,))
    if name == "stage-0-1":
        return CheckpointConfig(stages=(0, 1))
    if name == "all-expansion":
        return CheckpointConfig()
    return CheckpointConfig(style="block")


def measure(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    from mednext_accel import mednext_base

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    assert args.batch_size is not None
    assert args.checkpoint is not None
    torch.manual_seed(1234)
    torch.backends.cudnn.benchmark = True
    shape = (args.batch_size, 1, 128, 128, 128)
    model = (
        mednext_base(
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            deep_supervision=True,
            checkpointing=_checkpoint_config(args.checkpoint),
            optimization=args.optimization,
        )
        .cuda()
        .train()
    )
    model.compile(mode=args.compile_mode, fullgraph=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sample = torch.randn(shape, device="cuda")
    target = torch.randint(3, (args.batch_size, 128, 128, 128), device="cuda")
    targets = {
        (size, size, size): (
            target
            if size == 128
            else F.interpolate(target[:, None].float(), size=(size,) * 3, mode="nearest")[
                :, 0
            ].long()
        )
        for size in (128, 64, 32, 16, 8)
    }

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(sample)
            loss = sum(
                F.cross_entropy(head, targets[tuple(head.shape[2:])]) for head in outputs
            ) / len(outputs)
        loss.backward()
        optimizer.step()
        return loss

    try:
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        events = []
        for _ in range(args.steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = step()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
    except Exception as error:
        if "out of memory" not in str(error).lower():
            raise
        return {
            "checkpoint": args.checkpoint,
            "batch_size": args.batch_size,
            "status": "oom",
            "error": str(error).splitlines()[0],
        }

    step_times = [start.elapsed_time(end) for start, end in events]
    median_step_ms = statistics.median(step_times)
    return {
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "status": "ok",
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "median_step_ms": median_step_ms,
        "mean_step_ms": statistics.mean(step_times),
        "step_stdev_ms": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
        "samples_per_second": args.batch_size * 1_000 / median_step_ms,
        "milliseconds_per_sample": median_step_ms / args.batch_size,
        "final_loss": loss.item(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "compile_mode": args.compile_mode,
        "optimization": args.optimization,
        "fullgraph": True,
    }


def sweep(args: argparse.Namespace) -> None:
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise SystemExit("batch sizes must be positive")
    if args.warmup < 1:
        raise SystemExit("warmup must be positive")
    if args.steps < 1:
        raise SystemExit("steps must be positive")
    payload: dict[str, Any] = {
        "workload": "MedNeXt Base 128^3, BF16, 3 classes, deep supervision, AdamW",
        "compile_mode": args.compile_mode,
        "optimization": args.optimization,
        "fullgraph": True,
        "warmup": args.warmup,
        "steps": args.steps,
        "results": [],
    }
    for checkpoint in args.checkpoints:
        for batch_size in args.batch_sizes:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--checkpoint",
                checkpoint,
                "--batch-size",
                str(batch_size),
                "--compile-mode",
                args.compile_mode,
                "--optimization",
                args.optimization,
                "--warmup",
                str(args.warmup),
                "--steps",
                str(args.steps),
            ]
            print(f"running checkpoint={checkpoint}, batch={batch_size}", flush=True)
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                sys.stderr.write(completed.stderr)
                raise SystemExit(completed.returncode)
            result = json.loads(completed.stdout.splitlines()[-1])
            payload["results"].append(result)
            if args.output is not None:
                _write_json(args.output, payload)
            if result["status"] == "ok":
                print(
                    f"  {result['median_step_ms']:.2f} ms, "
                    f"{result['samples_per_second']:.2f} samples/s, "
                    f"{result['peak_allocated_mib']:.0f} MiB allocated, "
                    f"{result['peak_reserved_mib']:.0f} MiB reserved",
                    flush=True,
                )
            else:
                print("  OOM", flush=True)
    if args.output is None:
        print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    if args.child:
        print(json.dumps(measure(args)))
    else:
        sweep(args)


if __name__ == "__main__":
    main()
