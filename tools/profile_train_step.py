#!/usr/bin/env python3
"""Profile MedNeXt forward and backward with a reproducible configuration."""

from __future__ import annotations

import argparse
import json
import time
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
    parser.add_argument("--optimization", choices=("auto", "reference"), default="auto")
    parser.add_argument(
        "--compile-mode",
        choices=(
            "none",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
    )
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--checkpoint-stages", type=int, nargs="*")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    import mednext_accel
    from mednext_accel import CheckpointConfig

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.steps < 1 or args.warmup < 1 or args.profile_steps < 0:
        raise SystemExit("warmup and steps must be positive; profile-steps cannot be negative")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    checkpointing = (
        None
        if args.checkpoint_stages is None
        else CheckpointConfig(stages=tuple(args.checkpoint_stages))
    )
    factory = getattr(mednext_accel, f"mednext_{args.variant}")
    model = (
        factory(
            in_channels=args.shape[1],
            out_channels=args.classes,
            base_channels=args.base_channels,
            kernel_size=args.kernel_size,
            deep_supervision=args.deep_supervision,
            checkpointing=checkpointing,
            optimization=args.optimization,
        )
        .cuda()
        .train()
    )
    report = model.explain_optimization(input_shape=tuple(args.shape), dtype=dtype, device="cuda")
    if args.compile_mode != "none":
        model.compile(mode=args.compile_mode, fullgraph=args.fullgraph)
    example = torch.randn(tuple(args.shape), device="cuda", dtype=dtype)

    def step() -> None:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            output = model(example)
            values = output if isinstance(output, (tuple, list)) else (output,)
            loss = sum(value.float().square().mean() for value in values)
        loss.backward()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(args.steps):
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1_000)

    args.output.mkdir(parents=True, exist_ok=True)
    if args.profile_steps:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            acc_events=True,
        ) as profiler:
            for _ in range(args.profile_steps):
                step()
                profiler.step()
        profiler.export_chrome_trace(str(args.output / "trace.json"))
        (args.output / "operators.txt").write_text(
            profiler.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total", row_limit=200
            )
            + "\n"
        )
    summary = {
        "configuration": vars(args) | {"output": str(args.output), "dtype": str(dtype)},
        "environment": {
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "optimization": {
            "profile": report.profile,
            "decisions": [
                {
                    "family": item.descriptor.family,
                    "direction": item.descriptor.direction,
                    "phase": item.phase,
                    "implementation": item.implementation,
                    "parameters": dict(item.parameters),
                    "rule": item.rule,
                    "confidence": item.confidence,
                }
                for item in report.decisions
            ],
            "warnings": list(report.warnings),
        },
        "step_ms": durations,
        "median_step_ms": sorted(durations)[len(durations) // 2],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
