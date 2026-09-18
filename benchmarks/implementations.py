#!/usr/bin/env python3
"""Compare official, MONAI, and mednext-accel MedNeXt Base training."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-root",
        type=Path,
        required=True,
        help="checkout of https://github.com/MIC-DKFZ/MedNeXt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("all", "execution", "checkpoint"), default="all")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument("--fullgraph", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if min(args.warmup, args.steps, args.repeats) < 1:
        raise SystemExit("warmup, steps, and repeats must be positive")
    if not (args.official_root / "nnunet_mednext").is_dir():
        raise SystemExit("--official-root does not contain nnunet_mednext")

    sys.path.insert(0, str(args.official_root))
    import torch
    from torch.nn import functional as F

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    try:
        import monai
        from monai.networks.nets.mednext import MedNeXtBase as MonaiMedNeXtBase
    except ImportError as error:
        raise SystemExit("install the 'monai' extra to run this comparison") from error

    from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import (
        create_mednextv1_base,
    )
    from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import (
        MedNeXt as OfficialMedNeXt,
    )

    from mednext_accel import CheckpointConfig, mednext_base
    from mednext_accel.optimization import optimize

    expansion = [2, 3, 4, 4, 4, 4, 4, 3, 2]

    def build(config: dict[str, Any]):
        implementation = config["implementation"]
        checkpoint = config["checkpoint"]
        if implementation == "official":
            if checkpoint == "whole-block":
                model = OfficialMedNeXt(
                    in_channels=1,
                    n_channels=32,
                    n_classes=3,
                    exp_r=expansion,
                    kernel_size=3,
                    deep_supervision=True,
                    do_res=True,
                    do_res_up_down=True,
                    checkpoint_style="outside_block",
                    block_counts=[2] * 9,
                )
            else:
                model = create_mednextv1_base(1, 3, 3, True)
            return model.cuda().train()
        if implementation == "monai":
            return (
                MonaiMedNeXtBase(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=3,
                    kernel_size=3,
                    deep_supervision=True,
                )
                .cuda()
                .train()
            )

        checkpointing = None
        if checkpoint == "stage-0":
            checkpointing = CheckpointConfig(stages=(0,))
        elif checkpoint == "stage-0-1":
            checkpointing = CheckpointConfig(stages=(0, 1))
        elif checkpoint == "all-expansion":
            checkpointing = CheckpointConfig(stages=None)
        elif checkpoint == "whole-block":
            checkpointing = CheckpointConfig(style="block")
        model = (
            mednext_base(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
                deep_supervision=True,
                checkpointing=checkpointing,
            )
            .cuda()
            .train()
        )
        optimize(
            model,
            input_shape=(1, 1, 128, 128, 128),
            dtype=torch.bfloat16,
            policy=config["policy"],
            compile_mode=args.compile_mode,
        )
        return model

    def measure(config: dict[str, Any]) -> dict[str, Any]:
        torch.manual_seed(1234)
        model = build(config)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if config["compiled"]:
            model = torch.compile(model, mode=args.compile_mode, fullgraph=args.fullgraph)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sample = torch.randn(1, 1, 128, 128, 128, device="cuda")
        target = torch.randint(3, (1, 128, 128, 128), device="cuda")
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
                if isinstance(outputs, torch.Tensor):
                    outputs = (outputs,)
                loss = sum(
                    F.cross_entropy(head, targets[tuple(head.shape[2:])]) for head in outputs
                ) / len(outputs)
            loss.backward()
            optimizer.step()
            return loss

        for _ in range(args.warmup):
            loss = step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        events = []
        wall_start = time.perf_counter()
        for _ in range(args.steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = step()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        times = [start.elapsed_time(end) for start, end in events]
        result = config | {
            "parameters": parameters,
            "median_step_ms": statistics.median(times),
            "mean_wall_step_ms": (time.perf_counter() - wall_start) * 1_000 / args.steps,
            "step_stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "final_loss": loss.item(),
        }
        return result

    execution = [
        {"implementation": name, "policy": policy, "checkpoint": "none", "compiled": compiled}
        for compiled in (False, True)
        for name, policy in (
            ("official", "native"),
            ("monai", "native"),
            ("mednext-accel", "torch"),
            ("mednext-accel", "conservative"),
        )
    ]
    checkpoint = [
        {
            "implementation": "official",
            "policy": "native",
            "checkpoint": "whole-block",
            "compiled": True,
        },
        *[
            {
                "implementation": "mednext-accel",
                "policy": "conservative",
                "checkpoint": selection,
                "compiled": True,
            }
            for selection in ("stage-0", "stage-0-1", "all-expansion", "whole-block")
        ],
    ]
    configurations = {
        "all": execution + checkpoint,
        "execution": execution,
        "checkpoint": checkpoint,
    }[args.suite]
    try:
        official_commit = subprocess.run(
            ["git", "-C", str(args.official_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        official_commit = None

    payload: dict[str, Any] = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "monai": monai.__version__,
            "official_commit": official_commit,
            "compile_mode": args.compile_mode,
            "fullgraph": args.fullgraph,
            "warmup": args.warmup,
            "steps": args.steps,
            "repeats": args.repeats,
            "workload": "resident synthetic input, five-head mean cross entropy, AdamW",
        },
        "runs": [],
    }
    torch.backends.cudnn.benchmark = True
    for config in configurations:
        label = "/".join(str(config[key]) for key in config)
        for repeat in range(1, args.repeats + 1):
            print(f"running {label}, repeat {repeat}/{args.repeats}", flush=True)
            result = measure(config | {"repeat": repeat})
            gc.collect()
            torch.cuda.empty_cache()
            payload["runs"].append(result)
            _write_json(args.output, payload)
            print(
                f"  {result['median_step_ms']:.2f} ms, {result['peak_allocated_mib']:.0f} MiB",
                flush=True,
            )


if __name__ == "__main__":
    main()
