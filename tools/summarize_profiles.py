#!/usr/bin/env python3
"""Print a CSV summary for profile_train_step output directories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = []
    for path in args.paths:
        files.extend(path.rglob("summary.json") if path.is_dir() else [path])
    rows = []
    for path in sorted(set(files)):
        data = json.loads(path.read_text())
        config = data.get("configuration", {})
        environment = data.get("environment", {})
        optimization = data.get("optimization", {})
        rows.append(
            {
                "path": str(path),
                "device": environment.get("device"),
                "torch": environment.get("torch"),
                "variant": config.get("variant"),
                "shape": config.get("shape"),
                "classes": config.get("classes"),
                "dtype": config.get("dtype"),
                "policy": optimization.get("policy", config.get("policy")),
                "compile_mode": optimization.get("compile_mode", config.get("compile_mode")),
                "median_step_ms": data.get("median_step_ms"),
                "peak_allocated_bytes": data.get("peak_allocated_bytes"),
                "peak_reserved_bytes": data.get("peak_reserved_bytes"),
            }
        )
    fields = list(rows[0]) if rows else ["path"]
    if args.output is None:
        stream = sys.stdout
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("w", newline="")
    try:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output is not None:
            stream.close()


if __name__ == "__main__":
    main()
