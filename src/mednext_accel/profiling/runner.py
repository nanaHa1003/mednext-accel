"""GPU campaign orchestration with one subprocess per memory-sensitive probe."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .batch_search import ProbeResult, search_batches
from .benchmark import run_json_subprocess
from .campaign import Campaign, Workload
from .synthesize import Measurement


def _checkpoint(value: str):
    from ..checkpointing import CheckpointConfig

    if value == "none":
        return None
    return CheckpointConfig(style="block" if value == "whole-block" else "expansion")


def _factory(variant: str):
    from ..models.mednext_v1 import mednext_base, mednext_large, mednext_medium, mednext_small

    return {
        "small": mednext_small, "base": mednext_base,
        "medium": mednext_medium, "large": mednext_large,
    }[variant]


def _timed(function, *, warmup: int = 2, repetitions: int = 5) -> tuple[float, int]:
    import torch

    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repetitions, torch.cuda.max_memory_allocated()


def _model_probe(payload: dict[str, object]) -> dict[str, object]:
    import torch

    workload = payload["workload"]
    assert isinstance(workload, dict)
    model = _factory(str(workload["variant"]))(
        in_channels=int(workload["in_channels"]),
        out_channels=int(workload["out_channels"]),
        checkpointing=_checkpoint(str(workload["checkpointing"])),
        optimization="reference",
    ).cuda().train()
    compile_mode = str(payload.get("compile_mode", "default"))
    model.compile(mode=compile_mode, fullgraph=True)
    batch = int(payload["batch"])
    spatial = tuple(int(item) for item in workload["spatial"])
    sample = torch.randn(
        batch, int(workload["in_channels"]), *spatial, device="cuda", dtype=torch.bfloat16
    )
    optimizer = torch.optim.AdamW(model.parameters())

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(sample)
            primary = output[0] if isinstance(output, (tuple, list)) else output
            loss = primary.float().square().mean()
        loss.backward()
        optimizer.step()

    step_ms, peak = _timed(step, warmup=1, repetitions=1)
    return {"status": "ok", "step_ms": step_ms, "peak_bytes": peak}


def _pointwise_probe(payload: dict[str, object]) -> dict[str, object]:
    import torch
    from torch.nn import functional as functional

    batch = int(payload["batch"])
    channels = int(payload["in_channels"])
    outputs = int(payload["out_channels"])
    spatial = tuple(int(item) for item in payload["spatial_shape"])
    x = torch.randn(batch, channels, *spatial, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    weight = torch.randn(outputs, channels, 1, 1, 1, device="cuda", dtype=torch.bfloat16,
                         requires_grad=True)
    bias = torch.randn(outputs, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gradient = torch.randn(batch, outputs, *spatial, device="cuda", dtype=torch.bfloat16)

    def native():
        output = functional.conv3d(x, weight, bias)
        return (output, *torch.autograd.grad(output, (x, weight, bias), gradient))

    def candidate():
        flat = x.flatten(2)
        matrix = weight.flatten(1)
        output = torch.stack([
            torch.addmm(bias[:, None], matrix, sample) for sample in flat.unbind()
        ]).reshape(batch, outputs, *spatial)
        return (output, *torch.autograd.grad(output, (x, weight, bias), gradient))

    expected, actual = native(), candidate()
    valid = all(
        torch.allclose(a, b, rtol=0.02, atol=0.02)
        for a, b in zip(actual, expected, strict=True)
    )
    native_ms, native_peak = _timed(native)
    candidate_ms, candidate_peak = _timed(candidate)
    return {
        "status": "ok", "valid": valid, "reference_ms": native_ms,
        "candidate_ms": candidate_ms, "reference_peak_bytes": native_peak,
        "candidate_peak_bytes": candidate_peak, "parameters": [],
    }


def _depthwise_probe(payload: dict[str, object]) -> dict[str, object]:
    import torch
    from torch.nn import functional as functional

    from ..ops._triton import depthwise as backend

    batch = int(payload["batch"])
    channels = int(payload["in_channels"])
    kernel = int(payload["kernel_size"])
    spatial = tuple(int(item) for item in payload["spatial_shape"])
    direction = str(payload["direction"])
    phase = str(payload["phase"])
    stride = 2 if direction in ("downsample", "transpose") else 1
    x = torch.randn(batch, channels, *spatial, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        channels, 1, kernel, kernel, kernel, device="cuda", dtype=torch.bfloat16
    )
    padding = kernel // 2
    if direction == "transpose":
        output = functional.conv_transpose3d(
            x, weight, stride=2, padding=padding, groups=channels
        )
    else:
        output = functional.conv3d(
            x, weight, stride=stride, padding=padding, groups=channels
        )
    gradient = torch.randn_like(output)

    if direction == "transpose" and phase == "backward_weight":
        def native():
            return torch.ops.aten.convolution_backward(
                gradient, x, weight, None, [2] * 3, [padding] * 3, [1] * 3,
                True, [0] * 3, channels, [False, True, False],
            )[1]

        splits = max(1, min(512, round(64 * batch * spatial[0] ** 3 / 64 ** 3)))

        def candidate():
            return backend.depthwise_transpose_weight_grad(
                x, gradient, splits=splits, block=512, kernel_size=kernel
            )

        implementation = "triton_transpose_split_dw"
        parameters = (("dw_splits", splits), ("dw_block", 512))
    elif direction == "downsample" and phase == "backward_input":
        def native():
            return torch.ops.aten.convolution_backward(
                gradient, x, weight, None, [2] * 3, [padding] * 3, [1] * 3,
                False, [0] * 3, channels, [True, False, False],
            )[0]

        def candidate():
            return backend.depthwise_stride2_input_grad(
                gradient, weight, spatial, block=128
            )

        implementation = "triton_downsample_dx"
        parameters = (("dx_block", 128),)
    elif direction == "regular" and phase == "backward_input":
        def native():
            return torch.ops.aten.convolution_backward(
                gradient, x, weight, None, [1] * 3, [padding] * 3, [1] * 3,
                False, [0] * 3, channels, [True, False, False],
            )[0]

        def candidate():
            return backend.depthwise_input_grad(gradient, weight, block=128)

        implementation = "triton_depthwise_dx"
        parameters = (("dx_block", 128),)
    elif direction == "regular" and phase == "backward_weight":
        def native():
            return torch.ops.aten.convolution_backward(
                gradient, x, weight, None, [1] * 3, [padding] * 3, [1] * 3,
                False, [0] * 3, channels, [False, True, False],
            )[1]

        splits = max(1, min(512, round(64 * batch * spatial[0] ** 3 / 128 ** 3)))

        def candidate():
            return backend.depthwise_weight_grad(
                x, gradient, splits=splits, block=512, kernel_size=kernel
            )

        implementation = "triton_split_dw"
        parameters = (("dw_splits", splits), ("dw_block", 512))
    else:
        raise ValueError(f"unsupported depthwise probe {direction}/{phase}")

    expected, actual = native(), candidate()
    relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    native_ms, native_peak = _timed(native)
    candidate_ms, candidate_peak = _timed(candidate)
    return {
        "status": "ok", "valid": relative.item() < 0.02,
        "implementation": implementation, "reference_ms": native_ms,
        "candidate_ms": candidate_ms, "reference_peak_bytes": native_peak,
        "candidate_peak_bytes": candidate_peak, "parameters": parameters,
    }


def _child(payload: dict[str, object]) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if payload["kind"] == "model":
        return _model_probe(payload)
    if payload["kind"] == "pointwise":
        return _pointwise_probe(payload)
    if payload["kind"] == "depthwise":
        return _depthwise_probe(payload)
    raise ValueError(f"unknown probe kind {payload['kind']!r}")


def _invoke(payload: dict[str, object], *, timeout: float = 900) -> dict[str, object]:
    result = run_json_subprocess(
        [sys.executable, "-m", "mednext_accel.profiling.runner", "--child", json.dumps(payload)],
        timeout=timeout,
    )
    if result.status != "ok":
        return {"status": result.status, "message": result.message}
    return result.payload


def _pointwise_shapes(workload: Workload) -> tuple[tuple[int, int, tuple[int, int, int]], ...]:
    """Return unique MedNeXt pointwise shapes from a meta-tensor pass."""

    import torch
    from torch import nn

    model = _factory(workload.variant)(
        in_channels=workload.in_channels, out_channels=workload.out_channels,
        checkpointing=_checkpoint(workload.checkpointing), optimization="reference",
    ).to("meta").eval()
    found: set[tuple[int, int, tuple[int, int, int]]] = set()
    hooks = []
    for module in model.modules():
        if type(module) is nn.Conv3d and module.kernel_size == (1, 1, 1):
            hooks.append(module.register_forward_hook(
                lambda current, inputs, output: found.add((
                    current.in_channels, current.out_channels,
                    tuple(int(item) for item in inputs[0].shape[2:]),
                ))
            ))
    with torch.no_grad():
        model(torch.empty(
            1, workload.in_channels, *workload.spatial, device="meta"
        ))
    for hook in hooks:
        hook.remove()
    return tuple(sorted(found))


def _depthwise_shapes(
    workload: Workload,
) -> tuple[tuple[str, int, int, tuple[int, int, int]], ...]:
    import torch
    from torch import nn

    model = _factory(workload.variant)(
        in_channels=workload.in_channels, out_channels=workload.out_channels,
        checkpointing=_checkpoint(workload.checkpointing), optimization="reference",
    ).to("meta").eval()
    found: set[tuple[str, int, int, tuple[int, int, int]]] = set()
    hooks = []
    for module in model.modules():
        if (
            type(module) in (nn.Conv3d, nn.ConvTranspose3d)
            and module.groups == module.in_channels == module.out_channels
        ):
            direction = (
                "transpose" if type(module) is nn.ConvTranspose3d
                else "downsample" if module.stride[0] == 2 else "regular"
            )
            hooks.append(module.register_forward_hook(
                lambda current, inputs, output, direction=direction: found.add((
                    direction, current.in_channels, current.kernel_size[0],
                    tuple(int(item) for item in inputs[0].shape[2:]),
                ))
            ))
    with torch.no_grad():
        model(torch.empty(1, workload.in_channels, *workload.spatial, device="meta"))
    for hook in hooks:
        hook.remove()
    return tuple(sorted(found))


def _batches(campaign: Campaign, workload: Workload, total_vram: int) -> tuple[int, ...]:
    def probe(batch: int) -> ProbeResult:
        payload = {
            "kind": "model", "batch": batch, "workload": asdict(workload),
            "compile_mode": campaign.compile_mode,
        }
        result = _invoke(payload)
        feasible = result.get("status") == "ok"
        return ProbeResult(
            batch, feasible, int(result.get("peak_bytes", total_vram + 1)),
            None if feasible else str(result.get("message", result.get("status"))),
        )

    result = search_batches(
        campaign.batch_search, total_vram_bytes=total_vram, probe=probe
    )
    return tuple(item.batch for item in result.probes if item.feasible)


def run_campaign(campaign: Campaign) -> tuple[Measurement, ...]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("profiling requires an NVIDIA CUDA device")
    total_vram = torch.cuda.get_device_properties(0).total_memory
    measurements: list[Measurement] = []
    for workload_index, workload in enumerate(campaign.workloads, 1):
        print(
            f"[{workload_index}/{len(campaign.workloads)}] {workload.variant} "
            f"{workload.spatial} {workload.checkpointing}", file=sys.stderr,
        )
        batches = _batches(campaign, workload, total_vram)
        shapes = _pointwise_shapes(workload)
        depthwise_shapes = _depthwise_shapes(workload)
        for batch in batches:
            for in_channels, out_channels, spatial in shapes:
                result = _invoke({
                    "kind": "pointwise", "batch": batch,
                    "in_channels": in_channels, "out_channels": out_channels,
                    "spatial_shape": spatial,
                })
                if result.get("status") != "ok":
                    continue
                measurements.append(Measurement(
                    family="pointwise_conv3d", direction="regular", phase="training",
                    implementation="pointwise_gemm_per_sample", batch=batch,
                    spatial_shape=spatial, in_channels=in_channels,
                    out_channels=out_channels, dtype="bfloat16",
                    checkpointing=workload.checkpointing,
                    reference_ms=float(result["reference_ms"]),
                    candidate_ms=float(result["candidate_ms"]),
                    reference_peak_bytes=int(result["reference_peak_bytes"]),
                    candidate_peak_bytes=int(result["candidate_peak_bytes"]),
                    valid=bool(result["valid"]),
                    parameters=tuple(
                        (str(name), int(value)) for name, value in result["parameters"]
                    ),
                ))
            for direction, channels, kernel, spatial in depthwise_shapes:
                phases = (
                    ("backward_weight",) if direction == "transpose"
                    else ("backward_input",) if direction == "downsample"
                    else ("backward_input", "backward_weight")
                )
                for phase in phases:
                    result = _invoke({
                        "kind": "depthwise", "batch": batch,
                        "in_channels": channels, "kernel_size": kernel,
                        "spatial_shape": spatial, "direction": direction,
                        "phase": phase,
                    })
                    if result.get("status") != "ok":
                        continue
                    family = (
                        "depthwise_conv_transpose3d"
                        if direction == "transpose" else "depthwise_conv3d"
                    )
                    measurements.append(Measurement(
                        family=family, direction=direction, phase=phase,
                        implementation=str(result["implementation"]), batch=batch,
                        spatial_shape=spatial, in_channels=channels,
                        out_channels=channels, dtype="bfloat16",
                        checkpointing=workload.checkpointing,
                        reference_ms=float(result["reference_ms"]),
                        candidate_ms=float(result["candidate_ms"]),
                        reference_peak_bytes=int(result["reference_peak_bytes"]),
                        candidate_peak_bytes=int(result["candidate_peak_bytes"]),
                        valid=bool(result["valid"]),
                        parameters=tuple(
                            (str(name), int(value)) for name, value in result["parameters"]
                        ),
                    ))
    return tuple(measurements)


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--child")
    arguments = parser.parse_args(argv)
    try:
        result = _child(json.loads(arguments.child))
    except torch.cuda.OutOfMemoryError:  # type: ignore[name-defined]
        result = {"status": "oom", "message": "CUDA out of memory"}
    except Exception as error:
        result = {"status": "error", "message": f"{type(error).__name__}: {error}"}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
