"""GPU campaign orchestration with shared kernel evidence and isolated model probes."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from ..optimization.schema import profile_to_primitive
from .batch_search import ProbeResult, search_batches
from .benchmark import run_json_subprocess
from .campaign import Campaign, Workload
from .execution import BatchSearchResult, WorkloadShapes, build_execution_plan
from .grouped import run_group_with_bisection
from .progress import ProgressEvent, ProgressReporter
from .synthesize import Measurement, measurement_from_result, synthesize_profile
from .validation import (
    apply_validation_results,
    decision_signature,
    segment_decisions,
    validation_batches,
)


def _checkpoint(value: str):
    from ..checkpointing import CheckpointConfig

    if value == "none":
        return None
    return CheckpointConfig(style="block" if value == "whole-block" else "expansion")


def _factory(variant: str):
    from ..models.mednext_v1 import mednext_base, mednext_large, mednext_medium, mednext_small

    return {
        "small": mednext_small,
        "base": mednext_base,
        "medium": mednext_medium,
        "large": mednext_large,
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
    optimization = payload.get("optimization", "reference")
    model = (
        _factory(str(workload["variant"]))(
            in_channels=int(workload["in_channels"]),
            out_channels=int(workload["out_channels"]),
            checkpointing=_checkpoint(str(workload["checkpointing"])),
            optimization=optimization,
        )
        .cuda()
        .train()
    )
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
    x = torch.randn(
        batch, channels, *spatial, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.randn(
        outputs, channels, 1, 1, 1, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    bias = torch.randn(outputs, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gradient = torch.randn(batch, outputs, *spatial, device="cuda", dtype=torch.bfloat16)

    def native():
        output = functional.conv3d(x, weight, bias)
        return (output, *torch.autograd.grad(output, (x, weight, bias), gradient))

    def candidate():
        flat = x.flatten(2)
        matrix = weight.flatten(1)
        output = torch.stack(
            [torch.addmm(bias[:, None], matrix, sample) for sample in flat.unbind()]
        ).reshape(batch, outputs, *spatial)
        return (output, *torch.autograd.grad(output, (x, weight, bias), gradient))

    expected, actual = native(), candidate()
    valid = all(
        torch.allclose(a, b, rtol=0.02, atol=0.02) for a, b in zip(actual, expected, strict=True)
    )
    native_ms, native_peak = _timed(native)
    candidate_ms, candidate_peak = _timed(candidate)
    return {
        "status": "ok",
        "valid": valid,
        "reference_ms": native_ms,
        "candidate_ms": candidate_ms,
        "reference_peak_bytes": native_peak,
        "candidate_peak_bytes": candidate_peak,
        "parameters": [],
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
    weight = torch.randn(channels, 1, kernel, kernel, kernel, device="cuda", dtype=torch.bfloat16)
    padding = kernel // 2
    if direction == "transpose":
        output = functional.conv_transpose3d(x, weight, stride=2, padding=padding, groups=channels)
    else:
        output = functional.conv3d(x, weight, stride=stride, padding=padding, groups=channels)
    gradient = torch.randn_like(output)

    if direction == "transpose" and phase == "backward_weight":

        def native():
            return torch.ops.aten.convolution_backward(
                gradient,
                x,
                weight,
                None,
                [2] * 3,
                [padding] * 3,
                [1] * 3,
                True,
                [0] * 3,
                channels,
                [False, True, False],
            )[1]

        splits = max(1, min(512, round(64 * batch * spatial[0] ** 3 / 64**3)))

        def candidate():
            return backend.depthwise_transpose_weight_grad(
                x, gradient, splits=splits, block=512, kernel_size=kernel
            )

        implementation = "triton_transpose_split_dw"
        parameters = (("dw_splits", splits), ("dw_block", 512))
    elif direction == "downsample" and phase == "backward_input":

        def native():
            return torch.ops.aten.convolution_backward(
                gradient,
                x,
                weight,
                None,
                [2] * 3,
                [padding] * 3,
                [1] * 3,
                False,
                [0] * 3,
                channels,
                [True, False, False],
            )[0]

        def candidate():
            return backend.depthwise_stride2_input_grad(gradient, weight, spatial, block=128)

        implementation = "triton_downsample_dx"
        parameters = (("dx_block", 128),)
    elif direction == "regular" and phase == "backward_input":

        def native():
            return torch.ops.aten.convolution_backward(
                gradient,
                x,
                weight,
                None,
                [1] * 3,
                [padding] * 3,
                [1] * 3,
                False,
                [0] * 3,
                channels,
                [True, False, False],
            )[0]

        def candidate():
            return backend.depthwise_input_grad(gradient, weight, block=128)

        implementation = "triton_depthwise_dx"
        parameters = (("dx_block", 128),)
    elif direction == "regular" and phase == "backward_weight":

        def native():
            return torch.ops.aten.convolution_backward(
                gradient,
                x,
                weight,
                None,
                [1] * 3,
                [padding] * 3,
                [1] * 3,
                False,
                [0] * 3,
                channels,
                [False, True, False],
            )[1]

        splits = max(1, min(512, round(64 * batch * spatial[0] ** 3 / 128**3)))

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
        "status": "ok",
        "valid": relative.item() < 0.02,
        "implementation": implementation,
        "reference_ms": native_ms,
        "candidate_ms": candidate_ms,
        "reference_peak_bytes": native_peak,
        "candidate_peak_bytes": candidate_peak,
        "parameters": parameters,
    }


def _kernel_group_probe(payload: dict[str, object]) -> dict[str, object]:
    import torch

    cases = payload["cases"]
    assert isinstance(cases, list)
    results: list[dict[str, object]] = []
    for case in cases:
        assert isinstance(case, dict)
        case_id = str(case["case_id"])
        try:
            family = str(case["family"])
            if family == "pointwise_conv3d":
                result = _pointwise_probe(case)
            elif family in ("depthwise_conv3d", "depthwise_conv_transpose3d"):
                result = _depthwise_probe(case)
            else:
                raise ValueError(f"unknown kernel family {family!r}")
        except torch.cuda.OutOfMemoryError:
            result = {"status": "oom", "message": "CUDA out of memory"}
        except Exception as error:
            result = {"status": "error", "message": f"{type(error).__name__}: {error}"}
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        results.append({**result, "case_id": case_id})
    return {"status": "ok", "results": results}


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
    if payload["kind"] == "kernel_group":
        return _kernel_group_probe(payload)
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

    model = (
        _factory(workload.variant)(
            in_channels=workload.in_channels,
            out_channels=workload.out_channels,
            checkpointing=_checkpoint(workload.checkpointing),
            optimization="reference",
        )
        .to("meta")
        .eval()
    )
    found: set[tuple[int, int, tuple[int, int, int]]] = set()
    hooks = []
    for module in model.modules():
        if type(module) is nn.Conv3d and module.kernel_size == (1, 1, 1):
            hooks.append(
                module.register_forward_hook(
                    lambda current, inputs, output: found.add(
                        (
                            current.in_channels,
                            current.out_channels,
                            tuple(int(item) for item in inputs[0].shape[2:]),
                        )
                    )
                )
            )
    with torch.no_grad():
        model(torch.empty(1, workload.in_channels, *workload.spatial, device="meta"))
    for hook in hooks:
        hook.remove()
    return tuple(sorted(found))


def _depthwise_shapes(
    workload: Workload,
) -> tuple[tuple[str, int, int, tuple[int, int, int]], ...]:
    import torch
    from torch import nn

    model = (
        _factory(workload.variant)(
            in_channels=workload.in_channels,
            out_channels=workload.out_channels,
            checkpointing=_checkpoint(workload.checkpointing),
            optimization="reference",
        )
        .to("meta")
        .eval()
    )
    found: set[tuple[str, int, int, tuple[int, int, int]]] = set()
    hooks = []
    for module in model.modules():
        if (
            type(module) in (nn.Conv3d, nn.ConvTranspose3d)
            and module.groups == module.in_channels == module.out_channels
        ):
            direction = (
                "transpose"
                if type(module) is nn.ConvTranspose3d
                else "downsample"
                if module.stride[0] == 2
                else "regular"
            )
            hooks.append(
                module.register_forward_hook(
                    lambda current, inputs, output, direction=direction: found.add(
                        (
                            direction,
                            current.in_channels,
                            current.kernel_size[0],
                            tuple(int(item) for item in inputs[0].shape[2:]),
                        )
                    )
                )
            )
    with torch.no_grad():
        model(torch.empty(1, workload.in_channels, *workload.spatial, device="meta"))
    for hook in hooks:
        hook.remove()
    return tuple(sorted(found))


def discover_workload_shapes(workload: Workload) -> WorkloadShapes:
    return WorkloadShapes(
        pointwise=_pointwise_shapes(workload),
        depthwise=_depthwise_shapes(workload),
    )


def _batches(
    campaign: Campaign,
    workload: Workload,
    total_vram: int,
    progress: ProgressReporter | None = None,
) -> tuple[tuple[int, ...], dict[int, dict[str, object]]]:
    model_results: dict[int, dict[str, object]] = {}

    def probe(batch: int) -> ProbeResult:
        if progress is not None:
            progress.emit(ProgressEvent("status", "batch-search", message=f"probing batch={batch}"))
        payload = {
            "kind": "model",
            "batch": batch,
            "workload": asdict(workload),
            "compile_mode": campaign.compile_mode,
        }
        result = _invoke(payload)
        model_results[batch] = result
        feasible = result.get("status") == "ok"
        if progress is not None:
            peak = int(result.get("peak_bytes", 0)) / 1024**3
            state = "feasible" if feasible else str(result.get("status", "failed"))
            progress.emit(
                ProgressEvent(
                    "status",
                    "batch-search",
                    message=f"batch={batch} {state} peak={peak:.1f} GiB",
                )
            )
        return ProbeResult(
            batch,
            feasible,
            int(result.get("peak_bytes", total_vram + 1)),
            None if feasible else str(result.get("message", result.get("status"))),
        )

    result = search_batches(campaign.batch_search, total_vram_bytes=total_vram, probe=probe)
    return (
        tuple(item.batch for item in result.probes if item.feasible),
        model_results,
    )


def _whole_model_accepts(
    objective: str,
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> bool:
    if any(
        result.get("status") != "ok" or not {"step_ms", "peak_bytes"} <= result.keys()
        for result in (reference, candidate)
    ):
        return False
    reference_ms = float(reference["step_ms"])
    candidate_ms = float(candidate["step_ms"])
    reference_peak = int(reference["peak_bytes"])
    candidate_peak = int(candidate["peak_bytes"])
    if objective == "memory":
        return candidate_peak < reference_peak and candidate_ms <= reference_ms * 1.10
    if objective == "throughput":
        return candidate_ms < reference_ms and candidate_peak <= reference_peak * 1.25
    return candidate_ms < reference_ms and candidate_peak <= reference_peak * 1.15


@dataclass(frozen=True, slots=True)
class ExecutionStatistics:
    """Planned logical cases and completed physical execution attempts.

    ``kernel_group_count`` includes failed and bisected child attempts, while
    ``whole_model_validation_count`` excludes the reference batch search.
    """

    requested_case_count: int
    kernel_case_count: int
    kernel_group_count: int
    whole_model_validation_count: int


@dataclass(frozen=True, slots=True)
class CampaignRun:
    measurements: tuple[Measurement, ...]
    statistics: ExecutionStatistics


def execute_campaign(campaign: Campaign, progress: ProgressReporter | None = None) -> CampaignRun:
    """Execute shared kernel groups and validate each workload's decision boundaries."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("profiling requires an NVIDIA CUDA device")
    device = torch.cuda.current_device()
    total_vram = torch.cuda.get_device_properties(device).total_memory

    def search(workload: Workload) -> BatchSearchResult:
        if progress is not None:
            progress.emit(
                ProgressEvent(
                    "workload",
                    "workload",
                    message=f"{workload.variant} · {workload.spatial} · {workload.checkpointing}",
                )
            )
            progress.emit(ProgressEvent("stage_start", "batch-search"))
        batches, reference_steps = _batches(campaign, workload, total_vram, progress=progress)
        return BatchSearchResult(batches, reference_steps)

    plan = build_execution_plan(campaign, discover=discover_workload_shapes, search=search)
    raw_results: dict[str, dict[str, object]] = {}
    group_attempts = 0
    group_total = len(plan.kernel_groups)
    if progress is not None:
        progress.emit(ProgressEvent("stage_start", "kernel-groups", total=group_total))

    def on_attempt(event: str, attempts: int, resolved_cases: int) -> None:
        nonlocal group_attempts, group_total
        if event == "scheduled":
            group_total += attempts
        else:
            group_attempts += attempts
        if progress is not None:
            progress.emit(
                ProgressEvent(
                    "advance",
                    "kernel-groups",
                    completed=group_attempts,
                    total=group_total,
                )
            )

    for group in plan.kernel_groups:
        raw_results.update(run_group_with_bisection(group, _invoke, on_attempt=on_attempt))

    cases_by_key = {case.key: case for case in plan.kernel_cases}
    measurements: list[Measurement] = []
    validation_count = 0
    for workload_run in plan.workloads:
        workload = workload_run.workload
        context_measurements = tuple(
            measurement_from_result(
                cases_by_key[key],
                raw_results.get(cases_by_key[key].identifier, {}),
                checkpointing=workload.checkpointing,
            )
            for key in workload_run.case_keys
        )
        by_batch: dict[int, list[Measurement]] = {batch: [] for batch in workload_run.batches}
        for item in context_measurements:
            by_batch[item.batch].append(item)
        segments = segment_decisions(
            {
                batch: decision_signature(items, campaign.objective)
                for batch, items in by_batch.items()
            }
        )
        endpoints = validation_batches(segments)
        if not endpoints:
            measurements.extend(context_measurements)
            continue
        provisional = synthesize_profile(
            context_measurements,
            name="campaign-candidate",
            sm=torch.cuda.get_device_capability(),
            objective=campaign.objective,
        )
        if progress is not None:
            progress.emit(ProgressEvent("stage_start", "validation", total=len(endpoints)))
        accepted: dict[int, bool] = {}
        for completed, batch in enumerate(endpoints, 1):
            candidate_step = _invoke(
                {
                    "kind": "model",
                    "batch": batch,
                    "workload": asdict(workload),
                    "compile_mode": campaign.compile_mode,
                    "optimization": profile_to_primitive(provisional),
                }
            )
            validation_count += 1
            accepted[batch] = _whole_model_accepts(
                campaign.objective, workload_run.reference_steps.get(batch, {}), candidate_step
            )
            if progress is not None:
                progress.emit(
                    ProgressEvent(
                        "advance",
                        "validation",
                        completed=completed,
                        total=len(endpoints),
                        message=f"batch={batch} {'accepted' if accepted[batch] else 'rejected'}",
                    )
                )
        measurements.extend(apply_validation_results(context_measurements, segments, accepted))

    return CampaignRun(
        measurements=tuple(measurements),
        statistics=ExecutionStatistics(
            requested_case_count=plan.requested_case_count,
            kernel_case_count=len(plan.kernel_cases),
            kernel_group_count=group_attempts,
            whole_model_validation_count=validation_count,
        ),
    )


def run_campaign(
    campaign: Campaign, progress: ProgressReporter | None = None
) -> tuple[Measurement, ...]:
    """Return schema-v1 measurements for callers using the original runner API."""
    return execute_campaign(campaign, progress=progress).measurements


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
