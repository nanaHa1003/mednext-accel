"""GPU campaign orchestration with shared kernel evidence and isolated model probes."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass

from ..optimization.descriptors import ExecutionContext
from ..optimization.policy import policy_to_primitive
from .batch_search import ProbeResult, search_batches
from .benchmark import run_json_subprocess
from .campaign import Campaign, Workload
from .evidence import (
    BatchProbeEvidence,
    BatchSearchEvidence,
    ModelComparisonEvidence,
    ModelProbeEvidence,
)
from .execution import (
    WorkloadBatchSelection,
    WorkloadShapes,
    build_execution_plan,
    validate_campaign_dtypes,
)
from .grouped import run_group_with_bisection
from .matrix import depthwise_parameters
from .progress import ProgressEvent, ProgressReporter
from .synthesize import (
    Measurement,
    measurement_from_result,
    measurement_identity,
    synthesize_profile,
)
from .validation import validate_components
from .whole_model import compare_model_results, effective_policy_identity, model_probe_failure


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

    seed = int(payload.get("seed", 0))
    torch.manual_seed(seed)
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

    last_loss = None

    def step() -> None:
        nonlocal last_loss
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(sample)
            primary = output[0] if isinstance(output, (tuple, list)) else output
            loss = primary.float().square().mean()
        loss.backward()
        optimizer.step()
        last_loss = loss.detach()

    step_ms, peak = _timed(step, warmup=1, repetitions=1)
    return {
        "status": "ok",
        "seed": seed,
        "step_ms": step_ms,
        "peak_bytes": peak,
        "diagnostics": {"loss": last_loss.item()},
    }


def _pointwise_probe(payload: dict[str, object]) -> dict[str, object]:
    import torch
    from torch.nn import functional as functional

    seed = int(payload.get("seed", 0))
    torch.manual_seed(seed)
    batch = int(payload["batch"])
    channels = int(payload["in_channels"])
    outputs = int(payload["out_channels"])
    spatial = tuple(int(item) for item in payload["spatial_shape"])
    x = torch.randn(
        batch, channels, *spatial, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.randn(
        outputs, channels, 1, 1, 1, device="cuda", dtype=torch.float32, requires_grad=True
    )
    bias = torch.randn(outputs, device="cuda", dtype=torch.float32, requires_grad=True)
    gradient = torch.randn(batch, outputs, *spatial, device="cuda", dtype=torch.bfloat16)

    def native():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = functional.conv3d(x, weight, bias)
        return (output, *torch.autograd.grad(output, (x, weight, bias), gradient))

    def candidate():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            flat = x.flatten(2)
            matrix = weight.flatten(1)
            output = torch.stack(
                [torch.addmm(bias[:, None], matrix, sample) for sample in flat.unbind()]
            ).reshape(batch, outputs, *spatial)
        return (output, *torch.autograd.grad(output, (x, weight, bias), gradient))

    expected, actual = native(), candidate()
    components = ("output", "dX", "dW", "dB")
    validation = validate_components(
        dict(zip(components, actual, strict=True)), dict(zip(components, expected, strict=True))
    )
    del expected, actual
    native_ms, native_peak = _timed(native)
    candidate_ms, candidate_peak = _timed(candidate)
    return {
        "status": "ok",
        **validation,
        "seed": seed,
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

    seed = int(payload.get("seed", 0))
    torch.manual_seed(seed)
    batch = int(payload["batch"])
    channels = int(payload["in_channels"])
    kernel = int(payload["kernel_size"])
    spatial = tuple(int(item) for item in payload["spatial_shape"])
    direction = str(payload["direction"])
    phase = str(payload["phase"])
    parameters = (
        tuple((name, int(value)) for name, value in payload["parameters"])
        if "parameters" in payload
        else depthwise_parameters(
            direction,
            phase,
            batch,
            spatial,
            channels=channels,
            kernel_size=kernel,
            sm=torch.cuda.get_device_capability(),
        )
    )
    launch = dict(parameters)
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

        def candidate():
            return backend.depthwise_transpose_weight_grad(
                x,
                gradient,
                splits=launch["dw_splits"],
                block=launch["dw_block"],
                kernel_size=kernel,
            )

        implementation = "triton_transpose_split_dw"
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
            return backend.depthwise_stride2_input_grad(
                gradient, weight, spatial, block=launch["dx_block"]
            )

        implementation = "triton_downsample_dx"
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
            return backend.depthwise_input_grad(gradient, weight, block=launch["dx_block"])

        implementation = "triton_depthwise_dx"
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

        def candidate():
            return backend.depthwise_weight_grad(
                x,
                gradient,
                splits=launch["dw_splits"],
                block=launch["dw_block"],
                kernel_size=kernel,
            )

        implementation = "triton_split_dw"
    else:
        raise ValueError(f"unsupported depthwise probe {direction}/{phase}")

    expected, actual = native(), candidate()
    component = "dX" if phase == "backward_input" else "dW"
    validation = validate_components({component: actual}, {component: expected})
    native_ms, native_peak = _timed(native)
    candidate_ms, candidate_peak = _timed(candidate)
    return {
        "status": "ok",
        **validation,
        "seed": seed,
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
        [sys.executable, "-m", "mednext_accel.profiling.runner", "--child"],
        timeout=timeout,
        input=json.dumps(payload),
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
) -> WorkloadBatchSelection:
    model_results: dict[int, dict[str, object]] = {}
    completed_probes = 0

    def probe(batch: int) -> ProbeResult:
        nonlocal completed_probes
        if progress is not None:
            progress.emit(ProgressEvent("status", "batch-search", message=f"probing batch={batch}"))
        payload = {
            "kind": "model",
            "batch": batch,
            "workload": asdict(workload),
            "compile_mode": campaign.compile_mode,
            "seed": campaign.seed,
        }
        result = _invoke(payload)
        model_results[batch] = result
        recorded = ModelProbeEvidence.from_result(result, seed=campaign.seed)
        failure = model_probe_failure("probe", recorded)
        feasible = failure is None
        if progress is not None:
            peak = (recorded.peak_bytes or 0) / 1024**3
            state = "feasible" if feasible else str(result.get("status", "failed"))
            progress.emit(
                ProgressEvent(
                    "status",
                    "batch-search",
                    message=f"batch={batch} {state} peak={peak:.1f} GiB",
                )
            )
            completed_probes += 1
            progress.emit(
                ProgressEvent(
                    "advance",
                    "batch-search",
                    completed=completed_probes,
                    message=f"batch={batch} {state} peak={peak:.1f} GiB",
                )
            )
        return ProbeResult(
            batch,
            feasible,
            int(recorded.peak_bytes or 0),
            failure,
        )

    result = search_batches(campaign.batch_search, total_vram_bytes=total_vram, probe=probe)
    selected = result.maximum_feasible
    if progress is not None:
        progress.emit(
            ProgressEvent(
                "status",
                "batch-search",
                message=f"maximum feasible batch: {selected}" if selected else "no feasible batch",
            )
        )
    budget = int(total_vram * campaign.batch_search.memory_fraction)
    evidence = BatchSearchEvidence(
        workload=asdict(workload),
        budget_bytes=budget,
        attempts=tuple(
            BatchProbeEvidence(
                batch=probe.batch,
                result=ModelProbeEvidence.from_result(
                    model_results[probe.batch], seed=campaign.seed
                ),
                within_budget=(
                    (
                        peak := ModelProbeEvidence.from_result(
                            model_results[probe.batch], seed=campaign.seed
                        ).peak_bytes
                    )
                    is not None
                    and 0 < peak <= budget
                ),
                feasible=probe.feasible,
                reason=probe.reason,
            )
            for probe in result.probes
        ),
        selected_maximum=selected,
    )
    return WorkloadBatchSelection(
        (() if selected == 0 else (selected,)),
        {} if selected == 0 else {selected: model_results[selected]},
        evidence,
    )


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
    batch_searches: tuple[BatchSearchEvidence, ...] = ()
    model_comparisons: tuple[ModelComparisonEvidence, ...] = ()

    @property
    def policy_measurements(self) -> tuple[Measurement, ...]:
        """Temporary publication input; source numerical evidence stays intact.

        Final negative-only overlays and policy publication are handled separately.
        A rejection removes positive rules for every indistinguishable match.
        """
        rejected = {
            measurement_identity(item)
            for comparison in self.model_comparisons
            if not comparison.policy_accepted
            for item in self.measurements
            if item.batch == comparison.batch and item.workload == comparison.workload
        }
        return tuple(
            item for item in self.measurements if measurement_identity(item) not in rejected
        )


def execute_campaign(campaign: Campaign, progress: ProgressReporter | None = None) -> CampaignRun:
    """Execute shared kernel groups and validate each workload's decision boundaries."""
    validate_campaign_dtypes(campaign)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("profiling requires an NVIDIA CUDA device")
    device = torch.cuda.current_device()
    total_vram = torch.cuda.get_device_properties(device).total_memory

    def search(workload: Workload) -> WorkloadBatchSelection:
        if progress is not None:
            progress.emit(
                ProgressEvent(
                    "workload",
                    "workload",
                    message=f"{workload.variant} · {workload.spatial} · {workload.checkpointing}",
                )
            )
            progress.emit(ProgressEvent("stage_start", "batch-search"))
        return _batches(campaign, workload, total_vram, progress=progress)

    plan = build_execution_plan(
        campaign,
        sm=torch.cuda.get_device_capability(),
        discover=discover_workload_shapes,
        search=search,
        progress=progress,
    )
    raw_results: dict[str, dict[str, object]] = {}
    group_attempts = 0
    group_total = len(plan.kernel_groups)
    resolved_cases = 0
    if progress is not None:
        progress.emit(
            ProgressEvent(
                "status",
                "kernel-groups",
                message=(
                    f"kernel plan: {group_total} planned groups, "
                    f"{len(plan.kernel_cases)} unique experiments, "
                    f"{plan.requested_case_count} requested references"
                ),
            )
        )
        progress.emit(ProgressEvent("stage_start", "kernel-groups", total=group_total))

    def on_attempt(event: str, attempts: int, resolved_delta: int) -> None:
        nonlocal group_attempts, group_total, resolved_cases
        if event == "scheduled":
            group_total += attempts
        else:
            group_attempts += attempts
            resolved_cases += resolved_delta
        if progress is not None:
            progress.emit(
                ProgressEvent(
                    "advance",
                    "kernel-groups",
                    completed=group_attempts,
                    total=group_total,
                    secondary_completed=resolved_cases,
                    secondary_total=len(plan.kernel_cases),
                )
            )

    for group in plan.kernel_groups:
        raw_results.update(
            run_group_with_bisection(group, _invoke, on_attempt=on_attempt, seed=campaign.seed)
        )

    cases_by_key = {case.key: case for case in plan.kernel_cases}
    validation_plans = []
    for workload_run in plan.workloads:
        workload = workload_run.workload
        context_measurements = tuple(
            measurement_from_result(
                cases_by_key[key],
                raw_results.get(cases_by_key[key].identifier, {}),
                checkpointing=workload.checkpointing,
                objective=campaign.objective,
                workload=asdict(workload),
            )
            for key in workload_run.case_keys
        )
        endpoints = workload_run.batches
        provisional = (
            synthesize_profile(
                context_measurements,
                name="campaign-candidate",
                sm=torch.cuda.get_device_capability(),
                objective=campaign.objective,
            )
            if endpoints
            else None
        )
        validation_plans.append((workload_run, context_measurements, endpoints, provisional))

    validation_total = sum(len(item[2]) for item in validation_plans)
    if progress is not None:
        progress.emit(
            ProgressEvent(
                "status",
                "validation",
                message=f"validation plan: {validation_total} whole-model validations",
            )
        )
        progress.emit(ProgressEvent("stage_start", "validation", total=validation_total))

    measurements: list[Measurement] = []
    validation_count = 0
    model_comparisons = []
    for workload_run, context_measurements, endpoints, provisional in validation_plans:
        workload = workload_run.workload
        for batch in endpoints:
            assert provisional is not None
            candidate_step = _invoke(
                {
                    "kind": "model",
                    "batch": batch,
                    "workload": asdict(workload),
                    "compile_mode": campaign.compile_mode,
                    "optimization": policy_to_primitive(provisional),
                    "seed": campaign.seed,
                }
            )
            validation_count += 1
            context = ExecutionContext(
                phase="training",
                device_type="cuda",
                sm=torch.cuda.get_device_capability(),
                total_vram_bytes=total_vram,
                dtype=workload.dtypes[0],
                batch_size=batch,
                spatial_shape=workload.spatial,
                model_family=workload.model_family,
                variant=workload.variant,
                checkpointing=workload.checkpointing,
            )
            comparison = compare_model_results(
                campaign.objective,
                workload_run.reference_steps.get(batch, {}),
                candidate_step,
                workload=asdict(workload),
                batch=batch,
                seed=campaign.seed,
                effective_policy=effective_policy_identity(provisional, context),
            )
            model_comparisons.append(comparison)
            if progress is not None:
                progress.emit(
                    ProgressEvent(
                        "advance",
                        "validation",
                        completed=validation_count,
                        total=validation_total,
                        message=f"batch={batch} {comparison.reason}",
                    )
                )
        measurements.extend(context_measurements)

    return CampaignRun(
        measurements=tuple(measurements),
        batch_searches=tuple(
            item.batch_search for item in plan.workloads if item.batch_search is not None
        ),
        model_comparisons=tuple(model_comparisons),
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
    """Return immutable kernel evidence without model acceptance filtering."""
    return execute_campaign(campaign, progress=progress).measurements


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--child", nargs="?", const="-")
    arguments = parser.parse_args(argv)
    try:
        payload = sys.stdin.read() if arguments.child == "-" else arguments.child
        result = _child(json.loads(payload))
    except torch.cuda.OutOfMemoryError:  # type: ignore[name-defined]
        result = {"status": "oom", "message": "CUDA out of memory"}
    except Exception as error:
        result = {"status": "error", "message": f"{type(error).__name__}: {error}"}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
