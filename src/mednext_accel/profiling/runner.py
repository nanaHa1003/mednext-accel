"""GPU campaign orchestration with shared kernel evidence and isolated model probes."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass, field
from functools import wraps

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
    model_probe_failure,
)
from .execution import (
    WorkloadBatchSelection,
    WorkloadShapes,
    build_execution_plan,
    validate_campaign_dtypes,
)
from .grouped import run_group_with_bisection
from .matrix import KernelCase, depthwise_parameters
from .progress import ProgressEvent, ProgressReporter
from .synthesize import (
    Measurement,
    measurement_from_result,
    synthesize_profile,
)
from .validation import validate_components
from .whole_model import compare_model_results, effective_policy_identity


def _checkpoint(value: str):
    from ..checkpointing import CheckpointConfig

    if value == "none":
        return None
    return CheckpointConfig(style="block" if value == "whole-block" else "expansion")


def _factory(family: str, variant: str):
    from ..models.mednext_v1 import mednext_base, mednext_large, mednext_medium, mednext_small
    from ..models.mednext_v2 import mednext_v2_base, mednext_v2_wide

    if family == "mednext_v2":
        return {"base": mednext_v2_base, "wide": mednext_v2_wide}[variant]
    if family != "mednext_v1":
        raise ValueError(f"unknown model family {family!r}")
    return {
        "small": mednext_small,
        "base": mednext_base,
        "medium": mednext_medium,
        "large": mednext_large,
    }[variant]


@dataclass
class _ProbeRecord:
    stage: str
    result: dict[str, object] = field(default_factory=dict)

    def set_stage(self, stage: str) -> None:
        self.stage = stage


def _record_probe(kind):
    """Keep completed scalar evidence if a later allocation or operation fails."""

    def decorate(function):
        @wraps(function)
        def recorded(payload):
            import torch

            probe = _ProbeRecord(f"{kind}.setup", {"seed": int(payload.get("seed", 0))})
            if kind in ("pointwise", "depthwise"):
                probe.result["benchmark_kind"] = "raw_kernel_diagnostic"
            try:
                probe.result.update(function(payload, probe))
            except Exception as error:
                probe.result.update(
                    status="oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "error",
                    failure_stage=probe.stage,
                    message=f"{type(error).__name__}: {error}",
                )
            return probe.result

        return recorded

    return decorate


def _validate_pair(probe, name, native, candidate):
    # This frame owns precisely one detached reference/candidate pair. Returning
    # scalar metrics releases both tensors before the next component starts.
    probe.stage = f"validation.{name}.reference"
    expected = native()
    probe.stage = f"validation.{name}.candidate"
    actual = candidate()
    probe.stage = f"validation.{name}.metrics"
    result = validate_components({name: actual}, {name: expected})
    # A single numerical rejection is conclusive even when a later component
    # cannot finish. A passing verdict requires all components to complete.
    if not result["valid"]:
        probe.result["valid"] = False
    probe.result["validator"] = result["validator"]
    probe.result.setdefault("validation_metrics", {}).update(result["validation_metrics"])
    if probe.result.get("rejection_reason") is None:
        probe.result["rejection_reason"] = result["rejection_reason"]


def _time_pair(probe, native, candidate):
    for side, function in (("reference", native), ("candidate", candidate)):
        probe.stage = f"timing.{side}"
        duration, peak = _timed(
            function, on_stage=lambda stage, side=side: probe.set_stage(f"timing.{side}.{stage}")
        )
        probe.result.update({f"{side}_ms": duration, f"{side}_peak_bytes": peak})


def _timed(function, *, warmup: int = 2, repetitions: int = 5, on_stage=None) -> tuple[float, int]:
    import torch

    stage = on_stage if on_stage is not None else lambda value: None
    stage("warmup")
    for _ in range(warmup):
        function()
    stage("synchronize")
    torch.cuda.synchronize()
    stage("memory_reset")
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stage("measurement")
    start.record()
    for _ in range(repetitions):
        function()
    end.record()
    torch.cuda.synchronize()
    duration = start.elapsed_time(end) / repetitions
    stage("memory_read")
    return duration, torch.cuda.max_memory_allocated()


@_record_probe("model")
def _model_probe(payload: dict[str, object], probe: _ProbeRecord) -> dict[str, object]:
    import torch

    seed = int(payload.get("seed", 0))
    torch.manual_seed(seed)
    workload = payload["workload"]
    assert isinstance(workload, dict)
    optimization = payload.get("optimization", "reference")
    probe.stage = "model.initialization"
    model = (
        _factory(str(workload.get("model_family", "mednext_v1")), str(workload["variant"]))(
            in_channels=int(workload["in_channels"]),
            out_channels=int(workload["out_channels"]),
            checkpointing=_checkpoint(str(workload["checkpointing"])),
            optimization=optimization,
        )
        .cuda()
        .train()
    )
    compile_mode = str(payload.get("compile_mode", "default"))
    if compile_mode != "eager":
        probe.stage = "model.compile"
        model.compile(mode=compile_mode, fullgraph=True)
    batch = int(payload["batch"])
    spatial = tuple(int(item) for item in workload["spatial"])
    probe.stage = "model.allocation.input"
    sample = torch.randn(
        batch, int(workload["in_channels"]), *spatial, device="cuda", dtype=torch.bfloat16
    )
    probe.stage = "model.optimizer.initialization"
    optimizer = torch.optim.AdamW(model.parameters())

    last_loss = None
    timing_stage = "model.timing"

    def on_stage(stage):
        nonlocal timing_stage
        timing_stage = f"model.timing.{stage}"
        probe.stage = timing_stage

    def step() -> None:
        nonlocal last_loss
        probe.stage = f"{timing_stage}.zero_grad"
        optimizer.zero_grad(set_to_none=True)
        probe.stage = f"{timing_stage}.forward"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(sample)
            primary = output[0] if isinstance(output, (tuple, list)) else output
            loss = primary.float().square().mean()
        probe.stage = f"{timing_stage}.backward"
        loss.backward()
        probe.stage = f"{timing_stage}.optimizer"
        optimizer.step()
        last_loss = loss.detach()

    probe.stage = timing_stage
    step_ms, peak = _timed(step, warmup=1, repetitions=1, on_stage=on_stage)
    probe.result.update(step_ms=step_ms, peak_bytes=peak)
    probe.stage = "model.diagnostics"
    return {
        "status": "ok",
        "seed": seed,
        "step_ms": step_ms,
        "peak_bytes": peak,
        "diagnostics": {"loss": last_loss.item()},
    }


@_record_probe("pointwise")
def _pointwise_probe(payload: dict[str, object], probe: _ProbeRecord) -> dict[str, object]:
    import torch
    from torch.nn import functional as functional

    seed = int(payload.get("seed", 0))
    torch.manual_seed(seed)
    batch = int(payload["batch"])
    channels = int(payload["in_channels"])
    outputs = int(payload["out_channels"])
    spatial = tuple(int(item) for item in payload["spatial_shape"])
    probe.stage = "allocation.input"
    x = torch.randn(
        batch, channels, *spatial, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    probe.stage = "allocation.parameters"
    weight = torch.randn(
        outputs, channels, 1, 1, 1, device="cuda", dtype=torch.float32, requires_grad=True
    )
    bias = torch.randn(outputs, device="cuda", dtype=torch.float32, requires_grad=True)
    probe.stage = "allocation.gradient"
    gradient = torch.randn(batch, outputs, *spatial, device="cuda", dtype=torch.bfloat16)

    def native_forward(sample, matrix, offset):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return functional.conv3d(sample, matrix, offset)

    def candidate_forward(sample, matrix, offset):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            flat = sample.flatten(2)
            matrix = matrix.flatten(1)
            output = torch.stack(
                [torch.addmm(offset[:, None], matrix, item) for item in flat.unbind()]
            )
            return output.reshape(batch, outputs, *spatial)

    def component(forward, name):
        if name == "output":
            with torch.no_grad():
                return forward(x, weight, bias)
        # Only one gradient is requested and only its operand needs a graph.
        # Each call frees its forward/graph before the other side is evaluated.
        values = tuple(
            value.detach().requires_grad_(key == name)
            for key, value in zip(("dX", "dW", "dB"), (x, weight, bias), strict=True)
        )
        target = values[("dX", "dW", "dB").index(name)]
        output = forward(*values)
        return torch.autograd.grad(output, (target,), gradient)[0].detach()

    for name in ("output", "dX", "dW", "dB"):
        _validate_pair(
            probe,
            name,
            lambda name=name: component(native_forward, name),
            lambda name=name: component(candidate_forward, name),
        )
    probe.result["valid"] = probe.result["rejection_reason"] is None

    def training(forward):
        output = forward(x, weight, bias)
        return (output, *torch.autograd.grad(output, (x, weight, bias), gradient))

    _time_pair(probe, lambda: training(native_forward), lambda: training(candidate_forward))
    return {"status": "ok", "parameters": []}


@_record_probe("depthwise")
def _depthwise_probe(payload: dict[str, object], probe: _ProbeRecord) -> dict[str, object]:
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
    probe.stage = "allocation.input"
    x = torch.randn(batch, channels, *spatial, device="cuda", dtype=torch.bfloat16)
    probe.stage = "allocation.parameters"
    weight = torch.randn(channels, 1, kernel, kernel, kernel, device="cuda", dtype=torch.bfloat16)
    padding = kernel // 2
    probe.stage = "allocation.gradient_shape"
    if direction == "transpose":
        output = functional.conv_transpose3d(x, weight, stride=2, padding=padding, groups=channels)
    else:
        output = functional.conv3d(x, weight, stride=stride, padding=padding, groups=channels)
    probe.stage = "allocation.gradient"
    gradient = torch.randn_like(output)
    del output

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

    probe.result.update(implementation=implementation, parameters=parameters)
    component = "dX" if phase == "backward_input" else "dW"
    _validate_pair(probe, component, native, candidate)
    probe.result["valid"] = probe.result["rejection_reason"] is None
    _time_pair(probe, native, candidate)
    return {"status": "ok"}


@_record_probe("operator")
def _operator_probe(payload: dict[str, object], probe: _ProbeRecord) -> dict[str, object]:
    from .operator_benchmark import benchmark_operator

    probe.result.update(
        benchmark_kind="integrated_operator",
        compile_mode=payload["compile_mode"],
        gradient_mask=tuple(payload["gradient_mask"]),
    )
    return benchmark_operator(
        KernelCase.from_payload(payload),
        compile_mode=str(payload["compile_mode"]),
        seed=int(payload["seed"]),
        gradient_mask=tuple(payload["gradient_mask"]),
        probe=probe,
    )


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
                raw_probe = _pointwise_probe
            elif family in ("depthwise_conv3d", "depthwise_conv_transpose3d"):
                raw_probe = _depthwise_probe
            elif family == "global_response_norm3d":
                raw_probe = None
            else:
                raise ValueError(f"unknown kernel family {family!r}")
            if case.get("benchmark_kind") == "integrated_operator":
                result = _operator_probe(case)
                # Raw measurements are diagnostic only; their verdict and
                # timings never replace the integrated dispatch evidence.
                if raw_probe is not None:
                    gc.collect()
                    torch.cuda.empty_cache()
                    result["raw_diagnostic"] = {
                        **raw_probe(case),
                        "benchmark_kind": "raw_kernel_diagnostic",
                    }
            elif raw_probe is not None:
                result = raw_probe(case)
            else:
                raise ValueError("GRN requires an integrated training benchmark")
        except torch.cuda.OutOfMemoryError:
            result = {"status": "oom", "message": "CUDA out of memory", "failure_stage": "dispatch"}
        except Exception as error:
            result = {
                "status": "error",
                "message": f"{type(error).__name__}: {error}",
                "failure_stage": "dispatch",
            }
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
        return {
            "failure_stage": "subprocess",
            **result.payload,
            "status": result.status,
            "message": result.message,
        }
    return result.payload


def _meta_model(workload: Workload):
    import torch

    # Construct directly on meta, avoiding allocations for the Wide model.
    with torch.device("meta"):
        return _factory(workload.family, workload.variant)(
            in_channels=workload.in_channels,
            out_channels=workload.out_channels,
            checkpointing=_checkpoint(workload.checkpointing),
            optimization="reference",
        ).eval()


def _discovery_batch_size(workload: Workload) -> int:
    """Use the smallest batch for which the model's meta forward is valid."""

    bottleneck_shape = tuple((size + 15) // 16 for size in workload.spatial)
    if workload.family == "mednext_v2" and all(size == 1 for size in bottleneck_shape):
        return 2
    return 1


def _pointwise_shapes(workload: Workload) -> tuple[tuple[int, int, tuple[int, int, int]], ...]:
    """Return unique MedNeXt pointwise shapes from a meta-tensor pass."""

    import torch
    from torch import nn

    model = _meta_model(workload)
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
        model(
            torch.empty(
                _discovery_batch_size(workload),
                workload.in_channels,
                *workload.spatial,
                device="meta",
            )
        )
    for hook in hooks:
        hook.remove()
    return tuple(sorted(found))


def _depthwise_shapes(
    workload: Workload,
) -> tuple[tuple[str, int, int, tuple[int, int, int]], ...]:
    import torch
    from torch import nn

    model = _meta_model(workload)
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
        model(
            torch.empty(
                _discovery_batch_size(workload),
                workload.in_channels,
                *workload.spatial,
                device="meta",
            )
        )
    for hook in hooks:
        hook.remove()
    return tuple(sorted(found))


def _grn_shapes(workload: Workload) -> tuple[tuple[int, tuple[int, int, int]], ...]:
    import torch

    from ..ops.grn import GlobalResponseNorm3d

    if workload.family == "mednext_v1":
        return ()
    model = _meta_model(workload)
    found = set()
    hooks = [
        module.register_forward_hook(
            lambda current, inputs, output: found.add(
                (current.gamma.shape[1], tuple(int(size) for size in inputs[0].shape[2:]))
            )
        )
        for module in model.modules()
        if isinstance(module, GlobalResponseNorm3d)
    ]
    try:
        with torch.no_grad():
            model(
                torch.empty(
                    _discovery_batch_size(workload),
                    workload.in_channels,
                    *workload.spatial,
                    device="meta",
                )
            )
    finally:
        for hook in hooks:
            hook.remove()
    return tuple(sorted(found))


def discover_workload_shapes(workload: Workload) -> WorkloadShapes:
    return WorkloadShapes(
        pointwise=_pointwise_shapes(workload),
        depthwise=_depthwise_shapes(workload),
        grn=_grn_shapes(workload),
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
                    message=(
                        f"{workload.family}/{workload.variant} · "
                        f"{workload.spatial} · {workload.checkpointing}"
                    ),
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
            run_group_with_bisection(
                group,
                _invoke,
                on_attempt=on_attempt,
                seed=campaign.seed,
                compile_mode=campaign.compile_mode,
            )
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
        validation_plans.append((workload_run, context_measurements, endpoints))

    measurements = tuple(item for _, items, _ in validation_plans for item in items)
    sm = torch.cuda.get_device_capability()
    provisional = synthesize_profile(
        measurements,
        name=f"sm{sm[0]}{sm[1]}-local",
        sm=sm,
        objective=campaign.objective,
    )

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

    validation_count = 0
    model_comparisons = []
    for workload_run, _context_measurements, endpoints in validation_plans:
        workload = workload_run.workload
        for batch in endpoints:
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
        result = {"status": "oom", "message": "CUDA out of memory", "failure_stage": "dispatch"}
    except Exception as error:
        result = {
            "status": "error",
            "message": f"{type(error).__name__}: {error}",
            "failure_stage": "dispatch",
        }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
