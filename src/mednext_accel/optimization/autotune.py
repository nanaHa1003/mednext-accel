"""Per-shape CUDA benchmarking and versioned policy cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .. import __version__
from .report import BackendSelections, CompileMode, KernelMeasurement

_CACHE_VERSION = 1


@dataclass(frozen=True, slots=True)
class AutotuneResult:
    selections: BackendSelections
    measurements: tuple[KernelMeasurement, ...]
    cache_hit: bool
    cache_key: str


def _bench(function: Any, warmup: int, repetitions: int) -> float:
    import triton

    return float(triton.testing.do_bench(function, warmup=warmup, rep=repetitions))


def _benchmark_pointwise(
    in_channels: int,
    out_channels: int,
    spatial: tuple[int, ...],
    dtype: torch.dtype,
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    x = torch.randn((1, in_channels, *spatial), device="cuda", dtype=dtype)
    weight = torch.randn((out_channels, in_channels, 1, 1, 1), device="cuda", dtype=dtype)
    bias = torch.randn((out_channels,), device="cuda", dtype=dtype)
    gradient = torch.randn((1, out_channels, *spatial), device="cuda", dtype=dtype)

    def native() -> None:
        F.conv3d(x, weight, bias)
        torch.ops.aten.convolution_backward(
            gradient,
            x,
            weight,
            [out_channels],
            [1] * 3,
            [0] * 3,
            [1] * 3,
            False,
            [0] * 3,
            1,
            [True, True, True],
        )

    def candidate() -> None:
        flattened = x.flatten(2)[0]
        matrix = weight.flatten(1)
        matrix.t().mm(gradient.flatten(2)[0])
        gradient.flatten(2)[0].mm(flattened.t())
        gradient.flatten(2)[0].sum(dim=1)
        torch.addmm(bias[:, None], matrix, flattened)

    return _bench(native, warmup, repetitions), _bench(candidate, warmup, repetitions)


def _benchmark_depthwise(
    kind: str,
    channels: int,
    spatial: int,
    dtype: torch.dtype,
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    from ..ops._triton import depthwise as backend

    x = torch.randn((1, channels, spatial, spatial, spatial), device="cuda", dtype=dtype)
    weight = torch.randn((channels, 1, 3, 3, 3), device="cuda", dtype=dtype)
    bias = torch.randn((channels,), device="cuda", dtype=dtype)
    if kind == "regular":
        gradient_shape = (spatial,) * 3

        def forward() -> Tensor:
            return F.conv3d(x, weight, bias, padding=1, groups=channels)

    elif kind == "transpose":
        gradient_shape = (2 * spatial - 1,) * 3

        def forward() -> Tensor:
            return F.conv_transpose3d(x, weight, bias, stride=2, padding=1, groups=channels)

    elif kind == "downsample":
        gradient_shape = ((spatial + 1) // 2,) * 3

        def forward() -> Tensor:
            return F.conv3d(x, weight, bias, stride=2, padding=1, groups=channels)
    else:
        raise ValueError(f"unknown depthwise kind {kind!r}")
    gradient = torch.randn((1, channels, *gradient_shape), device="cuda", dtype=dtype)

    def native() -> None:
        forward()
        torch.ops.aten.convolution_backward(
            gradient,
            x,
            weight,
            [channels],
            [1 if kind == "regular" else 2] * 3,
            [1] * 3,
            [1] * 3,
            kind == "transpose",
            [0] * 3,
            channels,
            [True, True, True],
        )

    if kind == "regular":
        splits, block, dx_block = backend._REGULAR_CONFIGS[(channels, spatial)]

        def candidate() -> None:
            forward()
            backend.depthwise_input_grad(gradient, weight, block=dx_block)
            backend.depthwise_weight_grad(x, gradient, splits, block)
            gradient.sum(dim=(0, 2, 3, 4))

    elif kind == "transpose":
        splits, block = backend._TRANSPOSE_CONFIGS[(channels, spatial)]

        def candidate() -> None:
            forward()
            torch.ops.aten.convolution_backward(
                gradient,
                x,
                weight,
                [channels],
                [2] * 3,
                [1] * 3,
                [1] * 3,
                True,
                [0] * 3,
                channels,
                [True, False, True],
            )
            backend.depthwise_transpose_weight_grad(x, gradient, splits, block)

    else:
        dx_block = backend._DOWNSAMPLE_CONFIGS[(channels, spatial)]

        def candidate() -> None:
            forward()
            backend.depthwise_stride2_input_grad(gradient, weight, (spatial,) * 3, block=dx_block)
            torch.ops.aten.convolution_backward(
                gradient,
                x,
                weight,
                [channels],
                [2] * 3,
                [1] * 3,
                [1] * 3,
                False,
                [0] * 3,
                channels,
                [False, True, True],
            )

    return _bench(native, warmup, repetitions), _bench(candidate, warmup, repetitions)


def _capture_shapes(
    model: nn.Module,
    input_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> list[tuple[str, nn.Module, tuple[int, ...]]]:
    records: list[tuple[str, nn.Module, tuple[int, ...]]] = []
    handles: list[Any] = []

    def hook(name: str):
        def record(module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            records.append((name, module, tuple(inputs[0].shape[2:])))

        return record

    for name, module in model.named_modules():
        if (
            type(module) in (nn.Conv3d, nn.ConvTranspose3d)
            or getattr(module, "_mednext_accel_backend_kind", None) is not None
        ):
            handles.append(module.register_forward_pre_hook(hook(name)))
    training_states = {module: module.training for module in model.modules()}
    model.eval()
    example = torch.randn(input_shape, device="cuda", dtype=dtype)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            model(example)
    finally:
        for handle in handles:
            handle.remove()
        for module, training in training_states.items():
            module.training = training
    return records


def _cache_key(
    records: list[tuple[str, nn.Module, tuple[int, ...]]],
    input_shape: tuple[int, ...],
    dtype: torch.dtype,
    compile_mode: CompileMode,
) -> str:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    signature = [
        (
            name,
            type(module).__name__,
            module.in_channels,
            module.out_channels,
            (
                (module.kernel_size,) * 3
                if isinstance(module.kernel_size, int)
                else tuple(module.kernel_size)
            ),
            tuple(module.stride),
            spatial,
        )
        for name, module, spatial in records
    ]
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    value = {
        "schema": _CACHE_VERSION,
        "mednext_accel": __version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": triton_version,
        "device": properties.name,
        "capability": torch.cuda.get_device_capability(),
        "total_memory": properties.total_memory,
        "dtype": str(dtype),
        "compile_mode": compile_mode,
        "input_shape": input_shape,
        "model": signature,
        "implementation": hashlib.sha256(
            Path(__file__).read_bytes()
            + (Path(__file__).parents[1] / "ops" / "_triton" / "depthwise.py").read_bytes()
        ).hexdigest(),
    }
    encoded = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _result_to_json(result: AutotuneResult) -> dict[str, Any]:
    return {
        "selections": asdict(result.selections),
        "measurements": [asdict(measurement) for measurement in result.measurements],
    }


def _result_from_json(key: str, value: dict[str, Any]) -> AutotuneResult:
    selections = BackendSelections(
        **{
            name: tuple(tuple(item) for item in items)
            for name, items in value["selections"].items()
        }
    )
    measurements = tuple(
        KernelMeasurement(
            kind=item["kind"],
            shape=tuple(item["shape"]),
            native_ms=float(item["native_ms"]),
            candidate_ms=float(item["candidate_ms"]),
            selected=bool(item["selected"]),
        )
        for item in value["measurements"]
    )
    return AutotuneResult(selections, measurements, True, key)


def _read_cache(path: Path, key: str) -> AutotuneResult | None:
    try:
        data = json.loads(path.read_text())
        if data.get("version") != _CACHE_VERSION:
            return None
        value = data.get("entries", {}).get(key)
        return None if value is None else _result_from_json(key, value)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(path: Path, key: str, result: AutotuneResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError, TypeError):
        data = {}
    if data.get("version") != _CACHE_VERSION:
        data = {"version": _CACHE_VERSION, "entries": {}}
    data.setdefault("entries", {})[key] = _result_to_json(result)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def autotune_selections(
    model: nn.Module,
    *,
    input_shape: tuple[int, ...],
    dtype: torch.dtype,
    warmup: int = 10,
    repetitions: int = 50,
    cache_path: str | os.PathLike[str] | None = None,
    include_stride2_input_grad: bool = False,
    compile_mode: CompileMode = "default",
    minimum_speedup: float = 1.02,
) -> AutotuneResult:
    """Benchmark each unique eligible shape and return selected backends."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for autotuning")
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must have parameters") from error
    if device.type != "cuda":
        raise ValueError("model must be on CUDA before autotuning")
    if len(input_shape) != 5 or input_shape[0] != 1:
        raise ValueError("autotuning requires a batch-one NCDHW input shape")
    if warmup < 1 or repetitions < 1:
        raise ValueError("warmup and repetitions must be positive")

    records = _capture_shapes(model, input_shape, dtype)
    key = _cache_key(records, input_shape, dtype, compile_mode)
    path = None if cache_path is None else Path(cache_path)
    if path is not None:
        cached = _read_cache(path, key)
        if cached is not None:
            return cached

    from ..ops._triton import depthwise as backend

    selected: dict[str, set[tuple[int, ...]]] = {
        "pointwise_gemm": set(),
        "depthwise_regular": set(),
        "depthwise_transpose": set(),
        "depthwise_downsample": set(),
    }
    measurements: list[KernelMeasurement] = []
    seen: set[tuple[object, ...]] = set()
    for _, module, spatial in records:
        kind: str | None = None
        shape: tuple[int, ...]
        wrapper_kind = getattr(module, "_mednext_accel_backend_kind", None)
        if wrapper_kind == "pointwise_gemm" or (
            type(module) is nn.Conv3d
            and module.kernel_size == (1, 1, 1)
            and module.stride == (1, 1, 1)
            and module.groups == 1
        ):
            kind = "pointwise_gemm"
            shape = (module.in_channels, module.out_channels, *spatial)
        elif wrapper_kind == "depthwise_regular" or (
            type(module) is nn.Conv3d
            and module.groups == module.in_channels == module.out_channels
            and module.stride == (1, 1, 1)
            and (module.in_channels, spatial[0]) in backend._REGULAR_CONFIGS
        ):
            kind = "depthwise_regular"
            shape = (module.in_channels, spatial[0])
        elif wrapper_kind == "depthwise_transpose" or (
            type(module) is nn.ConvTranspose3d
            and module.groups == module.in_channels == module.out_channels
            and module.stride == (2, 2, 2)
            and (module.in_channels, spatial[0]) in backend._TRANSPOSE_CONFIGS
        ):
            kind = "depthwise_transpose"
            shape = (module.in_channels, spatial[0])
        elif include_stride2_input_grad and (
            wrapper_kind == "depthwise_downsample"
            or (
                type(module) is nn.Conv3d
                and module.groups == module.in_channels == module.out_channels
                and module.stride == (2, 2, 2)
                and (module.in_channels, spatial[0]) in backend._DOWNSAMPLE_CONFIGS
            )
        ):
            kind = "depthwise_downsample"
            shape = (module.in_channels, spatial[0])
        else:
            continue
        identity = (kind, *shape)
        if identity in seen:
            continue
        seen.add(identity)
        if kind == "pointwise_gemm":
            native_ms, candidate_ms = _benchmark_pointwise(
                shape[0], shape[1], shape[2:], dtype, warmup, repetitions
            )
        else:
            native_ms, candidate_ms = _benchmark_depthwise(
                kind.removeprefix("depthwise_"),
                shape[0],
                shape[1],
                dtype,
                warmup,
                repetitions,
            )
        use_candidate = native_ms / candidate_ms >= minimum_speedup
        measurements.append(KernelMeasurement(kind, shape, native_ms, candidate_ms, use_candidate))
        if use_candidate:
            selected[kind].add(shape)

    selections = BackendSelections(
        **{name: tuple(sorted(values)) for name, values in selected.items()}
    )
    result = AutotuneResult(selections, tuple(measurements), False, key)
    if path is not None:
        _write_cache(path, key, result)
    return result
