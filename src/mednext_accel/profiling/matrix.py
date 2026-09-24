"""Context-free kernel cases used by the profiling campaign."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

from ..optimization.descriptors import ExecutionContext, OperatorDescriptor
from ..optimization.parameters import parameter_defaults
from .campaign import Workload


@dataclass(frozen=True, slots=True, order=True)
class KernelCaseKey:
    family: str
    direction: str
    phase: str
    batch: int
    spatial_shape: tuple[int, int, int]
    in_channels: int
    out_channels: int
    kernel_size: int | None
    dtype: str
    implementation: str
    parameters: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.family == "global_response_norm3d":
            if self.kernel_size is not None:
                raise ValueError("GRN kernel_size must be None")
            if self.phase != "training" or self.direction != "regular":
                raise ValueError("GRN cases require regular direction and combined training phase")
            if self.in_channels != self.out_channels:
                raise ValueError("GRN input and output channels must match")
            if self.parameters:
                raise ValueError("GRN has no launch parameters")
        elif type(self.kernel_size) is not int or self.kernel_size < 1:
            raise ValueError("kernel_size must be a positive integer for convolution cases")


@dataclass(frozen=True, slots=True)
class KernelCase:
    key: KernelCaseKey

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> KernelCase:
        """Recover the case key without folding execution settings into identity."""
        values = {field.name: payload[field.name] for field in fields(KernelCaseKey)}
        values["spatial_shape"] = tuple(values["spatial_shape"])
        values["parameters"] = tuple(tuple(item) for item in values["parameters"])
        return cls(KernelCaseKey(**values))

    @property
    def identifier(self) -> str:
        document = json.dumps(asdict(self.key), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(document.encode()).hexdigest()[:16]

    @property
    def comparison_identifier(self) -> str:
        """Identify shared tensor inputs independently of the candidate recipe."""
        fields = asdict(self.key)
        del fields["implementation"]
        del fields["parameters"]
        document = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(document.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class KernelGroup:
    category: str
    batch: int
    cases: tuple[KernelCase, ...]


def depthwise_parameters(
    direction: str,
    phase: str,
    batch: int,
    spatial: tuple[int, int, int],
    *,
    channels: int = 1,
    kernel_size: int = 3,
    sm: tuple[int, int] | None = None,
) -> tuple[tuple[str, int], ...]:
    """Use the same Python launch recipe as runtime for the execution SM."""
    implementations = {
        ("regular", "backward_input"): "triton_depthwise_dx",
        ("downsample", "backward_input"): "triton_downsample_dx",
        ("regular", "backward_weight"): "triton_split_dw",
        ("transpose", "backward_weight"): "triton_transpose_split_dw",
    }
    implementation = implementations.get((direction, phase))
    if implementation is None:
        raise ValueError(f"unsupported depthwise probe {direction}/{phase}")
    descriptor = OperatorDescriptor(
        "depthwise_conv_transpose3d" if direction == "transpose" else "depthwise_conv3d",
        direction,
        channels,
        channels,
        (kernel_size,) * 3,
        (1 if direction == "regular" else 2,) * 3,
        (kernel_size // 2,) * 3,
        (1,) * 3,
        channels,
    )
    context = ExecutionContext(
        "training",
        "cuda",
        sm,
        0,
        "bfloat16",
        batch,
        spatial,
        "mednext_v1",
        "base",
        "none",
    )
    return tuple(parameter_defaults(implementation, descriptor, context).items())


def _case(
    *,
    family: str,
    direction: str,
    phase: str,
    batch: int,
    spatial_shape: tuple[int, int, int],
    in_channels: int,
    out_channels: int,
    kernel_size: int | None,
    dtype: str,
    implementation: str,
    parameters: tuple[tuple[str, int], ...] = (),
) -> KernelCase:
    return KernelCase(
        KernelCaseKey(
            family=family,
            direction=direction,
            phase=phase,
            batch=batch,
            spatial_shape=spatial_shape,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dtype=dtype,
            implementation=implementation,
            parameters=parameters,
        )
    )


def build_workload_cases(
    workload: Workload,
    batches: tuple[int, ...],
    *,
    pointwise_shapes: tuple[tuple[int, int, tuple[int, int, int]], ...],
    depthwise_shapes: tuple[tuple[str, int, int, tuple[int, int, int]], ...],
    grn_shapes: tuple[tuple[int, tuple[int, int, int]], ...] = (),
    sm: tuple[int, int] | None = None,
) -> tuple[KernelCase, ...]:
    """Build the isolated operator cases discovered for one workload.

    The workload contributes only dtype choices.  Model context such as variant,
    checkpointing, and output classes is deliberately absent from each case key.
    """

    cases: list[KernelCase] = []
    for dtype in workload.dtypes:
        for batch in batches:
            for channels, spatial in sorted(set(grn_shapes)):
                cases.append(
                    _case(
                        family="global_response_norm3d",
                        direction="regular",
                        phase="training",
                        batch=batch,
                        spatial_shape=spatial,
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=None,
                        dtype=dtype,
                        implementation="triton_fused_grn",
                    )
                )
            for in_channels, out_channels, spatial in pointwise_shapes:
                cases.append(
                    _case(
                        family="pointwise_conv3d",
                        direction="regular",
                        phase="training",
                        batch=batch,
                        spatial_shape=spatial,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        dtype=dtype,
                        implementation="pointwise_gemm_per_sample",
                    )
                )
            for direction, channels, kernel_size, spatial in depthwise_shapes:
                if direction == "regular":
                    cases.append(
                        _case(
                            family="depthwise_conv3d",
                            direction=direction,
                            phase="backward_input",
                            batch=batch,
                            spatial_shape=spatial,
                            in_channels=channels,
                            out_channels=channels,
                            kernel_size=kernel_size,
                            dtype=dtype,
                            implementation="triton_depthwise_dx",
                            parameters=depthwise_parameters(
                                direction,
                                "backward_input",
                                batch,
                                spatial,
                                channels=channels,
                                kernel_size=kernel_size,
                                sm=sm,
                            ),
                        )
                    )
                    cases.append(
                        _case(
                            family="depthwise_conv3d",
                            direction=direction,
                            phase="backward_weight",
                            batch=batch,
                            spatial_shape=spatial,
                            in_channels=channels,
                            out_channels=channels,
                            kernel_size=kernel_size,
                            dtype=dtype,
                            implementation="triton_split_dw",
                            parameters=depthwise_parameters(
                                direction,
                                "backward_weight",
                                batch,
                                spatial,
                                channels=channels,
                                kernel_size=kernel_size,
                                sm=sm,
                            ),
                        )
                    )
                elif direction == "downsample":
                    cases.append(
                        _case(
                            family="depthwise_conv3d",
                            direction=direction,
                            phase="backward_input",
                            batch=batch,
                            spatial_shape=spatial,
                            in_channels=channels,
                            out_channels=channels,
                            kernel_size=kernel_size,
                            dtype=dtype,
                            implementation="triton_downsample_dx",
                            parameters=depthwise_parameters(
                                direction,
                                "backward_input",
                                batch,
                                spatial,
                                channels=channels,
                                kernel_size=kernel_size,
                                sm=sm,
                            ),
                        )
                    )
                elif direction == "transpose":
                    cases.append(
                        _case(
                            family="depthwise_conv_transpose3d",
                            direction=direction,
                            phase="backward_weight",
                            batch=batch,
                            spatial_shape=spatial,
                            in_channels=channels,
                            out_channels=channels,
                            kernel_size=kernel_size,
                            dtype=dtype,
                            implementation="triton_transpose_split_dw",
                            parameters=depthwise_parameters(
                                direction,
                                "backward_weight",
                                batch,
                                spatial,
                                channels=channels,
                                kernel_size=kernel_size,
                                sm=sm,
                            ),
                        )
                    )
                else:
                    raise ValueError(f"unsupported depthwise direction {direction!r}")
    return tuple(cases)


def deduplicate_cases(
    workload_cases: tuple[tuple[KernelCase, ...], ...],
) -> tuple[tuple[KernelCase, ...], tuple[tuple[KernelCaseKey, ...], ...]]:
    """Return sorted unique cases and each workload's references to their keys."""

    unique_by_key: dict[KernelCaseKey, KernelCase] = {}
    references: list[tuple[KernelCaseKey, ...]] = []
    for cases in workload_cases:
        keys: list[KernelCaseKey] = []
        for case in cases:
            unique_by_key.setdefault(case.key, case)
            keys.append(case.key)
        references.append(tuple(keys))
    unique = tuple(sorted(unique_by_key.values(), key=lambda case: case.key))
    return unique, tuple(references)


def group_cases(cases: tuple[KernelCase, ...]) -> tuple[KernelGroup, ...]:
    """Group cases by batch and a kernel probe category."""

    categories = {
        ("global_response_norm3d", "regular", "training"): "grn",
        ("pointwise_conv3d", "regular", "training"): "pointwise",
        ("depthwise_conv3d", "regular", "backward_input"): "regular-dx",
        ("depthwise_conv3d", "regular", "backward_weight"): "regular-dw",
        ("depthwise_conv3d", "downsample", "backward_input"): "downsample-dx",
        ("depthwise_conv_transpose3d", "transpose", "backward_weight"): "transpose-dw",
    }
    grouped: dict[tuple[int, str], list[KernelCase]] = {}
    for case in cases:
        key = case.key
        category = categories.get((key.family, key.direction, key.phase))
        if category is None:
            raise ValueError(
                f"unsupported kernel case category {key.family}/{key.direction}/{key.phase}"
            )
        grouped.setdefault((key.batch, category), []).append(case)
    return tuple(
        KernelGroup(
            category=category,
            batch=batch,
            cases=tuple(sorted(group, key=lambda case: case.key)),
        )
        for (batch, category), group in sorted(grouped.items())
    )
