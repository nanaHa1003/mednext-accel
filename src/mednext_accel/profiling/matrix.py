"""Context-free kernel cases used by the profiling campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

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
    kernel_size: int
    dtype: str
    implementation: str
    parameters: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class KernelCase:
    key: KernelCaseKey

    @property
    def identifier(self) -> str:
        document = json.dumps(asdict(self.key), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(document.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class KernelGroup:
    category: str
    batch: int
    cases: tuple[KernelCase, ...]


def depthwise_parameters(
    direction: str, phase: str, batch: int, spatial: tuple[int, int, int]
) -> tuple[tuple[str, int], ...]:
    """Default launch parameters for planning and legacy single-case requests."""
    if direction in ("regular", "downsample") and phase == "backward_input":
        return (("dx_block", 128),)
    if direction in ("regular", "transpose") and phase == "backward_weight":
        anchor = 128 if direction == "regular" else 64
        splits = max(1, min(512, round(64 * batch * spatial[0] ** 3 / anchor**3)))
        return (("dw_splits", splits), ("dw_block", 512))
    raise ValueError(f"unsupported depthwise probe {direction}/{phase}")


def _case(
    *,
    family: str,
    direction: str,
    phase: str,
    batch: int,
    spatial_shape: tuple[int, int, int],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
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
) -> tuple[KernelCase, ...]:
    """Build the isolated operator cases discovered for one workload.

    The workload contributes only dtype choices.  Model context such as variant,
    checkpointing, and output classes is deliberately absent from each case key.
    """

    cases: list[KernelCase] = []
    for dtype in workload.dtypes:
        for batch in batches:
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
                                direction, "backward_input", batch, spatial
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
                                direction, "backward_weight", batch, spatial
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
                                direction, "backward_input", batch, spatial
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
                                direction, "backward_weight", batch, spatial
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
    """Group cases by batch and one of the five kernel probe categories."""

    categories = {
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
