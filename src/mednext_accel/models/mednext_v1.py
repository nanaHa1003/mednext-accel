"""Reference MedNeXt v1 architectures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from os import PathLike
from typing import Literal, TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..checkpointing import CheckpointConfig
from ..ops.adaptive import (
    AdaptiveDepthwise3d,
    AdaptivePointwise3d,
    ModelOptimizationContext,
    execution_context_for_shape,
    install_adaptive_operators,
)
from ..optimization.policies import PolicyRegistry
from ..optimization.report import OptimizationReport
from .blocks import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock, OutputHead
from .config import MedNeXtV1Config, get_mednext_v1_config

DeepSupervisionOutput = Literal["tuple", "list", "stacked"]
OptimizationSource: TypeAlias = (
    Literal["auto", "reference"] | str | PathLike[str] | Mapping[str, object]
)


class MedNeXtV1(nn.Module):
    """Pure-PyTorch MedNeXt v1 with a stable native parameter schema."""

    def __init__(
        self,
        config: MedNeXtV1Config,
        *,
        checkpointing: CheckpointConfig | None = None,
        deep_supervision_output: DeepSupervisionOutput = "tuple",
        approximate_gelu_eval: bool = False,
    ) -> None:
        super().__init__()
        if deep_supervision_output not in ("tuple", "list", "stacked"):
            raise ValueError("deep_supervision_output must be 'tuple', 'list', or 'stacked'")
        self.config = config
        self.checkpointing = checkpointing
        self.deep_supervision_output = deep_supervision_output
        self.approximate_gelu_eval = approximate_gelu_eval
        self.optimization_source: OptimizationSource = "reference"

        conv = nn.Conv2d if config.spatial_dims == 2 else nn.Conv3d
        self.stem = conv(config.in_channels, config.base_channels, kernel_size=1)
        channels = tuple(config.base_channels * (2**stage) for stage in range(5))

        def checkpoint_expansion_at(stage: int) -> bool:
            return (
                checkpointing is not None
                and checkpointing.checkpoints_expansion
                and checkpointing.includes(stage)
            )

        self.encoder_stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            spatial_dims=config.spatial_dims,
                            in_channels=channels[stage],
                            out_channels=channels[stage],
                            expansion_ratio=config.expansion_ratios[stage],
                            kernel_size=config.kernel_size,
                            residual=config.residual_blocks,
                            checkpoint_expansion=checkpoint_expansion_at(stage),
                            approximate_gelu_eval=approximate_gelu_eval,
                        )
                        for _ in range(config.block_counts[stage])
                    ]
                )
                for stage in range(4)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                MedNeXtDownBlock(
                    spatial_dims=config.spatial_dims,
                    in_channels=channels[stage],
                    out_channels=channels[stage + 1],
                    expansion_ratio=config.downsample_expansion_ratios[stage],
                    kernel_size=config.kernel_size,
                    residual=config.residual_resampling,
                    checkpoint_expansion=checkpoint_expansion_at(stage + 1),
                    approximate_gelu_eval=approximate_gelu_eval,
                )
                for stage in range(4)
            ]
        )
        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    spatial_dims=config.spatial_dims,
                    in_channels=channels[4],
                    out_channels=channels[4],
                    expansion_ratio=config.expansion_ratios[4],
                    kernel_size=config.kernel_size,
                    residual=config.residual_blocks,
                    checkpoint_expansion=checkpoint_expansion_at(4),
                    approximate_gelu_eval=approximate_gelu_eval,
                )
                for _ in range(config.block_counts[4])
            ]
        )

        decoder_stages = (3, 2, 1, 0)
        decoder_config_indices = (5, 6, 7, 8)
        self.upsamples = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for stage, config_index in zip(decoder_stages, decoder_config_indices, strict=True):
            ratio = config.expansion_ratios[config_index]
            self.upsamples.append(
                MedNeXtUpBlock(
                    spatial_dims=config.spatial_dims,
                    in_channels=channels[stage + 1],
                    out_channels=channels[stage],
                    expansion_ratio=ratio,
                    kernel_size=config.kernel_size,
                    residual=config.residual_resampling,
                    checkpoint_expansion=checkpoint_expansion_at(stage),
                    approximate_gelu_eval=approximate_gelu_eval,
                )
            )
            self.decoder_stages.append(
                nn.Sequential(
                    *[
                        MedNeXtBlock(
                            spatial_dims=config.spatial_dims,
                            in_channels=channels[stage],
                            out_channels=channels[stage],
                            expansion_ratio=ratio,
                            kernel_size=config.kernel_size,
                            residual=config.residual_blocks,
                            checkpoint_expansion=checkpoint_expansion_at(stage),
                            approximate_gelu_eval=approximate_gelu_eval,
                        )
                        for _ in range(config.block_counts[config_index])
                    ]
                )
            )

        self.head = OutputHead(
            spatial_dims=config.spatial_dims,
            in_channels=channels[0],
            out_channels=config.out_channels,
        )
        self.deep_supervision_heads = nn.ModuleList()
        if config.deep_supervision:
            self.deep_supervision_heads.extend(
                OutputHead(
                    spatial_dims=config.spatial_dims,
                    in_channels=channels[stage],
                    out_channels=config.out_channels,
                )
                for stage in (4, 3, 2, 1)
            )

    def _forward_module(self, module: nn.Module, x: Tensor, stage: int) -> Tensor:
        config = self.checkpointing
        if (
            config is not None
            and config.checkpoints_blocks
            and config.includes(stage)
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def _forward_stage(self, modules: nn.Sequential, x: Tensor, stage: int) -> Tensor:
        for module in modules:
            x = self._forward_module(module, x, stage)
        return x

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, ...] | list[Tensor]:
        x = self.stem(x)
        skips: list[Tensor] = []
        for stage, (encoder, downsample) in enumerate(
            zip(self.encoder_stages, self.downsamples, strict=True)
        ):
            x = self._forward_stage(encoder, x, stage)
            skips.append(x)
            x = self._forward_module(downsample, x, stage + 1)

        x = self._forward_stage(self.bottleneck, x, 4)
        auxiliary: list[Tensor] = []
        for index, (upsample, decoder) in enumerate(
            zip(self.upsamples, self.decoder_stages, strict=True)
        ):
            stage = 3 - index
            if self.training and self.config.deep_supervision:
                auxiliary.append(self.deep_supervision_heads[index](x))
            x = self._forward_module(upsample, x, stage) + skips[-1 - index]
            x = self._forward_stage(decoder, x, stage)

        primary = self.head(x)
        if not self.training or not self.config.deep_supervision:
            return primary

        outputs = [primary, *reversed(auxiliary)]
        if self.deep_supervision_output == "tuple":
            return tuple(outputs)
        if self.deep_supervision_output == "list":
            return outputs
        resized = [
            output
            if output.shape[2:] == primary.shape[2:]
            else F.interpolate(output, size=primary.shape[2:])
            for output in outputs
        ]
        return torch.stack(resized, dim=1)

    def explain_optimization(
        self,
        *,
        input_shape: tuple[int, int, int, int, int],
        dtype: str | torch.dtype,
        device: str | torch.device,
    ) -> OptimizationReport:
        """Resolve and report implementations without executing the model."""

        resolver = self.__dict__.get("_optimization_resolver")
        model_context = ModelOptimizationContext(
            "mednext_v1",
            self.config.variant,
            _checkpoint_name(self.checkpointing),
            self.approximate_gelu_eval,
        )
        target = torch.device(device)
        sm = None
        total_vram = 0
        if target.type == "cuda" and torch.cuda.is_available():
            sm = torch.cuda.get_device_capability(target)
            total_vram = torch.cuda.get_device_properties(target).total_memory
        context = execution_context_for_shape(
            model_context,
            batch_size=input_shape[0],
            spatial_shape=input_shape[2:],
            dtype=dtype,
            device_type=target.type,
            sm=sm,
            total_vram_bytes=total_vram,
            training=self.training,
        )
        if resolver is None:
            return OptimizationReport(("reference",), context, ())
        decisions = []
        for module in self.modules():
            module_context = replace(
                context,
                spatial_shape=_spatial_shape_for_role(module.descriptor.role, context.spatial_shape)
                if isinstance(module, (AdaptivePointwise3d, AdaptiveDepthwise3d))
                else context.spatial_shape,
            )
            if isinstance(module, AdaptivePointwise3d):
                decisions.append(resolver.resolve(module.descriptor, module_context, context.phase))
            elif isinstance(module, AdaptiveDepthwise3d):
                decisions.extend(
                    resolver.resolve(module.descriptor, module_context, phase)
                    for phase in ("backward_input", "backward_weight")
                )
        report_warnings = tuple(
            dict.fromkeys(
                decision.warning for decision in decisions if decision.warning is not None
            )
        )
        return OptimizationReport(
            tuple(policy.name for policy in resolver.context_layers(context)) or ("reference",),
            context,
            tuple(decisions),
            report_warnings,
        )


def _checkpoint_name(config: CheckpointConfig | None) -> str:
    if config is None:
        return "none"
    if config.stages is None:
        return "all-expansion" if config.style == "expansion" else "whole-block"
    stages = ",".join(map(str, config.stages))
    return f"{config.style}:{stages}"


def _spatial_shape_for_role(
    role: str | None, input_spatial: tuple[int, int, int]
) -> tuple[int, int, int]:
    if role is None or role == "stem" or role.startswith("head"):
        return input_spatial
    parts = role.split(".")
    reductions = 0
    if parts[0] == "encoder_stages":
        reductions = int(parts[1])
    elif parts[0] == "downsamples":
        reductions = int(parts[1]) + (0 if parts[-1] == "depthwise" else 1)
    elif parts[0] == "bottleneck":
        reductions = 4
    elif parts[0] == "upsamples":
        reductions = 4 - int(parts[1])
    elif parts[0] == "decoder_stages":
        reductions = 3 - int(parts[1])
    spatial = input_spatial
    for _ in range(reductions):
        spatial = tuple((size + 1) // 2 for size in spatial)
    if parts[0] == "upsamples" and parts[-1] != "depthwise":
        spatial = tuple(2 * size - 1 for size in spatial)
    return spatial


def _factory(
    variant: str,
    *,
    in_channels: int,
    out_channels: int,
    spatial_dims: Literal[2, 3] = 3,
    kernel_size: int = 3,
    base_channels: int = 32,
    deep_supervision: bool = False,
    checkpointing: CheckpointConfig | None = None,
    deep_supervision_output: DeepSupervisionOutput = "tuple",
    approximate_gelu_eval: bool = False,
    optimization: OptimizationSource = "auto",
) -> MedNeXtV1:
    config = get_mednext_v1_config(
        variant,
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
    )
    model = MedNeXtV1(
        config,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
    )
    return _configure_optimization(model, optimization)


def _configure_optimization(model: MedNeXtV1, optimization: OptimizationSource) -> MedNeXtV1:
    model.optimization_source = optimization
    if optimization == "reference":
        return model
    if isinstance(optimization, str) and optimization in (
        "torch",
        "conservative",
        "autotune",
    ):
        raise ValueError(f"optimization={optimization!r} was removed; use 'auto' or 'reference'")
    external = None if optimization == "auto" else optimization
    resolver = PolicyRegistry(external=external).resolver()
    model_context = ModelOptimizationContext(
        "mednext_v1",
        model.config.variant,
        _checkpoint_name(model.checkpointing),
        model.approximate_gelu_eval,
    )
    install_adaptive_operators(model, resolver=resolver, model_context=model_context)
    model.__dict__["_optimization_resolver"] = resolver
    model.__dict__["_optimization_model_context"] = model_context
    return model


def mednext_small(
    *,
    in_channels: int,
    out_channels: int,
    spatial_dims: Literal[2, 3] = 3,
    kernel_size: int = 3,
    base_channels: int = 32,
    deep_supervision: bool = False,
    checkpointing: CheckpointConfig | None = None,
    deep_supervision_output: DeepSupervisionOutput = "tuple",
    approximate_gelu_eval: bool = False,
    optimization: OptimizationSource = "auto",
) -> MedNeXtV1:
    """Build the published MedNeXt Small architecture."""

    return _factory(
        "small",
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
        optimization=optimization,
    )


def mednext_base(
    *,
    in_channels: int,
    out_channels: int,
    spatial_dims: Literal[2, 3] = 3,
    kernel_size: int = 3,
    base_channels: int = 32,
    deep_supervision: bool = False,
    checkpointing: CheckpointConfig | None = None,
    deep_supervision_output: DeepSupervisionOutput = "tuple",
    approximate_gelu_eval: bool = False,
    optimization: OptimizationSource = "auto",
) -> MedNeXtV1:
    """Build the published MedNeXt Base architecture."""

    return _factory(
        "base",
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
        optimization=optimization,
    )


def mednext_medium(
    *,
    in_channels: int,
    out_channels: int,
    spatial_dims: Literal[2, 3] = 3,
    kernel_size: int = 3,
    base_channels: int = 32,
    deep_supervision: bool = False,
    checkpointing: CheckpointConfig | None = None,
    deep_supervision_output: DeepSupervisionOutput = "tuple",
    approximate_gelu_eval: bool = False,
    optimization: OptimizationSource = "auto",
) -> MedNeXtV1:
    """Build the published MedNeXt Medium architecture."""

    return _factory(
        "medium",
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
        optimization=optimization,
    )


def mednext_large(
    *,
    in_channels: int,
    out_channels: int,
    spatial_dims: Literal[2, 3] = 3,
    kernel_size: int = 3,
    base_channels: int = 32,
    deep_supervision: bool = False,
    checkpointing: CheckpointConfig | None = None,
    deep_supervision_output: DeepSupervisionOutput = "tuple",
    approximate_gelu_eval: bool = False,
    optimization: OptimizationSource = "auto",
) -> MedNeXtV1:
    """Build the published MedNeXt Large architecture."""

    return _factory(
        "large",
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=spatial_dims,
        kernel_size=kernel_size,
        base_channels=base_channels,
        deep_supervision=deep_supervision,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
        optimization=optimization,
    )
