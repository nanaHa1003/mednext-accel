"""Reference MedNeXt v2 architectures with global response normalization."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..checkpointing import CheckpointConfig
from ..ops.adaptive import (
    AdaptiveDepthwise3d,
    AdaptivePointwise3d,
    ModelOptimizationContext,
    _conv_descriptor,
    execution_context_for_shape,
    install_adaptive_operators,
)
from ..ops.grn import AdaptiveGlobalResponseNorm3d, GlobalResponseNorm3d, install_adaptive_grn
from ..optimization.descriptors import OperatorDescriptor
from ..optimization.policies import PolicyRegistry
from ..optimization.policy_resolver import PolicyDecision
from ..optimization.report import OptimizationReport
from .blocks import OutputHead
from .blocks_v2 import MedNeXtV2Block, MedNeXtV2DownBlock, MedNeXtV2UpBlock
from .config_v2 import MedNeXtV2Config, get_mednext_v2_config
from .mednext_v1 import DeepSupervisionOutput, OptimizationSource, _checkpoint_name


def _reconcile_spatial(x: Tensor, target_shape: tuple[int, ...]) -> Tensor:
    """Crop or zero-pad the right edges to match an encoder skip."""

    x = x[..., : target_shape[0], : target_shape[1], : target_shape[2]]
    deficits = [max(want - have, 0) for have, want in zip(x.shape[2:], target_shape, strict=True)]
    return F.pad(x, (0, deficits[2], 0, deficits[1], 0, deficits[0]))


class MedNeXtV2(nn.Module):
    """Five-resolution 3D MedNeXt v2 with exact GELU and GRN blocks."""

    def __init__(
        self,
        config: MedNeXtV2Config,
        *,
        checkpointing: CheckpointConfig | None = None,
        deep_supervision_output: DeepSupervisionOutput = "tuple",
    ) -> None:
        super().__init__()
        if deep_supervision_output not in ("tuple", "list", "stacked"):
            raise ValueError("deep_supervision_output must be 'tuple', 'list', or 'stacked'")
        self.config = config
        self.checkpointing = checkpointing
        self.deep_supervision_output = deep_supervision_output
        self.optimization_source: OptimizationSource = "reference"
        channels = tuple(config.base_channels * 2**stage for stage in range(5))
        self.stem = nn.Conv3d(config.in_channels, channels[0], kernel_size=1)

        def checkpoint_expansion_at(stage: int) -> bool:
            return (
                checkpointing is not None
                and checkpointing.checkpoints_expansion
                and checkpointing.includes(stage)
            )

        def make_stage(stage: int, config_index: int) -> nn.Sequential:
            return nn.Sequential(
                *[
                    MedNeXtV2Block(
                        in_channels=channels[stage],
                        out_channels=channels[stage],
                        expansion_ratio=config.expansion_ratios[config_index],
                        checkpoint_expansion=checkpoint_expansion_at(stage),
                    )
                    for _ in range(config.block_counts[config_index])
                ]
            )

        self.encoder_stages = nn.ModuleList(make_stage(stage, stage) for stage in range(4))
        self.downsamples = nn.ModuleList(
            MedNeXtV2DownBlock(
                in_channels=channels[stage],
                out_channels=channels[stage + 1],
                expansion_ratio=config.downsample_expansion_ratios[stage],
                checkpoint_expansion=checkpoint_expansion_at(stage + 1),
            )
            for stage in range(4)
        )
        self.bottleneck = make_stage(4, 4)
        self.upsamples = nn.ModuleList(
            MedNeXtV2UpBlock(
                in_channels=channels[stage + 1],
                out_channels=channels[stage],
                expansion_ratio=config.upsample_expansion_ratios[index],
                checkpoint_expansion=checkpoint_expansion_at(stage),
            )
            for index, stage in enumerate((3, 2, 1, 0))
        )
        self.decoder_stages = nn.ModuleList(
            make_stage(stage, index + 5) for index, stage in enumerate((3, 2, 1, 0))
        )
        self.head = OutputHead(
            spatial_dims=3, in_channels=channels[0], out_channels=config.out_channels
        )
        self.deep_supervision_heads = nn.ModuleList()
        if config.deep_supervision:
            self.deep_supervision_heads.extend(
                OutputHead(
                    spatial_dims=3, in_channels=channels[stage], out_channels=config.out_channels
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
        # Four ceil-divide-by-two reductions leave one voxel when every extent
        # is at most 16. PyTorch GroupNorm rejects this only for batch size one.
        if x.shape[0] == 1 and all((size + 15) // 16 == 1 for size in x.shape[2:]):
            raise ValueError(
                "GroupNorm requires more than one value per channel for batch size one; "
                "the spatial volume after four downsamplings must exceed one"
            )
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
            skip = skips[-1 - index]
            x = _reconcile_spatial(self._forward_module(upsample, x, stage), skip.shape[2:])
            x = self._forward_stage(decoder, x + skip, stage)

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
        input_shape: tuple[int, ...],
        dtype: str | torch.dtype,
        device: str | torch.device,
    ) -> OptimizationReport:
        """Resolve each operator at its input resolution without running the model."""

        if len(input_shape) != 5:
            raise ValueError("input_shape must have rank 5 for a 3D model")
        resolver = self.__dict__.get("_optimization_resolver")
        model_context = ModelOptimizationContext(
            "mednext_v2", self.config.variant, _checkpoint_name(self.checkpointing)
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
        decisions = []
        for role, module in self.named_modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, GlobalResponseNorm3d)):
                descriptor = (
                    OperatorDescriptor.normalization(
                        family="global_response_norm3d", channels=module.gamma.shape[1], role=role
                    )
                    if isinstance(module, GlobalResponseNorm3d)
                    else _conv_descriptor(module, role=role)
                )
                decisions.append(
                    PolicyDecision(
                        descriptor,
                        context.phase,
                        "reference",
                        {},
                        "reference",
                        "unwrapped",
                        "guard",
                        disposition="unwrapped",
                    )
                )
                continue
            if not isinstance(
                module, (AdaptivePointwise3d, AdaptiveDepthwise3d, AdaptiveGlobalResponseNorm3d)
            ):
                continue
            module_context = replace(
                context, spatial_shape=_spatial_shape_for_role(role, context.spatial_shape)
            )
            if isinstance(module, AdaptiveGlobalResponseNorm3d) and not module.training:
                decisions.append(
                    PolicyDecision(
                        module.descriptor,
                        "inference",
                        "reference",
                        {},
                        "reference",
                        "context-guard",
                        "guard",
                        guard_reason="GRN evaluation uses reference",
                    )
                )
            elif isinstance(module, AdaptiveDepthwise3d):
                decisions.extend(
                    resolver.resolve(module.descriptor, module_context, phase)
                    for phase in ("backward_input", "backward_weight")
                )
            else:
                phase = (
                    "training"
                    if isinstance(module, AdaptiveGlobalResponseNorm3d)
                    else context.phase
                )
                decisions.append(resolver.resolve(module.descriptor, module_context, phase))
        report_warnings = tuple(
            dict.fromkeys(
                decision.warning for decision in decisions if decision.warning is not None
            )
        )
        return OptimizationReport(
            (tuple(policy.name for policy in resolver.context_layers(context)) or ("reference",))
            if resolver is not None
            else ("reference",),
            context,
            tuple(decisions),
            report_warnings,
        )


def _spatial_shape_for_role(role: str, input_spatial: tuple[int, ...]) -> tuple[int, ...]:
    parts = role.split(".")
    reductions = 0
    if parts[0] == "encoder_stages":
        reductions = int(parts[1])
    elif parts[0] == "downsamples":
        reductions = int(parts[1]) + (parts[-1] not in ("depthwise", "residual"))
    elif parts[0] == "bottleneck":
        reductions = 4
    elif parts[0] in ("upsamples", "deep_supervision_heads"):
        reductions = 4 - int(parts[1])
    elif parts[0] == "decoder_stages":
        reductions = 3 - int(parts[1])
    spatial = input_spatial
    for _ in range(reductions):
        spatial = tuple((size + 1) // 2 for size in spatial)
    if parts[0] == "upsamples" and parts[-1] not in ("depthwise", "residual"):
        # Expansion and GRN precede the up block's right-edge padding.
        spatial = tuple(2 * size - 1 for size in spatial)
    return spatial


def _configure_optimization(model: MedNeXtV2, optimization: OptimizationSource) -> MedNeXtV2:
    model.optimization_source = optimization
    if optimization == "reference":
        return model
    if isinstance(optimization, str) and optimization in ("torch", "conservative", "autotune"):
        raise ValueError(f"optimization={optimization!r} was removed; use 'auto' or 'reference'")
    external = None if optimization == "auto" else optimization
    resolver = PolicyRegistry(external=external).resolver()
    model_context = ModelOptimizationContext(
        "mednext_v2", model.config.variant, _checkpoint_name(model.checkpointing)
    )
    install_adaptive_operators(model, resolver=resolver, model_context=model_context)
    install_adaptive_grn(model, resolver=resolver, model_context=model_context)
    model.__dict__["_optimization_resolver"] = resolver
    model.__dict__["_optimization_model_context"] = model_context
    return model


def _factory(
    variant: str,
    *,
    in_channels: int,
    out_channels: int,
    deep_supervision: bool,
    deep_supervision_output: DeepSupervisionOutput,
    checkpointing: CheckpointConfig | None,
    optimization: OptimizationSource,
) -> MedNeXtV2:
    model = MedNeXtV2(
        get_mednext_v2_config(
            variant,
            in_channels=in_channels,
            out_channels=out_channels,
            deep_supervision=deep_supervision,
        ),
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
    )
    return _configure_optimization(model, optimization)


def mednext_v2_base(
    *,
    in_channels: int,
    out_channels: int,
    deep_supervision: bool = False,
    deep_supervision_output: DeepSupervisionOutput = "tuple",
    checkpointing: CheckpointConfig | None = None,
    optimization: OptimizationSource = "auto",
) -> MedNeXtV2:
    """Build published MedNeXt v2 Base, with policy-driven training operators."""

    return _factory(
        "base",
        in_channels=in_channels,
        out_channels=out_channels,
        deep_supervision=deep_supervision,
        deep_supervision_output=deep_supervision_output,
        checkpointing=checkpointing,
        optimization=optimization,
    )


def mednext_v2_wide(
    *,
    in_channels: int,
    out_channels: int,
    deep_supervision: bool = False,
    deep_supervision_output: DeepSupervisionOutput = "tuple",
    checkpointing: CheckpointConfig | None = None,
    optimization: OptimizationSource = "auto",
) -> MedNeXtV2:
    """Build published MedNeXt v2 Wide, with policy-driven training operators."""

    return _factory(
        "wide",
        in_channels=in_channels,
        out_channels=out_channels,
        deep_supervision=deep_supervision,
        deep_supervision_output=deep_supervision_output,
        checkpointing=checkpointing,
        optimization=optimization,
    )
