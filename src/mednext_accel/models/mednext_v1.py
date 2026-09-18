"""Reference MedNeXt v1 architectures."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..checkpointing import CheckpointConfig
from .blocks import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock, OutputHead
from .config import MedNeXtV1Config, get_mednext_v1_config

DeepSupervisionOutput = Literal["tuple", "list", "stacked"]


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
    return MedNeXtV1(
        config,
        checkpointing=checkpointing,
        deep_supervision_output=deep_supervision_output,
        approximate_gelu_eval=approximate_gelu_eval,
    )


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
    )
