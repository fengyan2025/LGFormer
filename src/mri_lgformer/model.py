"""Standalone MRI-LGFormer-T1 model.

The default ``proposed`` allocation matches the historical A1 implementation:
high-resolution local stages, intermediate mixed stages, and low-resolution
transposed-channel-attention stages.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .blocks import (
    Downsample,
    OverlapPatchEmbed,
    PlainSkipFusion,
    Upsample,
    make_global_stage,
    make_local_stage,
    make_mixed_stage,
)


class MRILGFormerT1(nn.Module):
    """Stage-specialized local-global Transformer for same-grid T1w MRI."""

    def __init__(
        self,
        inp_channels: int = 1,
        out_channels: int = 1,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        dual_pixel_task: bool = False,
        allocation: str = "proposed",
    ) -> None:
        super().__init__()
        if len(num_blocks) != 4 or len(heads) != 4:
            raise ValueError("num_blocks and heads must contain four stage values")
        if allocation not in {"proposed", "all_local", "reversed"}:
            raise ValueError(f"Unsupported allocation={allocation!r}")
        self.allocation = allocation
        self.dual_pixel_task = bool(dual_pixel_task)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)

        def local(channels: int, depth: int, _heads: int) -> nn.Sequential:
            return make_local_stage(channels, depth, bias, LayerNorm_type)

        def global_stage(
            channels: int,
            depth: int,
            stage_heads: int,
        ) -> nn.Sequential:
            return make_global_stage(
                channels,
                depth,
                stage_heads,
                ffn_expansion_factor,
                bias,
                LayerNorm_type,
            )

        def mixed(
            channels: int,
            depth: int,
            stage_heads: int,
        ) -> nn.Sequential:
            return make_mixed_stage(
                channels,
                depth,
                stage_heads,
                ffn_expansion_factor,
                bias,
                LayerNorm_type,
            )

        if allocation == "proposed":
            stage_builders = (local, mixed, global_stage, global_stage)
            refinement_builder = local
        elif allocation == "all_local":
            stage_builders = (local, local, local, local)
            refinement_builder = local
        else:
            stage_builders = (global_stage, mixed, local, local)
            refinement_builder = global_stage

        self.encoder_level1 = stage_builders[0](dim, num_blocks[0], heads[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = stage_builders[1](
            dim * 2,
            num_blocks[1],
            heads[1],
        )
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = stage_builders[2](
            dim * 4,
            num_blocks[2],
            heads[2],
        )
        self.down3_4 = Downsample(dim * 4)
        self.latent = stage_builders[3](
            dim * 8,
            num_blocks[3],
            heads[3],
        )

        self.up4_3 = Upsample(dim * 8)
        self.skip_level3 = PlainSkipFusion(dim * 4, bias)
        self.decoder_level3 = stage_builders[2](
            dim * 4,
            num_blocks[2],
            heads[2],
        )
        self.up3_2 = Upsample(dim * 4)
        self.skip_level2 = PlainSkipFusion(dim * 2, bias)
        self.decoder_level2 = stage_builders[1](
            dim * 2,
            num_blocks[1],
            heads[1],
        )
        self.up2_1 = Upsample(dim * 2)
        self.skip_level1 = PlainSkipFusion(dim, bias)
        self.decoder_level1 = stage_builders[0](
            dim,
            num_blocks[0],
            heads[0],
        )
        self.refinement = refinement_builder(
            dim,
            num_refinement_blocks,
            heads[0],
        )
        self.output = nn.Conv2d(
            dim,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.input_skip: nn.Module
        if inp_channels == out_channels:
            self.input_skip = nn.Identity()
        else:
            self.input_skip = nn.Conv2d(
                inp_channels,
                out_channels,
                1,
                bias=bias,
            )

    def forward_features(self, inp_img: torch.Tensor) -> torch.Tensor:
        if inp_img.ndim != 4:
            raise ValueError(f"Expected BCHW input, got {tuple(inp_img.shape)}")
        if inp_img.shape[-2] % 8 or inp_img.shape[-1] % 8:
            raise ValueError(
                "Input height and width must be divisible by 8, got "
                f"{tuple(inp_img.shape[-2:])}"
            )
        enc1 = self.encoder_level1(self.patch_embed(inp_img))
        enc2 = self.encoder_level2(self.down1_2(enc1))
        enc3 = self.encoder_level3(self.down2_3(enc2))
        latent = self.latent(self.down3_4(enc3))
        dec3 = self.decoder_level3(
            self.skip_level3(enc3, self.up4_3(latent))
        )
        dec2 = self.decoder_level2(
            self.skip_level2(enc2, self.up3_2(dec3))
        )
        dec1 = self.decoder_level1(
            self.skip_level1(enc1, self.up2_1(dec2))
        )
        return self.refinement(dec1)

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        residual = self.output(self.forward_features(inp_img))
        if self.dual_pixel_task:
            return residual
        return self.input_skip(inp_img) + residual


def build_model(config: dict | None = None) -> MRILGFormerT1:
    return MRILGFormerT1(**(config or {}))
