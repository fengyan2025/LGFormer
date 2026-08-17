"""Building blocks for the standalone MRI-LGFormer-T1 implementation.

The transposed-channel attention and gated depthwise feed-forward design are
adapted from Restormer. MRI-LGFormer-T1 adds the MRI Local Detail Block (MLDB)
and uses the blocks in a resolution-specialized stage allocation.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def _to_tokens(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def _to_image(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | Sequence[int]) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        shape = torch.Size(normalized_shape)
        if len(shape) != 1:
            raise ValueError(f"Expected one normalized dimension, got {shape}")
        self.weight = nn.Parameter(torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | Sequence[int]) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        shape = torch.Size(normalized_shape)
        if len(shape) != 1:
            raise ValueError(f"Expected one normalized dimension, got {shape}")
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str) -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        elif layer_norm_type == "WithBias":
            self.body = WithBiasLayerNorm(dim)
        else:
            raise ValueError(
                "layer_norm_type must be 'BiasFree' or 'WithBias', "
                f"got {layer_norm_type!r}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return _to_image(self.body(_to_tokens(x)), height, width)


class GatedDconvFeedForward(nn.Module):
    """Restormer-style gated depthwise-convolution feed-forward network."""

    def __init__(self, dim: int, expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            stride=1,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class MRILocalDetailBlock(nn.Module):
    """MRI-specific local block with isotropic and directional depthwise paths."""

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 2.0,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
    ) -> None:
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.norm = LayerNorm(dim, layer_norm_type)
        self.expand = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.local_3x3 = nn.Conv2d(
            hidden,
            hidden,
            3,
            stride=1,
            padding=1,
            groups=hidden,
            bias=bias,
        )
        self.local_horizontal = nn.Conv2d(
            hidden,
            hidden,
            (1, 5),
            stride=1,
            padding=(0, 2),
            groups=hidden,
            bias=bias,
        )
        self.local_vertical = nn.Conv2d(
            hidden,
            hidden,
            (5, 1),
            stride=1,
            padding=(2, 0),
            groups=hidden,
            bias=bias,
        )
        self.project = nn.Conv2d(hidden, dim, 1, bias=bias)
        # ReZero starts each block as an identity mapping.
        self.residual_scale = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        local_input, directional_input = self.expand(self.norm(x)).chunk(2, dim=1)
        local = self.local_3x3(local_input)
        directional = self.local_horizontal(directional_input)
        directional = directional + self.local_vertical(directional_input)
        gated = F.gelu(local) * torch.sigmoid(directional)
        return residual + self.residual_scale * self.project(gated)


class LocalOnlyTransposedAttention(nn.Module):
    """Transposed channel attention with local 3x3 QKV encoding."""

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_local = nn.Conv2d(
            dim * 3,
            dim * 3,
            3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        q, k, v = self.qkv_local(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = attention @ v
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=height,
            w=width,
        )
        return self.project_out(out)


class TransposedAttentionBlock(nn.Module):
    """Local-QKV transposed channel attention followed by GDFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attention = LocalOnlyTransposedAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.feed_forward = GatedDconvFeedForward(
            dim,
            ffn_expansion_factor,
            bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class PlainSkipFusion(nn.Module):
    """Concatenate same-scale features and reduce channels with a 1x1 conv."""

    def __init__(self, channels: int, bias: bool = False) -> None:
        super().__init__()
        self.projection = nn.Conv2d(channels * 2, channels, 1, bias=bias)

    def forward(
        self,
        encoder_features: torch.Tensor,
        decoder_features: torch.Tensor,
    ) -> torch.Tensor:
        if encoder_features.shape != decoder_features.shape:
            raise ValueError(
                "PlainSkipFusion inputs must have identical shapes, got "
                f"{tuple(encoder_features.shape)} and {tuple(decoder_features.shape)}"
            )
        return self.projection(
            torch.cat([decoder_features, encoder_features], dim=1)
        )


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, bias: bool) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels // 2,
                3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 2,
                3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def make_local_stage(
    dim: int,
    depth: int,
    bias: bool,
    layer_norm_type: str,
) -> nn.Sequential:
    return nn.Sequential(
        *[
            MRILocalDetailBlock(
                dim=dim,
                expansion_factor=2.0,
                bias=bias,
                layer_norm_type=layer_norm_type,
            )
            for _ in range(depth)
        ]
    )


def make_global_stage(
    dim: int,
    depth: int,
    num_heads: int,
    ffn_expansion_factor: float,
    bias: bool,
    layer_norm_type: str,
) -> nn.Sequential:
    return nn.Sequential(
        *[
            TransposedAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                layer_norm_type=layer_norm_type,
            )
            for _ in range(depth)
        ]
    )


def make_mixed_stage(
    dim: int,
    depth: int,
    num_heads: int,
    ffn_expansion_factor: float,
    bias: bool,
    layer_norm_type: str,
) -> nn.Sequential:
    local_depth = depth // 2
    global_depth = depth - local_depth
    blocks: list[nn.Module] = [
        MRILocalDetailBlock(
            dim=dim,
            expansion_factor=2.0,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )
        for _ in range(local_depth)
    ]
    blocks.extend(
        TransposedAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )
        for _ in range(global_depth)
    )
    return nn.Sequential(*blocks)
