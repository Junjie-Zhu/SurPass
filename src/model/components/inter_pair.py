"""Quadratic rectangular pair-map updates.

These blocks use shared row/column depthwise convolutions. They are O(L1*L2)
and transpose-equivariant, unlike cubic axial attention over a square map.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.model.components.transformer import SwiGLU


class SharedAxisConvolution(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        dilation = int(dilation)
        padding = dilation * (kernel_size // 2)
        self.convolution = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            dilation=(1, dilation),
            groups=channels,
            bias=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        row = self.convolution(value)
        column = self.convolution(value.transpose(-2, -1)).transpose(-2, -1)
        return row + column


class InterPairBlock(nn.Module):
    def __init__(
        self,
        dim_pair: int,
        *,
        expansion_factor: int = 2,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        dim_pair = int(dim_pair)
        hidden = dim_pair * int(expansion_factor)
        self.norm = nn.LayerNorm(dim_pair)
        self.axis_convolution = SharedAxisConvolution(
            dim_pair,
            kernel_size=3,
            dilation=int(dilation),
        )
        self.axis_projection = nn.Conv2d(dim_pair, dim_pair, kernel_size=1)
        self.transition_norm = nn.LayerNorm(dim_pair)
        self.transition_in = nn.Sequential(
            nn.Linear(dim_pair, hidden * 2),
            SwiGLU(),
        )
        self.transition_out = nn.Linear(hidden * 2, dim_pair)
        self.dropout = float(dropout)
        nn.init.zeros_(self.axis_projection.weight)
        nn.init.zeros_(self.axis_projection.bias)
        nn.init.zeros_(self.transition_out.weight)
        nn.init.zeros_(self.transition_out.bias)

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = pair_mask.to(dtype=pair.dtype)[..., None]
        normalized = self.norm(pair) * mask
        channels_first = normalized.permute(0, 3, 1, 2)
        update = self.axis_projection(self.axis_convolution(channels_first))
        update = update.permute(0, 2, 3, 1)
        if self.training and self.dropout:
            update = F.dropout(update, p=self.dropout)
        pair = pair + update * mask

        transition_value = self.transition_in(self.transition_norm(pair))
        transition = self.transition_out(transition_value)
        if self.training and self.dropout:
            transition = F.dropout(transition, p=self.dropout)
        return (pair + transition * mask) * mask
