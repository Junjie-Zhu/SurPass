"""Small fused-attention blocks for the two protein chains."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _attention_mask(key_mask: torch.Tensor, query_length: int) -> torch.Tensor | None:
    key_mask = key_mask.to(dtype=torch.bool)
    if bool(key_mask.all()):
        return None
    return key_mask[:, None, None, :].expand(-1, 1, int(query_length), -1)


class GatedAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        *,
        dropout: float = 0.0,
        cross_attention: bool = False,
    ):
        super().__init__()
        dimension = int(dimension)
        heads = int(heads)
        if dimension % heads:
            raise ValueError(f"heads ({heads}) must divide dimension ({dimension})")
        self.dimension = dimension
        self.heads = heads
        self.head_dimension = dimension // heads
        self.dropout = float(dropout)
        self.cross_attention = bool(cross_attention)
        self.query_norm = nn.LayerNorm(dimension)
        self.context_norm = nn.LayerNorm(dimension)
        self.query_projection = nn.Linear(dimension, dimension, bias=False)
        self.key_value_projection = nn.Linear(dimension, 2 * dimension, bias=False)
        self.gate_projection = nn.Linear(dimension, dimension)
        self.output_projection = nn.Linear(dimension, dimension)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_length, _ = query.shape
        context_length = int(context.shape[1])
        normalized_query = self.query_norm(query)
        normalized_context = self.context_norm(context)
        q = self.query_projection(normalized_query)
        k, v = self.key_value_projection(normalized_context).chunk(2, dim=-1)
        gate = self.gate_projection(normalized_query).sigmoid()

        q = self.split_heads(q, query_length)
        k = self.split_heads(k, context_length)
        v = self.split_heads(v, context_length)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=_attention_mask(context_mask, query_length),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, query_length, self.dimension)
        update = self.output_projection(attended * gate)
        return update * query_mask.to(dtype=update.dtype)[..., None]

    @staticmethod
    def split_heads(value: torch.Tensor, length: int) -> torch.Tensor:
        return value.view(
            batch,
            length,
            self.heads,
            self.head_dimension,
        ).transpose(1, 2)


class SwiGLU(torch.nn.Module):
    """SwiGLU layer."""

    def forward(self, x):
        """
        Args:
            x: input tensor, shape [..., d]

        Returns:
            Tensor of shape [..., d//2].
        """
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


class Transition(nn.Module):
    def __init__(self, dimension: int, expansion: int = 2):
        super().__init__()
        hidden = int(dimension) * int(expansion)
        self.norm = nn.LayerNorm(int(dimension))
        self.input_projection = nn.Sequential(
            nn.Linear(int(dimension), hidden * 2),
            SwiGLU(),
        )
        self.output_projection = nn.Linear(hidden * 2, int(dimension))
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = self.input_projection(self.norm(value))
        update = self.output_projection(value)
        return update * mask.to(dtype=update.dtype)[..., None]


class ResidueTransformer(nn.Module):
    """Shared per-chain self-attention block."""

    def __init__(
        self,
        dim_token: int,
        nheads: int,
        *,
        dropout: float = 0.0,
        expansion_factor: int = 2,
        **_: object,
    ):
        super().__init__()
        self.attention = GatedAttention(
            int(dim_token),
            int(nheads),
            dropout=dropout,
        )
        self.transition = Transition(int(dim_token), int(expansion_factor))

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = value + self.attention(value, value, mask, mask)
        value = value + self.transition(value, mask)
        return value * mask.to(dtype=value.dtype)[..., None]


class BidirectionalCrossAttention(nn.Module):
    """One shared cross-attention update, equivariant to swapping the chains."""

    def __init__(
        self,
        dim_token: int,
        nheads: int,
        *,
        dropout: float = 0.0,
        expansion_factor: int = 2,
    ):
        super().__init__()
        self.attention = GatedAttention(
            int(dim_token),
            int(nheads),
            dropout=dropout,
            cross_attention=True,
        )
        self.transition = Transition(int(dim_token), int(expansion_factor))

    def forward(
        self,
        chain_1: torch.Tensor,
        chain_2: torch.Tensor,
        mask_1: torch.Tensor,
        mask_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update_1 = self.attention(chain_1, chain_2, mask_1, mask_2)
        update_2 = self.attention(chain_2, chain_1, mask_2, mask_1)
        chain_1 = chain_1 + update_1
        chain_2 = chain_2 + update_2
        chain_1 = chain_1 + self.transition(chain_1, mask_1)
        chain_2 = chain_2 + self.transition(chain_2, mask_2)
        return (
            chain_1 * mask_1.to(dtype=chain_1.dtype)[..., None],
            chain_2 * mask_2.to(dtype=chain_2.dtype)[..., None],
        )
