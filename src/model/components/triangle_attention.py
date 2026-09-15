from functools import partialmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm

from src.model.components.triangle_update import permute_final_dims


class BiasGatedAttention(nn.Module):
    """Multi-head attention with additive biases and output gating. No rotary."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if int(dim) % int(n_heads) != 0:
            raise ValueError(f"n_heads ({n_heads}) must divide dim ({dim}).")
        self.n_heads = int(n_heads)
        self.d_head = int(dim) // self.n_heads
        self.dropout = float(dropout)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_g = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        heads = self.n_heads
        q = self.to_q(q_x).unflatten(-1, (heads, self.d_head)).transpose(-3, -2)
        k = self.to_k(kv_x).unflatten(-1, (heads, self.d_head)).transpose(-3, -2)
        v = self.to_v(kv_x).unflatten(-1, (heads, self.d_head)).transpose(-3, -2)
        g = self.to_g(q_x)

        attn_mask = None
        if biases:
            attn_mask = biases[0]
            for bias in biases[1:]:
                attn_mask = attn_mask + bias

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(-3, -2).reshape(*q_x.shape[:-1], heads * self.d_head)
        return self.to_out(out * g.sigmoid())


class TriangleAttention(nn.Module):
    def __init__(
        self, c_in, c_hidden, no_heads, starting, inf=1e9, dropout=0.0
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Unused; kept for OpenFold-compatible construction
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)
        self.linear = Linear(self.c_in, self.no_heads, bias=False)
        self.mha = BiasGatedAttention(
            dim=self.c_in,
            n_heads=self.no_heads,
            dropout=dropout,
        )

    def _attention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        triangle_bias: torch.Tensor,
    ) -> torch.Tensor:
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        return self.mha(q_x=x, kv_x=x, biases=[mask_bias, triangle_bias])

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        triangle_bias: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        length = int(x.shape[-3])
        chunks = []
        for start in range(0, length, int(chunk_size)):
            end = min(start + int(chunk_size), length)
            chunks.append(
                self._attention(
                    x[..., start:end, :, :],
                    mask[..., start:end, :],
                    triangle_bias,
                )
            )
        return torch.cat(chunks, dim=-3)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        mask = mask.to(dtype=x.dtype)

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1)).unsqueeze(-4)

        if chunk_size is not None:
            x = self._chunk(x, mask, triangle_bias, chunk_size)
        else:
            x = self._attention(x, mask, triangle_bias)

        if not self.starting:
            x = x.transpose(-2, -3)
        return x


class TriangleAttentionStartingNode(TriangleAttention):
    """
    Implements Algorithm 13.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=True)


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
