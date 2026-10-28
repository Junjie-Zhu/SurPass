"""Pair-biased single updates and sequence-only inter/intra triangles.

The contraction follows PPLM's InterTriangleMultiplication_S, using learned
intrachain context rather than structure features. Both chains share weights.
Intrachain features are transient row chunks, never a second pair trunk.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.model.components.transformer import Transition


class PairBiasedCrossAttention(nn.Module):
    """Simultaneous cross-attention with one transposable pair bias."""

    def __init__(self, dim_token, dim_pair, n_heads, dropout=0.0, expansion_factor=2):
        super().__init__()
        self.n_heads = int(n_heads)
        self.head_dim = int(dim_token) // self.n_heads
        self.dropout = float(dropout)
        self.single_norm = nn.LayerNorm(dim_token)
        self.pair_norm = nn.LayerNorm(dim_pair)
        self.qkv = nn.Linear(dim_token, 3 * dim_token, bias=False)
        self.gate = nn.Linear(dim_token, dim_token)
        self.pair_bias = nn.Linear(dim_pair, n_heads, bias=False)
        self.output_projection = nn.Linear(dim_token, dim_token)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.transition = Transition(dim_token, expansion_factor)

    def _prepare(self, single):
        single = self.single_norm(single)
        qkv = self.qkv(single).view(
            *single.shape[:2], 3, self.n_heads, self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        return (*qkv.unbind(0), self.gate(single).sigmoid())

    def _attend(self, query, context, mask, context_mask, bias):
        q, _, _, gate = query
        _, k, v, _ = context
        bias = bias.to(q.dtype).masked_fill(~context_mask[:, None, None, :], -torch.inf)
        value = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2).flatten(2)
        # Also suppress the output bias when a sample has no context residues.
        valid = mask & context_mask.any(-1, keepdim=True)
        return self.output_projection(value * gate) * valid[..., None]

    def forward(self, chain_1, chain_2, mask_1, mask_2, pair):
        first, second = self._prepare(chain_1), self._prepare(chain_2)
        bias = self.pair_bias(self.pair_norm(pair)).permute(0, 3, 1, 2)
        update_1 = self._attend(first, second, mask_1, mask_2, bias)
        update_2 = self._attend(second, first, mask_2, mask_1, bias.transpose(-1, -2))
        chain_1, chain_2 = chain_1 + update_1, chain_2 + update_2
        chain_1 = chain_1 + self.transition(chain_1, mask_1)
        chain_2 = chain_2 + self.transition(chain_2, mask_2)
        return chain_1 * mask_1[..., None], chain_2 * mask_2[..., None]


class SymmetricSingleToPair(nn.Module):
    """RFD4 Hadamard update, symmetrized to preserve chain-swap equivariance."""

    def __init__(self, dim_token, dim_pair):
        super().__init__()
        self.norm = nn.LayerNorm(dim_token)
        self.projection = nn.Linear(dim_token, 2 * dim_pair, bias=False)

    def forward(self, single):
        return self.projection(self.norm(single)).chunk(2, dim=-1)

    @staticmethod
    def combine(first, second):
        a, b = first
        c, d = second
        return 0.5 * (a[:, :, None] * d[:, None, :] + b[:, :, None] * c[:, None, :])


class InterIntraTriangle(nn.Module):
    """Gated intra/inter multiplication on both ends of a rectangular pair.

    T_ijc = sum_k A_ikc I_kjc + sum_l I_ilc B_jlc.
    A and B use the same projections of sequence-derived intrachain pairs.
    Unlike AF2 outgoing/ingoing updates, this only evolves the interchain map.
    Runtime is O(L1*L2*(L1+L2)*hidden). Chunking bounds temporary intrachain
    storage; it does not change the arithmetic complexity.
    """

    def __init__(self, dim_pair, hidden_dim, chunk_size=64, checkpoint_chunks=True):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.checkpoint_chunks = bool(checkpoint_chunks)
        self.norm = nn.LayerNorm(dim_pair)
        self.inter_projection = nn.Linear(dim_pair, 2 * hidden_dim, bias=False)
        self.intra_projection = nn.Linear(dim_pair, 2 * hidden_dim, bias=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, dim_pair, bias=False)
        self.output_gate = nn.Linear(dim_pair, dim_pair)
        nn.init.zeros_(self.output_projection.weight)

    @staticmethod
    def _gated(value):
        value, gate = value.chunk(2, dim=-1)
        return value * gate.sigmoid()

    def _contract(self, inter, factors, mask):
        length = mask.shape[1]
        chunk = self.chunk_size or length
        updates = []
        for start in range(0, length, chunk):
            stop = min(start + chunk, length)
            intra_mask = mask[:, start:stop, None] & mask[:, None, :]
            args = (inter, factors[0][:, start:stop], factors[1][:, start:stop],
                    *factors, intra_mask)
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                update = checkpoint(self._row_contract, *args, use_reentrant=False)
            else:
                update = self._row_contract(*args)
            updates.append(update)
        return torch.cat(updates, dim=1)

    def _row_contract(self, inter, a_rows, b_rows, a, b, mask):
        intra = SymmetricSingleToPair.combine((a_rows, b_rows), (a, b))
        intra = self._gated(self.intra_projection(self.norm(intra))) * mask[..., None]
        return torch.einsum("bikc,bkjc->bijc", intra, inter)

    def forward(self, pair, factors_1, factors_2, mask_1, mask_2, pair_mask):
        normalized = self.norm(pair)
        inter = self._gated(self.inter_projection(normalized)) * pair_mask[..., None]
        left = self._contract(inter, factors_1, mask_1)
        right = self._contract(inter.transpose(1, 2), factors_2, mask_2).transpose(1, 2)
        update = self.output_projection(self.output_norm(left + right))
        return update * self.output_gate(normalized).sigmoid() * pair_mask[..., None]


class ResPairBlock(nn.Module):
    """Single -> pair, triangle multiplication, then pair transition."""

    def __init__(self, dim_token, dim_pair, hidden_dim, chunk_size=64,
                 expansion_factor=2, dropout=0.0, checkpoint_chunks=True):
        super().__init__()
        self.single_to_pair = SymmetricSingleToPair(dim_token, dim_pair)
        self.triangle = InterIntraTriangle(dim_pair, hidden_dim, chunk_size, checkpoint_chunks)
        self.transition = Transition(dim_pair, expansion_factor)
        self.dropout = nn.Dropout(dropout)

    def forward(self, chain_1, chain_2, pair, mask_1, mask_2, pair_mask):
        # These projections serve both the interchain and intrachain updates.
        first = self.single_to_pair(chain_1)
        second = self.single_to_pair(chain_2)
        pair = pair + self.dropout(SymmetricSingleToPair.combine(first, second)) * pair_mask[..., None]
        pair = pair + self.dropout(self.triangle(pair, first, second, mask_1, mask_2, pair_mask))
        pair = pair + self.dropout(self.transition(pair, pair_mask))
        return pair * pair_mask[..., None]
