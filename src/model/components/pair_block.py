import torch
import torch.nn.functional as F
from torch import nn

from src.model.components.outer_product_mean import EfficientOuterProductMean
from src.model.components.primitives import PairTransition


def _row_spans(rows: int, chunk_size: int):
    """Half-open spans covering ``rows`` in groups of ``chunk_size``."""
    chunk = int(chunk_size)
    if chunk < 1:
        raise ValueError(f"row chunk size must be positive, got {chunk_size}.")
    start = 0
    while start < int(rows):
        end = min(start + chunk, int(rows))
        yield start, end
        start = end


class AxialAttention(nn.Module):
    """Pre-norm self-attention along the last residue axis of a pair tensor.

    ``pair`` is ``[B, I, J, C]`` and attention runs over ``J`` independently for
    each row. The module returns the update, not the residual sum.

    Rows are scored in chunks. A fully valid chunk passes no attention mask so
    scaled-dot-product attention can use the flash kernel. A chunk with padding
    uses a key mask of shape ``[rows, 1, 1, cols]`` instead of a dense
    ``[rows, cols, cols]`` mask.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        row_chunk_size: int = 32,
    ):
        super().__init__()
        if int(n_heads) <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}.")
        if int(dim) % int(n_heads) != 0:
            raise ValueError(f"n_heads ({n_heads}) must divide dim ({dim}).")
        if int(row_chunk_size) < 1:
            raise ValueError(f"row_chunk_size must be positive, got {row_chunk_size}.")
        self.n_heads = int(n_heads)
        self.d_head = int(dim) // self.n_heads
        self.dropout = float(dropout)
        self.row_chunk_size = int(row_chunk_size)
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_g = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        nn.init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            nn.init.zeros_(self.to_out.bias)

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        rows = int(pair.shape[-3])
        if self.row_chunk_size >= rows:
            return self._attend_rows(pair, pair_mask)
        parts = [
            self._attend_rows(pair[..., start:end, :, :], pair_mask[..., start:end, :])
            for start, end in _row_spans(rows, self.row_chunk_size)
        ]
        return torch.cat(parts, dim=-3)

    def _attend_rows(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        batch, rows, cols, channels = pair.shape
        hidden = self.norm(pair).reshape(batch * rows, cols, channels)
        mask = pair_mask.reshape(batch * rows, cols).to(dtype=torch.bool)
        heads = self.n_heads

        query = self.to_q(hidden).unflatten(-1, (heads, self.d_head)).transpose(1, 2)
        key = self.to_k(hidden).unflatten(-1, (heads, self.d_head)).transpose(1, 2)
        value = self.to_v(hidden).unflatten(-1, (heads, self.d_head)).transpose(1, 2)
        gate = self.to_g(hidden)

        attn_mask = None
        if not bool(mask.all()):
            valid_row = mask.any(dim=-1)
            key_keep = mask | ~valid_row[:, None]
            attn_mask = key_keep[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch * rows, cols, channels)
        update = self.to_out(attended * gate.sigmoid())
        update = update * mask.to(dtype=update.dtype)[..., None]
        return update.reshape(batch, rows, cols, channels)


class PairBlock(nn.Module):
    """Outer product, row attention, column attention, then a transition.

    Output projections start at zero, so the block is a no-op at initialization.
    """

    def __init__(
        self,
        dim_token: int,
        dim_pair: int,
        n_heads: int,
        dim_opm_inner: int,
        dropout: float = 0.0,
        expansion_factor: int = 2,
        row_chunk_size: int = 32,
    ):
        super().__init__()
        if int(dim_opm_inner) <= 0:
            raise ValueError(f"dim_opm_inner must be positive, got {dim_opm_inner}.")
        if int(row_chunk_size) < 1:
            raise ValueError(f"row_chunk_size must be positive, got {row_chunk_size}.")
        self.row_chunk_size = int(row_chunk_size)
        self.opm = EfficientOuterProductMean(
            c_m=int(dim_token),
            c_hidden=int(dim_opm_inner),
            c_z=int(dim_pair),
        )
        nn.init.zeros_(self.opm.linear_out.weight)
        if self.opm.linear_out.bias is not None:
            nn.init.zeros_(self.opm.linear_out.bias)
        self.row_attn = AxialAttention(
            int(dim_pair),
            int(n_heads),
            dropout=dropout,
            row_chunk_size=self.row_chunk_size,
        )
        self.col_attn = AxialAttention(
            int(dim_pair),
            int(n_heads),
            dropout=dropout,
            row_chunk_size=self.row_chunk_size,
        )
        self.transition = PairTransition(
            dim=int(dim_pair),
            expansion_factor=int(expansion_factor),
            layer_norm=True,
        )
        nn.init.zeros_(self.transition.linear_out.weight)
        if self.transition.linear_out.bias is not None:
            nn.init.zeros_(self.transition.linear_out.bias)

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        residue_mask = mask.to(dtype=torch.bool)
        pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]
        pair = pair + self.opm(
            single,
            residue_mask,
            row_chunk_size=self.row_chunk_size,
        )
        pair = pair + self.row_attn(pair, pair_mask)
        swapped = pair.transpose(-3, -2)
        swapped = swapped + self.col_attn(swapped, pair_mask.transpose(-1, -2))
        pair = swapped.transpose(-3, -2)
        pair = pair + self._transition_rows(pair, pair_mask)
        return pair * pair_mask.to(dtype=pair.dtype)[..., None]

    def _transition_rows(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        rows = int(pair.shape[-3])
        if self.row_chunk_size >= rows:
            return self.transition(pair, pair_mask)
        parts = [
            self.transition(pair[:, start:end], pair_mask[:, start:end])
            for start, end in _row_spans(rows, self.row_chunk_size)
        ]
        return torch.cat(parts, dim=1)
