from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from src.model.components.embedder import ResidueEmbedder
from src.model.components.transformer import ResidueTransformer

_RESIDUE_LENGTH_KEYS = frozenset({"p1_length", "p2_length"})


@dataclass
class SurpassOutput:
    protein_logit: torch.Tensor


class ResOnly(nn.Module):
    def __init__(
        self,
        n_layers=3,
        dim_token=256,
        dim_pair=128,
        n_heads=8,
        residual_mha=True,
        residual_transition=True,
        use_attn_pair_bias=True,
        use_qkln=True,
        dropout=0.0,
        expansion_factor=2,
        **kwargs,
    ):
        super().__init__()
        if int(dim_token) % int(n_heads) != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must divide dim_token ({dim_token})."
            )
        self.dim_token = dim_token
        self.dim_pair = dim_pair
        for unused_key in (
            "contact_bins",
            "checkpoint_pair_blocks",
            "num_classes",
            "dim_opm_inner",
            "dim_triangle_hidden",
        ):
            kwargs.pop(unused_key, None)

        self.residue_embedder = ResidueEmbedder(
            dim_token=dim_token,
            dim_pair=dim_pair,
            **kwargs,
        )

        self.n_layers = n_layers
        self.residue_blocks = nn.ModuleList([
            ResidueTransformer(
                dim_token=dim_token,
                dim_pair=dim_pair,
                nheads=n_heads,
                residual_mha=residual_mha,
                residual_transition=residual_transition,
                use_attn_pair_bias=use_attn_pair_bias,
                use_qkln=use_qkln,
                dropout=dropout,
                expansion_factor=expansion_factor,
            ) for _ in range(n_layers)
        ])

        self.protein_head = nn.Sequential(
            nn.LayerNorm(dim_token * 6),
            nn.Linear(dim_token * 6, dim_token),
            nn.GELU(),
            nn.Linear(dim_token, 1),
        )

    def forward(
        self,
        residue_batch: Dict[str, torch.Tensor],
    ) -> SurpassOutput:
        embedder_inputs = {
            key: value
            for key, value in residue_batch.items()
            if key not in _RESIDUE_LENGTH_KEYS
        }
        single_repr, pair_repr, mask = self.residue_embedder(**embedder_inputs)
        mask = mask.to(dtype=torch.bool)

        for block in self.residue_blocks:
            single_repr = block(single_repr, pair_repr, mask)

        protein_logit = self._protein_logit(
            single_repr,
            mask,
            residue_batch["p1_length"],
            residue_batch["p2_length"],
        )
        return SurpassOutput(protein_logit=protein_logit)

    def _protein_logit(
        self,
        single_repr: torch.Tensor,
        mask: torch.Tensor,
        p1_length: torch.Tensor,
        p2_length: torch.Tensor,
    ) -> torch.Tensor:
        m1 = _masked_chain_mean(single_repr, mask, p1_length, p2_length, chain=1)
        m2 = _masked_chain_mean(single_repr, mask, p1_length, p2_length, chain=2)
        x1 = _masked_chain_max(single_repr, mask, p1_length, p2_length, chain=1)
        x2 = _masked_chain_max(single_repr, mask, p1_length, p2_length, chain=2)
        features = torch.cat([m1, m2, m1 * m2, x1, x2, x1 * x2], dim=-1)
        return self.protein_head(features).squeeze(-1)


def _chain_selector(
    single_repr: torch.Tensor,
    mask: torch.Tensor,
    p1_length: torch.Tensor,
    p2_length: torch.Tensor,
    chain: int,
) -> torch.Tensor:
    length = single_repr.shape[1]
    index = torch.arange(length, device=single_repr.device).view(1, length)
    p1 = p1_length.to(device=single_repr.device).view(-1, 1)
    p2 = p2_length.to(device=single_repr.device).view(-1, 1)
    if chain == 1:
        return (index < p1) & mask
    return (index >= p1) & (index < p1 + p2) & mask


def _masked_chain_mean(
    single_repr: torch.Tensor,
    mask: torch.Tensor,
    p1_length: torch.Tensor,
    p2_length: torch.Tensor,
    chain: int,
) -> torch.Tensor:
    selected = _chain_selector(single_repr, mask, p1_length, p2_length, chain)
    weight = selected.to(dtype=single_repr.dtype)
    denom = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (single_repr * weight[..., None]).sum(dim=1) / denom


def _masked_chain_max(
    single_repr: torch.Tensor,
    mask: torch.Tensor,
    p1_length: torch.Tensor,
    p2_length: torch.Tensor,
    chain: int,
) -> torch.Tensor:
    selected = _chain_selector(single_repr, mask, p1_length, p2_length, chain)
    filled = single_repr.masked_fill(
        ~selected[..., None],
        torch.finfo(single_repr.dtype).min,
    )
    pooled = filled.amax(dim=1)
    empty = ~selected.any(dim=1)
    return torch.where(empty[:, None], torch.zeros_like(pooled), pooled)
