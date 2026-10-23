"""SurPass v2: an O(L1*L2) inter-chain contact model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from src.data.labels import contact_bin_count
from src.model.components.embedder import ResidueEmbedder
from src.model.components.inter_pair import InterPairBlock
from src.model.components.transformer import (
    BidirectionalCrossAttention,
    ResidueTransformer,
    SwiGLU,
    Transition,
)
from src.model.loss import noisy_or_contact_logit


@dataclass
class SurpassOutput:
    """Model outputs.

    ``pair_logits`` and ``pair_mask`` are rectangular [B,L1,L2,...] tensors.
    ``protein_logit`` is the noisy-OR training logit. Inference and checkpoint
    selection use the maximum valid contact probability from ``pair_logits``.
    """

    protein_logit: torch.Tensor
    aux_protein_logit: torch.Tensor
    pair_logits: torch.Tensor
    pair_mask: torch.Tensor


def _layout_mask(
    lengths_1: torch.Tensor,
    lengths_2: torch.Tensor,
    width: int,
) -> torch.Tensor:
    total = lengths_1.long() + lengths_2.long()
    if bool((total > int(width)).any()):
        raise ValueError(
            f"p1_length + p2_length exceeds concatenated width {int(width)}"
        )
    index = torch.arange(int(width), device=total.device).view(1, -1)
    return index < total[:, None]


def _gather_chain(
    value: torch.Tensor,
    start: torch.Tensor,
    length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(1, int(length.max().item()))
    offset = torch.arange(maximum, device=value.device).view(1, -1)
    index = start.long()[:, None] + offset
    valid = offset < length.long()[:, None]
    index = index.clamp(max=max(0, value.shape[1] - 1))
    gather_shape = [value.shape[0], maximum] + list(value.shape[2:])
    gather_index = index.view(value.shape[0], maximum, *([1] * (value.ndim - 2)))
    gather_index = gather_index.expand(*gather_shape)
    gathered = value.gather(1, gather_index)
    return gathered, valid


def split_concatenated(
    value: torch.Tensor,
    lengths_1: torch.Tensor,
    lengths_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zeros = torch.zeros_like(lengths_1)
    chain_1, mask_1 = _gather_chain(value, zeros, lengths_1)
    chain_2, mask_2 = _gather_chain(value, lengths_1, lengths_2)
    return chain_1, chain_2, mask_1, mask_2


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    return (value * weight[..., None]).sum(dim=1) / weight.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0)


def _masked_max(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    filled = value.masked_fill(
        ~mask[..., None],
        torch.finfo(value.dtype).min,
    )
    maximum = filled.amax(dim=1)
    return torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))


class ResOnly(nn.Module):
    """Sequence-only PPI model with a rectangular inter-chain pair track."""

    architecture = "inter_pair_v3"

    def __init__(
        self,
        n_layers: int = 2,
        dim_token: int = 384,
        dim_pair: int = 96,
        n_heads: int = 8,
        dropout: float = 0.0,
        expansion_factor: int = 2,
        distance_bin_start: float = 2.0,
        distance_bin_end: float = 22.0,
        distance_bin_count: int = 16,
        contact_threshold: float = 8.0,
        contact_prior: float = 1e-4,
        **embedder_kwargs: object,
    ):
        super().__init__()
        dim_token = int(dim_token)
        dim_pair = int(dim_pair)
        n_heads = int(n_heads)
        if dim_token % n_heads:
            raise ValueError("n_heads must divide dim_token")
        self.dim_token = dim_token
        self.dim_pair = dim_pair
        self.distance_bin_count = int(distance_bin_count)
        self.contact_bins = contact_bin_count(
            contact_threshold,
            distance_bin_start,
            distance_bin_end,
            self.distance_bin_count,
        )
        if not 0 < self.contact_bins < self.distance_bin_count:
            raise ValueError("The distance bins must include non-contact distances")
        if not 0.0 < float(contact_prior) < 1.0:
            raise ValueError("contact_prior must be between zero and one")
        if int(n_layers) < 0:
            raise ValueError("n_layers must be non-negative")

        self.residue_embedder = ResidueEmbedder(
            dim_token=dim_token,
            dim_pair=dim_pair,
            distance_bin_start=distance_bin_start,
            distance_bin_end=distance_bin_end,
            distance_bin_count=self.distance_bin_count,
            **embedder_kwargs,
        )
        self.residue_blocks = nn.ModuleList(
            ResidueTransformer(
                dim_token,
                n_heads,
                dropout=dropout,
                expansion_factor=expansion_factor,
            )
            for _ in range(int(n_layers))
        )
        self.cross_blocks = nn.ModuleList(
            BidirectionalCrossAttention(
                dim_token,
                n_heads,
                dropout=dropout,
                expansion_factor=expansion_factor,
            )
            for _ in range(int(n_layers))
        )
        self.pair_blocks = nn.ModuleList(
            InterPairBlock(
                dim_pair,
                expansion_factor=expansion_factor,
                dilation=1 if index == 0 else 2,
                dropout=dropout,
            )
            for index in range(int(n_layers))
        )
        self.pair_left = nn.ModuleList(
            nn.Linear(dim_token, dim_pair)
            for _ in range(int(n_layers))
        )
        self.pair_product = nn.ModuleList(
            nn.Linear(dim_token, dim_pair)
            for _ in range(int(n_layers))
        )
        self.pair_update_bias = nn.ParameterList(
            nn.Parameter(torch.zeros(dim_pair))
            for _ in range(int(n_layers))
        )
        self.pair_transition = nn.ModuleList(
            Transition(dim_pair, expansion_factor)
            for _ in range(int(n_layers))
        )
        self.pair_token_norm = nn.ModuleList(
            nn.LayerNorm(dim_token) for _ in range(int(n_layers))
        )
        self.n_layers = int(n_layers)

        self.pair_head = nn.Sequential(
            nn.LayerNorm(dim_pair),
            nn.Linear(dim_pair, distance_bin_count),
        )
        # A sparse contact prior prevents a uniform distance softmax from
        # immediately saturating noisy-OR on thousands of residue pairs.
        with torch.no_grad():
            self.pair_head[-1].bias[:self.contact_bins].fill_(
                math.log(float(contact_prior) / self.contact_bins)
            )
            self.pair_head[-1].bias[self.contact_bins:].fill_(
                math.log((1.0 - float(contact_prior)) / (
                    self.distance_bin_count - self.contact_bins
                ))
            )
        self.auxiliary_head = nn.Sequential(
            nn.LayerNorm(dim_token * 6),
            nn.Linear(dim_token * 6, dim_token * 2),
            SwiGLU(),
            nn.Linear(dim_token, 1),
        )

    def cross_pair_update(
        self, 
        chain_1: torch.Tensor, 
        chain_2: torch.Tensor, 
        pair: torch.Tensor, 
        pair_mask: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        chain_1 = self.pair_token_norm[layer](chain_1)
        chain_2 = self.pair_token_norm[layer](chain_2)
        left_update = self.pair_left[layer](chain_1)
        right_update = self.pair_left[layer](chain_2)
        left_product_update = self.pair_product[layer](chain_1)
        right_product_update = self.pair_product[layer](chain_2)
        pair_update = (
            left_update[:, :, None, :]
            + right_update[:, None, :, :]
            + left_product_update[:, :, None, :] * right_product_update[:, None, :, :]
        )
        pair_update = pair_update + self.pair_update_bias[layer]
        pair_update = self.pair_transition[layer](pair_update, pair_mask)
        return (pair + pair_update) * pair_mask[..., None]


    def forward(
        self,
        residue_batch: Mapping[str, torch.Tensor],
    ) -> SurpassOutput:
        lengths_1 = residue_batch["p1_length"].long()
        lengths_2 = residue_batch["p2_length"].long()
        width = int(residue_batch["plm_emb"].shape[1])
        layout_mask = _layout_mask(lengths_1, lengths_2, width)
        if "padding_mask" in residue_batch:
            supplied = residue_batch["padding_mask"].to(dtype=torch.bool)
            if supplied.shape != layout_mask.shape or not torch.equal(supplied, layout_mask):
                raise ValueError(
                    "padding_mask must contain p1 followed by p2 followed by tail padding"
                )

        single = self.residue_embedder(
            plm_emb=residue_batch["plm_emb"],
            residue_type=residue_batch["residue_type"],
            chain_index=residue_batch["chain_index"],
            mask=layout_mask,
        )
        chain_1, chain_2, mask_1, mask_2 = split_concatenated(
            single,
            lengths_1,
            lengths_2,
        )
        type_1, type_2, _, _ = split_concatenated(
            residue_batch["residue_type"],
            lengths_1,
            lengths_2,
        )
        pair, pair_mask = self.residue_embedder.cross_pair(
            chain_1,
            chain_2,
            type_1,
            type_2,
            mask_1,
            mask_2,
        )

        for layer in range(self.n_layers):
            # Update chains separately, then exchange context and update pairs.
            chain_1 = self.residue_blocks[layer](chain_1, mask_1)
            chain_2 = self.residue_blocks[layer](chain_2, mask_2)

            # update with cross attention
            chain_1, chain_2 = self.cross_blocks[layer](chain_1, chain_2, mask_1, mask_2)

            # update pair
            pair = self.pair_blocks[layer](pair, pair_mask)

            # update pair with single chains
            pair = self.cross_pair_update(chain_1, chain_2, pair, pair_mask, layer)

        pair_logits = self.pair_head(pair)

        protein_logit = noisy_or_contact_logit(
            pair_logits,
            pair_mask,
            self.contact_bins,
        )

        mean_1 = _masked_mean(chain_1, mask_1)
        mean_2 = _masked_mean(chain_2, mask_2)
        max_1 = _masked_max(chain_1, mask_1)
        max_2 = _masked_max(chain_2, mask_2)
        auxiliary_features = torch.cat(
            (
                mean_1 + mean_2,
                (mean_1 - mean_2).abs(),
                mean_1 * mean_2,
                max_1 + max_2,
                (max_1 - max_2).abs(),
                max_1 * max_2,
            ),
            dim=-1,
        )
        auxiliary_logit = self.auxiliary_head(auxiliary_features).squeeze(-1)
        return SurpassOutput(
            protein_logit=protein_logit,
            aux_protein_logit=auxiliary_logit,
            pair_logits=pair_logits,
            pair_mask=pair_mask,
        )
