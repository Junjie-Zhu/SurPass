"""ResPair: sequence-only pair reasoning with optional hidden-state recycling.

References: RFD4-Proteina SI S3.1 (pp.45-46) and PPLM's
InterTriangleMultiplication_S. ResOnly and its checkpoint contract are unchanged.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Mapping

import torch
from torch import nn

from src.data.labels import contact_bin_count, distance_bin_spec
from src.model.components.embedder import (
    NUM_RESIDUE_TYPES, ResidueEmbedder, sinusoidal_position_embedding,
)
from src.model.components.res_pair import PairBiasedCrossAttention, ResPairBlock
from src.model.components.transformer import ResidueTransformer, SwiGLU
from src.model.loss import protein_contact_logit
from src.model.surpass import (
    SurpassOutput, _layout_mask, _masked_max, _masked_mean,
)


def _chain_gather_plan(start, length, width):
    offset = torch.arange(max(1, int(length.max().item())), device=length.device)[None]
    index = (start[:, None] + offset).clamp(max=width - 1)
    return index, offset < length[:, None]


def _gather_with_plan(value, index):
    if value.ndim == 2:
        return value.gather(1, index)
    return value.gather(1, index[..., None].expand(-1, -1, value.shape[-1]))


class _ResPairEmbedder(ResidueEmbedder):
    """Use the validated contiguous layout for linear-cost residue positions."""

    def forward(self, plm_emb, residue_type, position_index, mask):
        types = residue_type.long().clamp(0, NUM_RESIDUE_TYPES - 1)
        plm = self.plm_projection(plm_emb)
        position = sinusoidal_position_embedding(position_index, self.index_emb_dim)
        features = torch.cat((
            plm, self.residue_embedding(types).to(plm.dtype), position.to(plm.dtype),
        ), dim=-1)
        return self.single_projection(features) * mask[..., None]


class ResPair(nn.Module):
    """Rectangular interchain predictions with learned intrachain triangle context.

    ``recycle_passes`` counts ALL trunk passes, including the first. Only the
    last pass builds a trunk autograd graph. Initial embeddings are computed
    once per forward and receive gradients through the last pass. There is no
    cache across forward calls, samples, or optimizer steps.
    """

    architecture = "res_pair_v1"

    def __init__(
        self, n_layers=12, dim_token=384, dim_pair=96, n_heads=8, dropout=0.0,
        expansion_factor=2, distance_bin_start=2.0, distance_bin_end=22.0,
        distance_bin_count=16, contact_threshold=8.0, contact_prior=1e-4,
        triangle_hidden_dim=None, triangle_chunk_size=64, recycle_passes=1,
        pair_update_every=2, triangle_checkpoint=True, protein_pooling="noisy_or",
        distance_bin_convention="legacy", **embedder_kwargs,
    ):
        super().__init__()
        dim_token, dim_pair, n_heads = int(dim_token), int(dim_pair), int(n_heads)
        hidden = dim_pair if triangle_hidden_dim is None else int(triangle_hidden_dim)
        if min(dim_token, dim_pair, n_heads, hidden) <= 0 or dim_token % n_heads:
            raise ValueError("Positive dimensions required; n_heads must divide dim_token")
        if int(n_layers) < 0 or int(triangle_chunk_size) < 0:
            raise ValueError("n_layers and triangle_chunk_size must be non-negative")
        if int(recycle_passes) < 1 or int(pair_update_every) < 1:
            raise ValueError("recycle_passes and pair_update_every must be positive")
        if not 0.0 < float(contact_prior) < 1.0:
            raise ValueError("contact_prior must be between zero and one")
        self.protein_pooling = protein_pooling
        if protein_pooling not in {"max", "noisy_or"}:
            raise ValueError("protein_pooling must be max or noisy_or")
        self.bin_spec = distance_bin_spec(distance_bin_start, distance_bin_end, distance_bin_count, distance_bin_convention)
        self.dim_token, self.dim_pair = dim_token, dim_pair
        self.n_layers, self.recycle_passes = int(n_layers), int(recycle_passes)
        self.pair_update_every = int(pair_update_every)
        self.distance_bin_count = int(distance_bin_count)
        self.contact_bins = contact_bin_count(
            contact_threshold, distance_bin_start, distance_bin_end, self.distance_bin_count,
            convention=distance_bin_convention,
        )
        if not 0 < self.contact_bins < self.distance_bin_count:
            raise ValueError("The distance bins must include non-contact distances")

        self.residue_embedder = _ResPairEmbedder(
            dim_token=dim_token, dim_pair=dim_pair,
            distance_bin_start=distance_bin_start, distance_bin_end=distance_bin_end,
            distance_bin_count=self.distance_bin_count, **embedder_kwargs,
        )
        self.residue_blocks = nn.ModuleList(
            ResidueTransformer(dim_token, n_heads, dropout=dropout,
                               expansion_factor=expansion_factor)
            for _ in range(self.n_layers)
        )
        self.cross_blocks = nn.ModuleList(
            PairBiasedCrossAttention(dim_token, dim_pair, n_heads, dropout, expansion_factor)
            for _ in range(self.n_layers)
        )
        # Omit unused modules when reducing pair-update frequency.
        self.pair_blocks = nn.ModuleDict({
            str(i): ResPairBlock(dim_token, dim_pair, hidden, triangle_chunk_size,
                                 expansion_factor, dropout, triangle_checkpoint)
            for i in range(self.n_layers) if (i + 1) % self.pair_update_every == 0
        })
        self.recycle_single = nn.Sequential(nn.LayerNorm(dim_token), nn.Linear(dim_token, dim_token, bias=False))
        self.recycle_pair = nn.Sequential(nn.LayerNorm(dim_pair), nn.Linear(dim_pair, dim_pair, bias=False))
        self.pair_head = nn.Sequential(nn.LayerNorm(dim_pair), nn.Linear(dim_pair, self.distance_bin_count))
        with torch.no_grad():
            self.pair_head[-1].bias[:self.contact_bins].fill_(math.log(float(contact_prior) / self.contact_bins))
            self.pair_head[-1].bias[self.contact_bins:].fill_(math.log(
                (1.0 - float(contact_prior)) / (self.distance_bin_count - self.contact_bins),
            ))
        self.auxiliary_head = nn.Sequential(
            nn.LayerNorm(dim_token * 6), nn.Linear(dim_token * 6, dim_token * 2),
            SwiGLU(), nn.Linear(dim_token, 1),
        )

    def _trunk(self, chain_1, chain_2, pair, mask_1, mask_2, pair_mask):
        for i in range(self.n_layers):
            chain_1 = self.residue_blocks[i](chain_1, mask_1)
            chain_2 = self.residue_blocks[i](chain_2, mask_2)
            chain_1, chain_2 = self.cross_blocks[i](chain_1, chain_2, mask_1, mask_2, pair)
            if str(i) in self.pair_blocks:
                pair = self.pair_blocks[str(i)](chain_1, chain_2, pair, mask_1, mask_2, pair_mask)
        return chain_1, chain_2, pair

    def forward(self, residue_batch: Mapping[str, torch.Tensor]) -> SurpassOutput:
        lengths_1, lengths_2 = residue_batch["p1_length"].long(), residue_batch["p2_length"].long()
        width = residue_batch["plm_emb"].shape[1]
        if width < 1:
            raise ValueError("The concatenated input must have at least one position")
        if bool(((lengths_1 < 0) | (lengths_2 < 0)).any()):
            raise ValueError("Chain lengths must be non-negative")
        layout = _layout_mask(lengths_1, lengths_2, width)
        if "padding_mask" in residue_batch:
            supplied = residue_batch["padding_mask"].bool()
            if supplied.shape != layout.shape or not torch.equal(supplied, layout):
                raise ValueError("padding_mask must contain p1 followed by p2 followed by tail padding")
        index = torch.arange(width, device=layout.device)[None, :]
        positions = torch.where(index < lengths_1[:, None], index, index - lengths_1[:, None])
        single = self.residue_embedder(
            residue_batch["plm_emb"], residue_batch["residue_type"], positions, layout,
        )
        # Reuse gather indices and masks for both representations and identities.
        index_1, mask_1 = _chain_gather_plan(torch.zeros_like(lengths_1), lengths_1, width)
        index_2, mask_2 = _chain_gather_plan(lengths_1, lengths_2, width)
        base_1, base_2 = _gather_with_plan(single, index_1), _gather_with_plan(single, index_2)
        type_1 = _gather_with_plan(residue_batch["residue_type"], index_1)
        type_2 = _gather_with_plan(residue_batch["residue_type"], index_2)
        base_pair, pair_mask = self.residue_embedder.cross_pair(base_1, base_2, type_1, type_2, mask_1, mask_2)

        chain_1, chain_2, pair = torch.zeros_like(base_1), torch.zeros_like(base_2), torch.zeros_like(base_pair)
        for recycle in range(self.recycle_passes):
            # RFD4 Algorithm S1: reset each call, add recycled states to the
            # ORIGINAL embeddings, share trunk weights, final-pass gradients.
            context = nullcontext() if recycle == self.recycle_passes - 1 else torch.no_grad()
            # Cached autocast weights from a no-grad pass can otherwise be
            # reused as detached weights in the final pass (and interfere with
            # chunk-checkpoint recomputation). Keep this cache off in the trunk.
            device_type = base_1.device.type
            autocast = torch.autocast(
                device_type, enabled=torch.is_autocast_enabled(device_type),
                dtype=torch.get_autocast_dtype(device_type), cache_enabled=False,
            )
            with context, autocast:
                chain_1 = (base_1 + self.recycle_single(chain_1.detach())) * mask_1[..., None]
                chain_2 = (base_2 + self.recycle_single(chain_2.detach())) * mask_2[..., None]
                pair = (base_pair + self.recycle_pair(pair.detach())) * pair_mask[..., None]
                chain_1, chain_2, pair = self._trunk(chain_1, chain_2, pair, mask_1, mask_2, pair_mask)

        logits = self.pair_head(pair)
        mean_1, mean_2 = _masked_mean(chain_1, mask_1), _masked_mean(chain_2, mask_2)
        max_1, max_2 = _masked_max(chain_1, mask_1), _masked_max(chain_2, mask_2)
        auxiliary = torch.cat((mean_1 + mean_2, (mean_1 - mean_2).abs(), mean_1 * mean_2,
                               max_1 + max_2, (max_1 - max_2).abs(), max_1 * max_2), dim=-1)
        return SurpassOutput(
            protein_logit=protein_contact_logit(logits, pair_mask, self.contact_bins, self.protein_pooling),
            aux_protein_logit=self.auxiliary_head(auxiliary).squeeze(-1),
            pair_logits=logits, pair_mask=pair_mask,
        )
