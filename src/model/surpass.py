from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.model.components.embedder import ResidueEmbedder
from src.model.components.outer_product_mean import EfficientOuterProductMean
from src.model.components.primitives import PairTransition
from src.model.components.transformer import ResidueTransformer

_RESIDUE_LENGTH_KEYS = frozenset({"p1_length", "p2_length"})


@dataclass
class SurpassOutput:
    pair_logits: torch.Tensor
    residue_logits: torch.Tensor
    protein_logit: torch.Tensor
    pair_mask: torch.Tensor


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
        dim_opm_inner=32,
        checkpoint_pair_blocks=True,
        contact_bins: int = 1,
        **kwargs,
    ):
        super().__init__()
        if int(dim_token) % int(n_heads) != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must divide dim_token ({dim_token})."
            )
        xt_pair_dist_dim = int(kwargs.get("xt_pair_dist_dim", 64))
        num_classes = int(kwargs.get("num_classes", xt_pair_dist_dim))
        if num_classes != xt_pair_dist_dim:
            raise ValueError(
                f"num_classes ({num_classes}) must equal "
                f"xt_pair_dist_dim ({xt_pair_dist_dim})."
            )
        if int(contact_bins) <= 0 or int(contact_bins) > num_classes:
            raise ValueError(
                f"contact_bins ({contact_bins}) must be in 1..{num_classes}."
            )
        self.dim_token = dim_token
        self.dim_pair = dim_pair
        self.contact_bins = int(contact_bins)
        self.checkpoint_pair_blocks = bool(checkpoint_pair_blocks)

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
                dim_cond=dim_token,
                nheads=n_heads,
                residual_mha=residual_mha,
                residual_transition=residual_transition,
                use_attn_pair_bias=use_attn_pair_bias,
                use_qkln=use_qkln,
                dropout=dropout,
                expansion_factor=expansion_factor,
            ) for _ in range(n_layers)
        ])

        self.outer_product_mean = nn.ModuleList([
            EfficientOuterProductMean(
                c_m=dim_token,
                c_z=dim_pair,
                c_hidden=dim_opm_inner,
            ) for _ in range(n_layers)
        ])

        self.pair_blocks = nn.ModuleList([
            PairTransition(
                dim=dim_pair,
                expansion_factor=expansion_factor,
                layer_norm=True,
            ) for _ in range(n_layers)
        ])

        self.pair_out_layernorm = nn.LayerNorm(dim_pair)
        self.pair_out_linear = nn.Linear(dim_pair, num_classes)
        self.residue_bind_norm = nn.LayerNorm(dim_token)
        self.residue_bind_linear = nn.Linear(dim_token, 1)
        self.protein_head = nn.Sequential(
            nn.LayerNorm(dim_token * 3),
            nn.Linear(dim_token * 3, dim_token),
            nn.GELU(),
            nn.Linear(dim_token, 1),
        )
        self.residue_contact_gate = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    def reset_parameters(self):
        for opm in self.outer_product_mean:
            nn.init.zeros_(opm.linear_out.weight)
            if opm.linear_out.bias is not None:
                nn.init.zeros_(opm.linear_out.bias)
        for block in self.pair_blocks:
            nn.init.zeros_(block.linear_out.weight)
            if block.linear_out.bias is not None:
                nn.init.zeros_(block.linear_out.bias)

    def forward(
        self,
        residue_batch: Dict[str, torch.Tensor],
        chunk_size: Optional[int] = None,
    ) -> SurpassOutput:
        chunk_size = _normalize_chunk_size(chunk_size)
        embedder_inputs = {
            key: value
            for key, value in residue_batch.items()
            if key not in _RESIDUE_LENGTH_KEYS
        }
        single_repr, pair_repr, mask = self.residue_embedder(**embedder_inputs)
        mask = mask.to(dtype=torch.bool)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair_mask_float = pair_mask.to(dtype=pair_repr.dtype)

        for i in range(self.n_layers):
            single_repr = self.residue_blocks[i](
                single_repr,
                pair_repr,
                single_repr,
                mask,
            )
            pair_repr = self._apply_pair_stack(
                i,
                single_repr,
                pair_repr,
                mask,
                pair_mask_float,
                chunk_size,
            )

        pair_logits = self.pair_out_linear(self.pair_out_layernorm(pair_repr))
        pair_logits = pair_logits + pair_logits.transpose(-2, -3)
        pair_logits = pair_logits / 2
        residue_logits = self.residue_bind_linear(
            self.residue_bind_norm(single_repr)
        ).squeeze(-1)
        residue_logits = residue_logits.masked_fill(~mask, 0.0)
        pair_logits = pair_logits + self._contact_bin_bias(residue_logits)
        pair_mask = pair_mask | pair_mask.transpose(-1, -2)
        protein_logit = self._protein_logit(
            single_repr,
            mask,
            residue_batch["p1_length"],
            residue_batch["p2_length"],
        )
        return SurpassOutput(
            pair_logits=pair_logits,
            residue_logits=residue_logits,
            protein_logit=protein_logit,
            pair_mask=pair_mask,
        )

    def _contact_bin_bias(self, residue_logits: torch.Tensor) -> torch.Tensor:
        pair_bias = residue_logits[:, :, None] + residue_logits[:, None, :]
        pair_bias = pair_bias * self.residue_contact_gate
        class_count = self.pair_out_linear.out_features
        far_bins = class_count - self.contact_bins
        contact_bias = pair_bias.unsqueeze(-1).expand(
            *pair_bias.shape, self.contact_bins
        )
        if far_bins <= 0:
            return contact_bias
        far_bias = pair_bias.new_zeros(*pair_bias.shape, far_bins)
        return torch.cat([contact_bias, far_bias], dim=-1)

    def _protein_logit(
        self,
        single_repr: torch.Tensor,
        mask: torch.Tensor,
        p1_length: torch.Tensor,
        p2_length: torch.Tensor,
    ) -> torch.Tensor:
        h1 = _masked_chain_mean(single_repr, mask, p1_length, p2_length, chain=1)
        h2 = _masked_chain_mean(single_repr, mask, p1_length, p2_length, chain=2)
        features = torch.cat([h1, h2, h1 * h2], dim=-1)
        return self.protein_head(features).squeeze(-1)

    def _pair_stack(
        self,
        layer_index: int,
        single_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        mask: torch.Tensor,
        pair_mask_float: torch.Tensor,
        chunk_size: Optional[int],
    ) -> torch.Tensor:
        pair_repr = pair_repr + self.outer_product_mean[layer_index](
            single_repr,
            mask,
        ) * pair_mask_float[..., None]
        pair_repr = pair_repr + self.pair_blocks[layer_index](
            pair_repr,
            pair_mask_float,
            chunk_size=chunk_size,
        )
        return pair_repr

    def _apply_pair_stack(
        self,
        layer_index: int,
        single_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        mask: torch.Tensor,
        pair_mask_float: torch.Tensor,
        chunk_size: Optional[int],
    ) -> torch.Tensor:
        use_checkpoint = (
            self.checkpoint_pair_blocks
            and self.training
            and pair_repr.requires_grad
        )
        if not use_checkpoint:
            return self._pair_stack(
                layer_index,
                single_repr,
                pair_repr,
                mask,
                pair_mask_float,
                chunk_size,
            )

        def _run(single, pair, node_mask, pair_mask):
            return self._pair_stack(
                layer_index,
                single,
                pair,
                node_mask,
                pair_mask,
                chunk_size,
            )

        return checkpoint(
            _run,
            single_repr,
            pair_repr,
            mask,
            pair_mask_float,
            use_reentrant=False,
        )


def _masked_chain_mean(
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
        selected = (index < p1) & mask
    else:
        selected = (index >= p1) & (index < p1 + p2) & mask
    weight = selected.to(dtype=single_repr.dtype)
    denom = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (single_repr * weight[..., None]).sum(dim=1) / denom


def _normalize_chunk_size(chunk_size: Optional[int]) -> Optional[int]:
    if chunk_size is None:
        return None
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        return None
    return chunk_size
