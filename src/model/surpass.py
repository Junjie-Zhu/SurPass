from typing import Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn

from src.model.components.transformer import (
    ResidueTransformer,
    AtomEncoder,
    TransitionADALN,
    Transition,
)
from src.model.components.embedder import (
    ResidueEmbedder,
    AtomEmbedder,
)


class OuterProductMean(nn.Module):
    def __init__(
        self,
        dim_token=256,
        dim_inner=32,
        dim_pair=128,
    ):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim_token)
        self.linear_no_bias = nn.Linear(dim_token, dim_inner, bias=False)
        self.linear_out = nn.Linear(dim_inner ** 2, dim_pair)

    def forward(
        self,
        token_repr: torch.Tensor,
        mask: torch.Tensor,
    ):
        token_repr = self.layernorm(token_repr)
        token_repr = self.linear_no_bias(token_repr).unsqueeze(-2)

        outer_product = torch.einsum("biad,bjae->bijde", token_repr, token_repr)
        outer_product = outer_product.reshape(outer_product.shape[:-2] + (-1,))

        outer_product = self.linear_out(outer_product)  # B, L, L, D_pair
        return outer_product


class ResOnly(nn.Module):
    def __init__(
        self,
        n_layers=3,
        dim_token=256,
        dim_pair=128,
        n_heads=12,
        residual_mha=True,
        residual_transition=True,
        use_attn_pair_bias=True,
        use_qkln=True,
        dropout=0.0,
        expansion_factor=2,
        **kwargs,
    ):
        super().__init__()
        self.dim_token = dim_token
        self.dim_pair = dim_pair

        # feature embedders
        self.residue_embedder = ResidueEmbedder(
            dim_token=dim_token,
            dim_pair=dim_pair,
            **kwargs,
        )

        # main trunk
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
            OuterProductMean(
                dim_token=dim_token,
                dim_inner=32,
                dim_pair=dim_pair,
            ) for _ in range(n_layers)
        ])

        self.pair_blocks = nn.ModuleList([
            Transition(
                dim=dim_pair,
                expansion_factor=expansion_factor,
            ) for _ in range(n_layers)
        ])

        # prediction head
        self.pair_out_layernorm = nn.LayerNorm(dim_pair)
        self.pair_out_linear = nn.Linear(dim_pair, kwargs.get('num_classes', 1))

    def forward(
        self,
        p1_batch: Dict[str, torch.Tensor],
        p2_batch: Dict[str, torch.Tensor],
    ):
        # featurize
        p1_single_repr, p1_pair_repr, p1_mask = self.residue_embedder(**p1_batch)
        p2_single_repr, p2_pair_repr, p2_mask = self.residue_embedder(**p2_batch)

        # prepare full representation
        batch_size = p1_single_repr.shape[0]
        p1_length = p1_single_repr.shape[1]
        p2_length = p2_single_repr.shape[1]
        single_repr = torch.cat([p1_single_repr, p2_single_repr], dim=-2)
        pair_repr = p1_pair_repr.new_zeros(
            batch_size,
            p1_length + p2_length,
            p1_length + p2_length,
            self.dim_pair,
        )
        pair_repr[..., :p1_length, :p1_length, :] = p1_pair_repr
        pair_repr[..., p1_length:, p1_length:, :] = p2_pair_repr

        mask = torch.cat([p1_mask, p2_mask], dim=-1)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        # main trunk
        for i in range(self.n_layers):
            # first update single representation
            single_repr = self.residue_blocks[i](
                single_repr,
                pair_repr,
                single_repr, # conditioning by itself
                mask,
            )
            # outer pruduct mean
            pair_repr = pair_repr + self.outer_product_mean[i](
                single_repr, 
                mask,
            ) * pair_mask.to(single_repr.dtype)[..., None]
            # update pair representation
            pair_repr = self.pair_blocks[i](
                pair_repr,
                pair_mask.to(single_repr.dtype),
            )

        # output head
        pair_logits = self.pair_out_linear(self.pair_out_layernorm(pair_repr))
        return pair_logits, pair_mask

