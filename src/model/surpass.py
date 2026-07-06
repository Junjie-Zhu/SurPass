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
        self.reset_parameters()

    def reset_parameters(self):
        for parameter in self.parameters():
            nn.init.zeros_(parameter)

    def forward(
        self,
        p1_batch: Dict[str, torch.Tensor],
        p2_batch: Dict[str, torch.Tensor],
        self_conditioning_bins: torch.Tensor | None = None,
        recycle_rounds: int = 1,
    ):
        recycle_rounds = max(1, int(recycle_rounds))
        if self_conditioning_bins is None:
            self_conditioning_bins = self._zero_self_conditioning(p1_batch, p2_batch)

        recycled_bins = self_conditioning_bins
        for _ in range(recycle_rounds - 1):
            with torch.no_grad():
                logits, _ = self._forward_once(p1_batch, p2_batch, recycled_bins)
                recycled_bins = torch.softmax(logits, dim=-1).detach()

        return self._forward_once(p1_batch, p2_batch, recycled_bins)

    def _zero_self_conditioning(
        self,
        p1_batch: Dict[str, torch.Tensor],
        p2_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        p1_mask = p1_batch["mask"]
        p2_mask = p2_batch["mask"]
        total_length = p1_mask.shape[1] + p2_mask.shape[1]
        return p1_mask.new_zeros(
            p1_mask.shape[0],
            total_length,
            total_length,
            self.pair_out_linear.out_features,
            dtype=torch.float32,
        )

    @staticmethod
    def _concat_batches(
        p1_batch: Dict[str, torch.Tensor],
        p2_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        concat_batch = {}
        for key, p1_value in p1_batch.items():
            p2_value = p2_batch[key]
            if key == "chain_index":
                p1_chain_index = p1_value.long()
                p2_chain_index = p2_value.long() + p1_chain_index.amax(dim=1, keepdim=True) + 1
                concat_batch[key] = torch.cat([p1_chain_index, p2_chain_index], dim=1)
            else:
                concat_batch[key] = torch.cat([p1_value, p2_value], dim=1)
        return concat_batch

    def _forward_once(
        self,
        p1_batch: Dict[str, torch.Tensor],
        p2_batch: Dict[str, torch.Tensor],
        self_conditioning_bins: torch.Tensor,
    ):
        residue_batch = self._concat_batches(p1_batch, p2_batch)
        expected_shape = (
            residue_batch["mask"].shape[0],
            residue_batch["mask"].shape[1],
            residue_batch["mask"].shape[1],
            self.residue_embedder.xt_pair_dist_dim,
        )
        if tuple(self_conditioning_bins.shape) != expected_shape:
            raise ValueError(
                f"self_conditioning_bins shape {tuple(self_conditioning_bins.shape)} "
                f"does not match expected shape {expected_shape}."
            )

        single_repr, pair_repr, mask = self.residue_embedder(
            **residue_batch,
            pairwise_dist_bins=self_conditioning_bins,
        )

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
        
        # add transposed pair_logits to ensure symmetry
        pair_logits = pair_logits + pair_logits.transpose(-1, -2)
        pair_mask = pair_mask | pair_mask.transpose(-1, -2)
        return pair_logits, pair_mask

