from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.model.components.embedder import ResidueEmbedder
from src.model.components.outer_product_mean import OuterProductMean
from src.model.components.primitives import PairTransition
from src.model.components.transformer import ResidueTransformer
from src.model.components.triangle_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from src.model.components.triangle_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

_RESIDUE_LENGTH_KEYS = frozenset({"p1_length", "p2_length"})


# class OuterProductMean(nn.Module):
#     def __init__(
#         self,
#         c_m=256,
#         c_hidden=32,
#         c_z=128,
#     ):
#         super().__init__()
#         self.layernorm = nn.LayerNorm(c_m)
#         self.linear_no_bias = nn.Linear(c_m, c_hidden, bias=False)
#         self.linear_out = nn.Linear(c_hidden ** 2, c_z)

#     def forward(
#         self,
#         token_repr: torch.Tensor,
#         mask: torch.Tensor,
#     ):
#         token_repr = self.layernorm(token_repr)
#         token_repr = self.linear_no_bias(token_repr)
#         if mask is not None:
#             token_repr = token_repr * mask.to(dtype=token_repr.dtype)[..., None]

#         # Project a_i ⊗ a_j without materializing [B, L, L, C, C].
#         pair_dim, inner = self.linear_out.weight.shape[0], token_repr.shape[-1]
#         weight = self.linear_out.weight.view(pair_dim, inner, inner)
#         projected = torch.einsum("pcd,bjd->bjpc", weight, token_repr)
#         outer_product = torch.einsum("bic,bjpc->bijp", token_repr, projected)
#         if self.linear_out.bias is not None:
#             outer_product = outer_product + self.linear_out.bias
#         return outer_product


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
        dim_triangle_hidden=32,
        n_pair_heads=4,
        checkpoint_pair_blocks=True,
        **kwargs,
    ):
        super().__init__()
        if int(dim_token) % int(n_heads) != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must divide dim_token ({dim_token})."
            )
        if int(dim_pair) % int(n_pair_heads) != 0:
            raise ValueError(
                f"n_pair_heads ({n_pair_heads}) must divide dim_pair ({dim_pair})."
            )
        xt_pair_dist_dim = int(kwargs.get("xt_pair_dist_dim", 64))
        num_classes = int(kwargs.get("num_classes", xt_pair_dist_dim))
        if num_classes != xt_pair_dist_dim:
            raise ValueError(
                f"num_classes ({num_classes}) must equal "
                f"xt_pair_dist_dim ({xt_pair_dist_dim})."
            )
        self.dim_token = dim_token
        self.dim_pair = dim_pair
        self.checkpoint_pair_blocks = bool(checkpoint_pair_blocks)

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
                c_m=dim_token,
                c_z=dim_pair,
                c_hidden=dim_opm_inner,
            ) for _ in range(n_layers)
        ])

        self.triangle_multiplication_outgoing = nn.ModuleList([
            TriangleMultiplicationOutgoing(
                c_z=dim_pair,
                c_hidden=dim_triangle_hidden,
            ) for _ in range(n_layers)
        ])

        self.triangle_multiplication_incoming = nn.ModuleList([
            TriangleMultiplicationIncoming(
                c_z=dim_pair,
                c_hidden=dim_triangle_hidden,
            ) for _ in range(n_layers)
        ])

        self.triangle_attention_starting = nn.ModuleList([
            TriangleAttentionStartingNode(
                c_in=dim_pair,
                c_hidden=dim_pair,
                no_heads=n_pair_heads,
                dropout=dropout,
            ) for _ in range(n_layers)
        ])

        self.triangle_attention_ending = nn.ModuleList([
            TriangleAttentionEndingNode(
                c_in=dim_pair,
                c_hidden=dim_pair,
                no_heads=n_pair_heads,
                dropout=dropout,
            ) for _ in range(n_layers)
        ])

        self.pair_blocks = nn.ModuleList([
            PairTransition(
                dim=dim_pair,
                expansion_factor=expansion_factor,
                layer_norm=True,
            ) for _ in range(n_layers)
        ])

        # prediction heads
        self.pair_out_layernorm = nn.LayerNorm(dim_pair)
        self.pair_out_linear = nn.Linear(dim_pair, num_classes)
        self.residue_bind_norm = nn.LayerNorm(dim_token)
        self.residue_bind_linear = nn.Linear(dim_token, 1)
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
        for module in (
            *self.triangle_multiplication_outgoing,
            *self.triangle_multiplication_incoming,
        ):
            nn.init.zeros_(module.linear_z.weight)
            if module.linear_z.bias is not None:
                nn.init.zeros_(module.linear_z.bias)
        for module in (
            *self.triangle_attention_starting,
            *self.triangle_attention_ending,
        ):
            nn.init.zeros_(module.mha.to_out.weight)
            if module.mha.to_out.bias is not None:
                nn.init.zeros_(module.mha.to_out.bias)

    def forward(
        self,
        residue_batch: Dict[str, torch.Tensor],
        self_conditioning_bins: torch.Tensor | None = None,
        recycle_rounds: int = 1,
        chunk_size: Optional[int] = None,
    ):
        recycle_rounds = max(1, int(recycle_rounds))
        chunk_size = _normalize_chunk_size(chunk_size)
        if self_conditioning_bins is None:
            self_conditioning_bins = self._init_self_conditioning(residue_batch)

        recycled_bins = self_conditioning_bins
        for _ in range(recycle_rounds - 1):
            with torch.no_grad():
                logits, _, _ = self._forward_once(
                    residue_batch,
                    recycled_bins,
                    chunk_size=chunk_size,
                )
                recycled_bins = torch.softmax(logits, dim=-1).detach()

        return self._forward_once(
            residue_batch,
            recycled_bins,
            chunk_size=chunk_size,
        )

    def _init_self_conditioning(
        self,
        residue_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.residue_embedder.calvados_pair_energies(residue_batch["residue_type"])

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
            single_repr.unsqueeze(-3),
            mask.unsqueeze(-2),
            chunk_size=chunk_size,
        ) * pair_mask_float[..., None]
        pair_repr = pair_repr + self.triangle_multiplication_outgoing[layer_index](
            pair_repr,
            pair_mask_float,
        )
        pair_repr = pair_repr + self.triangle_multiplication_incoming[layer_index](
            pair_repr,
            pair_mask_float,
        )
        pair_repr = pair_repr + self.triangle_attention_starting[layer_index](
            pair_repr,
            pair_mask_float,
            chunk_size=chunk_size,
        )
        pair_repr = pair_repr + self.triangle_attention_ending[layer_index](
            pair_repr,
            pair_mask_float,
            chunk_size=chunk_size,
        )
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

    def _forward_once(
        self,
        residue_batch: Dict[str, torch.Tensor],
        self_conditioning_bins: torch.Tensor,
        chunk_size: Optional[int] = None,
    ):
        embedder_inputs = {
            key: value
            for key, value in residue_batch.items()
            if key not in _RESIDUE_LENGTH_KEYS
        }
        single_repr, pair_repr, mask = self.residue_embedder(
            **embedder_inputs,
            pairwise_dist_bins=self_conditioning_bins,
        )
        mask = mask.to(dtype=torch.bool)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair_mask_float = pair_mask.to(dtype=pair_repr.dtype)
        chunk_size = _normalize_chunk_size(chunk_size)

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
        pair_mask = pair_mask | pair_mask.transpose(-1, -2)
        
        residue_logits = self.residue_bind_linear(
            self.residue_bind_norm(single_repr)
        ).squeeze(-1)
        residue_logits = residue_logits.masked_fill(~mask, 0.0)
        return pair_logits, residue_logits, pair_mask


def _normalize_chunk_size(chunk_size: Optional[int]) -> Optional[int]:
    if chunk_size is None:
        return None
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        return None
    return chunk_size
