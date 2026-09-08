from typing import Dict, Optional

import torch
import torch.nn as nn

from src.model.components.transformer import (
    ResidueTransformer,
    Transition,
)
from src.model.components.embedder import ResidueEmbedder
from src.model.components.triangle_update import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming
# from src.model.components.outer_product_mean import OuterProductMean

_RESIDUE_LENGTH_KEYS = frozenset({"p1_length", "p2_length"})


class OuterProductMean(nn.Module):
    def __init__(
        self,
        c_m=256,
        c_hidden=32,
        c_z=128,
    ):
        super().__init__()
        self.layernorm = nn.LayerNorm(c_m)
        self.linear_no_bias = nn.Linear(c_m, c_hidden, bias=False)
        self.linear_out = nn.Linear(c_hidden ** 2, c_z)

    def forward(
        self,
        token_repr: torch.Tensor,
        mask: torch.Tensor,
    ):
        token_repr = self.layernorm(token_repr)
        token_repr = self.linear_no_bias(token_repr)
        if mask is not None:
            token_repr = token_repr * mask.to(dtype=token_repr.dtype)[..., None]

        # Project a_i ⊗ a_j without materializing [B, L, L, C, C].
        pair_dim, inner = self.linear_out.weight.shape[0], token_repr.shape[-1]
        weight = self.linear_out.weight.view(pair_dim, inner, inner)
        projected = torch.einsum("pcd,bjd->bjpc", weight, token_repr)
        outer_product = torch.einsum("bic,bjpc->bijp", token_repr, projected)
        if self.linear_out.bias is not None:
            outer_product = outer_product + self.linear_out.bias
        return outer_product


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

        self.pair_blocks = nn.ModuleList([
            Transition(
                dim=dim_pair,
                expansion_factor=expansion_factor,
                layer_norm=True,
            ) for _ in range(n_layers)
        ])

        # prediction head
        self.pair_out_layernorm = nn.LayerNorm(dim_pair)
        self.pair_out_linear = nn.Linear(dim_pair, num_classes)
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

    def forward(
        self,
        residue_batch: Dict[str, torch.Tensor],
        self_conditioning_bins: torch.Tensor | None = None,
        recycle_rounds: int = 1,
    ):
        recycle_rounds = max(1, int(recycle_rounds))
        if self_conditioning_bins is None:
            self_conditioning_bins = self._init_self_conditioning(residue_batch)

        recycled_bins = self_conditioning_bins
        for _ in range(recycle_rounds - 1):
            with torch.no_grad():
                logits, _ = self._forward_once(residue_batch, recycled_bins)
                recycled_bins = torch.softmax(logits, dim=-1).detach()

        return self._forward_once(residue_batch, recycled_bins)

    def _init_self_conditioning(
        self,
        residue_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.residue_embedder.calvados_pair_energies(residue_batch["residue_type"])

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

        # main trunk
        for i in range(self.n_layers):
            # first update single representation
            single_repr = self.residue_blocks[i](
                single_repr,
                pair_repr,
                single_repr, # conditioning by itself
                mask,
            )
            # outer product mean
            pair_repr = pair_repr + self.outer_product_mean[i](
                single_repr.unsqueeze(-3),  # [*, 1, N_res, C_m]
                mask.unsqueeze(-2),  # [*, 1, N_res]
                chunk_size=chunk_size,
            ) * pair_mask_float[..., None]
            # triangle multiplication
            pair_repr = pair_repr + self.triangle_multiplication_outgoing[i](
                pair_repr,
                pair_mask_float,
            )
            pair_repr = pair_repr + self.triangle_multiplication_incoming[i](
                pair_repr,
                pair_mask_float,
            )
            # pair transition
            pair_repr = pair_repr + self.pair_blocks[i](
                pair_repr,
                pair_mask_float,
            )

        # output head
        pair_logits = self.pair_out_linear(self.pair_out_layernorm(pair_repr))
        
        # add transposed pair_logits to ensure symmetry
        pair_logits = pair_logits + pair_logits.transpose(-2, -3)
        pair_logits = pair_logits / 2
        pair_mask = pair_mask | pair_mask.transpose(-1, -2)
        return pair_logits, pair_mask

