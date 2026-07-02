# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn

from src.model.components.primitives import (    
    AdaptiveLayerNorm,
    AdaptiveLayerNormOutputScale,
    Transition,
)
from src.model.components.rotary import RotaryEmbedding


def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


max_neg_value = lambda x: torch.finfo(x.dtype).min


class GatedAttention(nn.Module):
    def __init__(
        self,
        d_model,
        d_head,
        n_heads=8,
        qk_layernorm=True,
        elementwise_attn_output_gate=False,
        qkv_bias=False,
        use_sdpa=True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.inner_dim = n_heads * d_head
        self.scale = self.d_head ** -0.5

        self.qkv_proj = nn.Linear(d_model, self.inner_dim * 3, bias=qkv_bias)
 
        self.g_proj = nn.Linear(d_model, self.inner_dim, bias=qkv_bias)
        self.to_out = nn.Linear(self.inner_dim, d_model)

        if qk_layernorm:
            self.norm_q = nn.LayerNorm(self.inner_dim)
            self.norm_k = nn.LayerNorm(self.inner_dim)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        self.rotary = RotaryEmbedding(self.d_head)
        self._use_sdpa = True

    def _apply_rotary(self, q, k):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, attention_mask=None):
        input_shape = x.shape
        batch_shape = input_shape[:-2]
        L, D = input_shape[-2], input_shape[-1]
        
        # Flatten leading dimensions into batch dimension
        x = x.reshape(-1, L, D)
        B = x.shape[0]
        h = self.n_heads

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        g = self.g_proj(x)

        q = self.norm_q(q)
        k = self.norm_k(k)
        q, k = self._apply_rotary(q, k)

        q = q.reshape(B, L, h, self.d_head).transpose(1, 2)
        k = k.reshape(B, L, h, self.d_head).transpose(1, 2)
        v = v.reshape(B, L, h, self.d_head).transpose(1, 2)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.reshape(B, 1, 1, L).to(dtype=torch.bool)
        # Use SDPA when available to reduce memory traffic and kernel launches.
        # If backend/hardware does not support it, cache fallback to matmul path.
        # out = None
        # if self._use_sdpa:
        sdpa_mask = None
        if key_padding_mask is not None:
            # torch_npu flash attention requires Sq and Skv dimensions in mask.
            sdpa_mask = key_padding_mask.expand(B, 1, L, L)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            is_causal=False,
        )
            # except (RuntimeError, NotImplementedError):
            #     self._use_sdpa = False

        # if out is None:
        #     attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        #     if key_padding_mask is not None:
        #         attn_scores = attn_scores.masked_fill(
        #             ~key_padding_mask, torch.finfo(attn_scores.dtype).min
        #         )
        #     attn_probs = F.softmax(attn_scores, dim=-1)
        #     out = torch.matmul(attn_probs, v)

        out = out.transpose(1, 2).reshape(B, L, self.inner_dim)
        out = self.to_out(out.mul_(g.sigmoid_()))
        out = out.reshape(*batch_shape, L, D)

        return out


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    Code modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: Optional[int] = None,
        **kawrgs  # noqa
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head**-0.5
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e)
        :param beta_mask: local pair mask of shape (b,n,n)
        :param mask: boolean tensor of node adjacencies
        :return:
        """
        assert exists(self.to_bias) or not exists(pair_feats)
        node_feats, h = self.node_norm(node_feats), self.heads
        pair_feats = self.pair_norm(pair_feats) if exists(pair_feats) else None
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)
        b = (
            rearrange(self.to_bias(pair_feats), "b ... h -> b h ...")
            if exists(pair_feats)
            else 0
        )
        q, k, v, g = map(
            lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v, g)
        )
        attn_feats = self._attn(q, k, v, b, mask)
        attn_feats = rearrange(
            torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h
        )
        return self.to_out_node(attn_feats)

    def _attn(self, q, k, v, b, mask: Optional[Tensor]) -> Tensor:
        """Perform attention update"""
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)


class MultiHeadAttention(torch.nn.Module):
    """Typical multi-head self-attention attention using pytorch's module."""

    def __init__(self, dim_token, nheads, dropout=0.0):
        super().__init__()

        # self.to_q = torch.nn.Linear(dim_token, dim_token)
        # self.to_kv = torch.nn.Linear(dim_token, 2 * dim_token, bias=False)

        # self.mha = torch.nn.MultiheadAttention(
        #     embed_dim=dim_token,
        #     num_heads=nheads,
        #     dropout=dropout,
        #     batch_first=True,
        # )
        self.mha = GatedAttention(dim_token, dim_token // nheads, nheads)

    def forward(self, x, mask, cond=None):
        """
        Args:
            x: Input sequence, shape [b, n, dim_token]
            mask: binary mask, shape [b, n]
            cond: Conditioning variables, shape [b, n, dim_cond]
        Returns:
            Updated sequence, shape [b, n, dim_token]
        """
        return self.mha(x, mask)
        # query = self.to_q(x)  # [b, n, dim_token]
        # if cond is not None:
        #     key, value = self.to_kv(cond).chunk(2, dim=-1)  # cross attention
        # else:
        #     key, value = self.to_kv(x).chunk(2, dim=-1)  # Each [b, n, dim_token]
        # return (
        #     self.mha(
        #         query=query,
        #         key=key,
        #         value=value,
        #         key_padding_mask=~mask,  # Indicated what should be ignores with True, that's why the ~
        #         need_weights=False,
        #         is_causal=False,
        #     )[0]
        #     * mask[..., None]
        # )  # [b, n, dim_token]


class MultiHeadAttentionADALN(torch.nn.Module):
    """Typical multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, nheads, dim_cond, dropout=0.0):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        # self.cond_ln = nn.LayerNorm(dim_cond)
        self.mha = MultiHeadAttention(
            dim_token=dim_token, nheads=nheads, dropout=dropout
        )
        self.scale_output = AdaptiveLayerNormOutputScale(
            dim=dim_token, dim_cond=dim_cond
        )

    def forward(self, x, cond, mask, cross_attention=False):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            mask: Binary mask, shape [b, n]
            cross_attention: Whether to perform cross attention
        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        x = self.adaln(x, cond, mask)

        # if cross_attention:
        #     cond = self.cond_ln(cond)
        #     x = self.mha(x, mask, cond)
        # else:
        x = self.mha(x, mask)
            
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class MultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = PairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.scale_output = AdaptiveLayerNormOutputScale(
            dim=dim_token, dim_cond=dim_cond
        )

    def forward(self, x, pair_rep, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class TransitionADALN(torch.nn.Module):
    """Transition layer with adaptive layer norm applied to input and adaptive
    scaling aplied to output."""

    def __init__(self, *, dim, dim_cond, expansion_factor=4):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)
        self.transition = Transition(
            dim=dim, expansion_factor=expansion_factor, layer_norm=False
        )
        self.scale_output = AdaptiveLayerNormOutputScale(dim=dim, dim_cond=dim_cond)

    def forward(self, x, cond, mask, **kwargs):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim]
        """
        x = self.adaln(x, cond, mask)  # [b, n, dim]
        x = self.transition(x, mask)  # [b, n, dim]
        x = self.scale_output(x, cond, mask)  # [b, n, dim]
        return x * mask[..., None]  # [b, n, dim]


class ResidueTransformer(torch.nn.Module):
    def __init__(
        self, 
        dim_token,
        dim_pair,
        dim_cond,
        nheads,
        residual_mha,
        residual_transition,
        use_attn_pair_bias,
        use_qkln,
        dropout=0.0,
        expansion_factor=2,
    ):
        super().__init__()
        self.use_attn_pair_bias = use_attn_pair_bias
        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        if self.use_attn_pair_bias:
            self.mha = MultiHeadBiasedAttentionADALN_MM(
                dim_token=dim_token,
                dim_pair=dim_pair,
                nheads=nheads,
                dim_cond=dim_cond,
                use_qkln=use_qkln,
            )
        else:
            self.mha = MultiHeadAttentionADALN(
                dim_token=dim_token,
                nheads=nheads,
                dim_cond=dim_cond,
                dropout=dropout,
            )

        self.transition = TransitionADALN(
            dim=dim_token, dim_cond=dim_cond, expansion_factor=expansion_factor)
        
    def _apply_mha(self, x, pair_rep, cond, mask):
        if self.use_attn_pair_bias:
            x_attn = self.mha(x, pair_rep, cond, mask)
        else:
            x_attn = self.mha(x, cond, mask)
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]
    
    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]
    
    def forward(self, x, pair_rep, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]
        """
        x = x * mask[..., None]
        x = self._apply_mha(x, pair_rep, cond, mask)
        x = self._apply_transition(x, cond, mask)
        return x * mask[..., None]


class AtomEncoder(torch.nn.Module):
    """Local attention encoder that process vertices or atoms within a residue"""
    def __init__(
        self, 
        dim_atom, 
        dim_cond,
        use_qkln,
        nheads_atom,
        nheads_token, 
        residual_mha, 
        residual_transition, 
        dropout=0.0, 
        expansion_factor=2
    ):
        super().__init__()
        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        # self.mha = MultiHeadAttentionADALN(
        #     dim_token=dim_atom,
        #     nheads=nheads,
        #     dim_cond=dim_atomcond,
        #     dropout=dropout,
        # )

        # self.transition = TransitionADALN(
        #     dim=dim_atom, dim_cond=dim_atomcond, expansion_factor=expansion_factor)

        # local attention
        self.pre_mha_ln = nn.LayerNorm(dim_atom)
        self.mha = MultiHeadAttention(
            dim_token=dim_atom, nheads=nheads_atom, dropout=dropout
        )

        # global attention
        self.global_mha = MultiHeadAttentionADALN(
            dim_token=dim_atom,
            nheads=nheads_token,
            dim_cond=dim_cond,
            dropout=dropout,
        )
        self.global_transition = TransitionADALN(
            dim=dim_atom, dim_cond=dim_cond, expansion_factor=expansion_factor)
        
    def _apply_mha(self, x, mask):
        x_in = x
        x = self.pre_mha_ln(x)
        x_attn = self.mha(x, mask)
        if self.residual_mha:
            x_attn = x_attn + x_in
        return x_attn * mask[..., None]

    def _pooled_global_mha(self, x, cond, mask):
        # first pool x to residue level
        pooled = (x * mask[..., None]).sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        res_mask = mask.sum(dim=-1) > 0

        # apply global attention
        x_attn = self.global_mha(pooled, cond, res_mask)
        x_tr = self.global_transition(x_attn, cond, res_mask)
        if self.residual_transition:
            x_attn = x_attn + x_tr
        else:
            x_attn = x_tr

        # unpool x to atom level
        x = x + x_attn.unsqueeze(-2) * mask.unsqueeze(-1)
        return x
    
    def forward(self, x, cond, mask):
        """
        Args:
            x: Input atom14 representation, shape [b, n, 14, dim_atom]
            cond: Conditioning variables, shape [b, n, dim_cond]
            mask: Binary mask, shape [b, n, 14]
        """
        x = x * mask[..., None]

        # apply residue-local attention in parallel by folding residues into batch
        b, n = x.shape[:2]
        x = rearrange(x, "b n a d -> (b n) a d")
        mask = rearrange(mask, "b n a -> (b n) a")

        x = self._apply_mha(x, mask)

        x = rearrange(x, "(b n) a d -> b n a d", b=b, n=n)
        mask = rearrange(mask, "(b n) a -> b n a", b=b, n=n)

        # apply global attention
        x = self._pooled_global_mha(x, cond, mask)
        return x * mask[..., None]


