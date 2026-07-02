import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# Adapted from frameflow code
def get_index_embedding(indices, edim, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of type integer, shape either [n] or [b, n].
        edim: dimension of the embeddings to create.
        max_len: maximum length.

    Returns:
        positional embedding of shape either [n, edim] or [b, n, edim]
    """
    # indices [n] of [b, n]
    K = torch.arange(edim // 2, device=indices.device)  # [edim / 2]

    if len(indices.shape) == 1:  # [n]
        K = K[None, ...]
    elif len(indices.shape) == 2:  # [b, n]
        K = K[None, None, ...]

    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K / edim))
    ).to(indices.device)
    # [n, 1] / [1, edim/2] -> [n, edim/2] or [b, n, 1] / [1, 1, edim/2] -> [b, n, edim/2]
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K / edim))
    ).to(indices.device)
    pos_embedding = torch.cat(
        [pos_embedding_sin, pos_embedding_cos], axis=-1
    )  # [n, edim]
    return pos_embedding


def relative_position_encoding(
    token_index: torch.Tensor,
    residue_index: torch.Tensor,
    chain_index: torch.Tensor,
    mask: torch.Tensor,
    r_max: int,
):
    pair_mask = mask.to(dtype=torch.bool)
    pair_mask = pair_mask[..., None] & pair_mask[..., None, :]

    same_chain_mask = (chain_index[:, :, None] == chain_index[:, None, :]).long()
    rel_pos_chain = F.one_hot(same_chain_mask, 2)

    d_residue = torch.clip(
        input=residue_index[:, :, None] - residue_index[:, None, :] + r_max,
        min=0, max=2 * r_max,
    ) * same_chain_mask + (1 - same_chain_mask) * (2 * r_max + 1)
    rel_pos_residue = F.one_hot(d_residue.long(), 2 * (r_max + 1))

    # add a global position encoding
    d_token = torch.clip(
        input=token_index[:, :, None] - token_index[:, None, :] + r_max,
        min=0, max=2 * r_max,
    )
    rel_pos_token = F.one_hot(d_token.long(), 2 * (r_max + 1))

    rel_pos = torch.cat([rel_pos_token, rel_pos_residue, rel_pos_chain], dim=-1).float()
    return rel_pos * pair_mask[..., None].to(dtype=rel_pos.dtype)  # [b, n, n, 2 + 4 * (r_max + 1)]


def bin_pairwise_distances(x, min_dist, max_dist, dim):
    """
    Takes coordinates and bins the pairwise distances.

    Args:
        x: Coordinates of shape [b, n, 3]
        min_dist: Right limit of first bin
        max_dist: Left limit of last bin
        dim: Dimension of the final one hot vectors

    Returns:
        Tensor of shape [b, n, n, dim] consisting of one-hot vectors
    """
    pair_dists_nm = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # [b, n, n]
    bin_limits = torch.linspace(
        min_dist, max_dist, dim - 1, device=x.device
    )  # Open left and right
    return bin_and_one_hot(pair_dists_nm, bin_limits)  # [b, n, n, pair_dist_dim]


def bin_and_one_hot(tensor, bin_limits):
    """
    Converts a tensor of shape [*] to a tensor of shape [*, d] using the given bin limits.

    Args:
        tensor (Tensor): Input tensor of shape [*]
        bin_limits (Tensor): bin limits [l1, l2, ..., l_{d-1}]. d-1 limits define
            d-2 bins, and the first one is <l1, the last one is >l_{d-1}, giving a total of d bins.

    Returns:
        torch.Tensor: Output tensor of shape [*, d] where d = len(bin_limits) + 1
    """
    bin_indices = torch.bucketize(tensor, bin_limits)
    return torch.nn.functional.one_hot(bin_indices, len(bin_limits) + 1) * 1.0


class ResidueEmbedder(nn.Module):
    def __init__(
        self,
        dim_token: int,
        dim_pair: int,
        **kwargs,
    ):
        # residue features contain:
        # 1. plm embedding (single)
        # 2. residue type (single)
        # 3. residue index (single)
        # 4. chain break per residue (single)
        # 5. pairwise distance (pair)
        # 6. relative position encoding (pair)

        super().__init__()
        self.dim_token = dim_token
        self.dim_pair = dim_pair

        self.r_max = kwargs.get('r_max', 32)
        self.xt_pair_dist_min = kwargs.get('xt_pair_dist_min', 1)
        self.xt_pair_dist_max = kwargs.get('xt_pair_dist_max', 33)
        self.xt_pair_dist_dim = kwargs.get('xt_pair_dist_dim', 64)  # 0.5 A per bin
        self.residue_index_dim = kwargs.get('idx_emb_dim', 128)
        self.plm_in_dim = kwargs.get('plm_in_dim', 1280)
        self.plm_out_dim = kwargs.get('plm_out_dim', 256)
        
        self.plm_embedder = nn.Linear(self.plm_in_dim, self.plm_out_dim)
        
        single_dim = self.plm_out_dim + 20 + self.residue_index_dim + 1
        self.single_out = nn.Linear(single_dim, self.dim_token, bias=False)
    
        pair_dim = self.xt_pair_dist_dim + 2 + 4 * (self.r_max + 1)
        self.pair_out = nn.Linear(pair_dim, self.dim_pair, bias=False)

    def forward(
        self,
        plm_emb: torch.Tensor,
        residue_type: torch.Tensor,
        residue_index: torch.Tensor,
        residue_position: torch.Tensor,
        chain_index: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ):
        plm_emb = self.plm_embedder(plm_emb)
        
        residue_type_emb = F.one_hot(residue_type, num_classes=20) * 1.0
        residue_index_emb = get_index_embedding(
            residue_index, self.residue_index_dim, max_len=2056)
        
        chain_break_emb = (chain_index[:, 1:] != chain_index[:, :-1]).float()  # [b, n-1]
        chain_break_emb = F.pad(chain_break_emb, (0, 1), mode="constant", value=0.0)[..., None]  # [b, n, 1]
        
        token_index = torch.arange(
            residue_index.shape[1], device=residue_index.device, dtype=residue_index.dtype
        ).unsqueeze(0).expand_as(residue_index)
        rel_pos_emb = relative_position_encoding(
            token_index, residue_index, chain_index, mask, self.r_max
        )
        pairwise_dist_emb = bin_pairwise_distances(
            residue_position, 
            self.xt_pair_dist_min,
            self.xt_pair_dist_max, 
            self.xt_pair_dist_dim)

        single_repr = self.single_out(
            torch.cat([plm_emb, residue_type_emb, residue_index_emb, chain_break_emb], dim=-1)
            * mask[..., None]
        )
        pair_mask = mask[..., None] * mask[..., None, :]
        pair_repr = self.pair_out(
            torch.cat([pairwise_dist_emb, rel_pos_emb], dim=-1)
            * pair_mask[..., None]
        )
        return single_repr, pair_repr, mask


class AtomEmbedder(nn.Module):
    def __init__(
        self,
        dim_atom: int,
        **kwargs,
    ):
        # atom features contain:
        # 1. atom name (single)
        # 2. atom element (single)
        # 3. atom charge (single)
        # 4. atom position (single)
        super().__init__()
        self.atom_name_vocab_size = int(kwargs.get("atom_name_vocab_size", 64))
        self.linear_out = nn.Linear(3 + 1 + 4 * self.atom_name_vocab_size, dim_atom, bias=False)

    def forward(
        self,
        atom_name: torch.Tensor,
        atom_charge: torch.Tensor,
        atom_position: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ):
        if atom_charge.dim() == atom_position.dim() - 1:
            atom_charge = atom_charge.unsqueeze(-1)

        if atom_name.dim() == atom_position.dim():
            atom_name = F.one_hot(
                atom_name.long().clamp_min(0).clamp_max(self.atom_name_vocab_size - 1),
                num_classes=self.atom_name_vocab_size,
            )
        elif atom_name.dim() != atom_position.dim() + 1:
            raise ValueError(
                "atom_name must be either index-encoded (..., 4) or one-hot (..., 4, vocab)."
            )

        atom_name = atom_name.to(dtype=atom_position.dtype)
        atom_name_feat = atom_name.reshape(*atom_position.shape[:-1], -1)

        atom_repr = self.linear_out(
            torch.cat([atom_position, atom_charge, atom_name_feat], dim=-1)
            * mask[..., None]
        )
        return atom_repr, mask
