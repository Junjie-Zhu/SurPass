import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.calvados.calvados_functions import (
    compute_ashbaugh_hatch,
    compute_yukawa,
    load_calvados_model,
)
from src.common.mpipi.mpipi_functions import coulomb, load_mpipi_arrays, wang_frenkel
from src.common.residue_constants import restypes, restypes_with_x
from src.model.loss import distance_bin_centers

NUM_RESIDUE_TYPES = len(restypes_with_x)
N_ENERGY_CHANNELS = 4
k_BOLTZMANN = 0.0083144721  # kJ/mol/K
CALVADOS_TEMP = 298.15
_MIN_CALVADOS_DISTANCE_NM = 1e-6


def normalize_pair_energy(energy_kt):
    """asinh(E/kT) keeps attractive wells nearly linear and compresses the repulsive wall."""
    return np.arcsinh(np.asarray(energy_kt, dtype=np.float64))


def _calvados_energy_channels(centers_a: np.ndarray, temp: float) -> tuple[np.ndarray, np.ndarray]:
    """Yukawa and Ashbaugh-Hatch, each asinh(E/kT), shape [20, 20, bins]."""
    model = load_calvados_model(
        version="CALVADOS2",
        salt=0.150,
        pH=7.4,
        temp=temp,
    )
    r_nm = np.maximum(np.asarray(centers_a, dtype=np.float64) * 0.1, _MIN_CALVADOS_DISTANCE_NM)
    aa_names = list(restypes)
    sigma = model.sigmamap.loc[aa_names, aa_names].to_numpy(dtype=np.float64)
    lam = model.lambdamap.loc[aa_names, aa_names].to_numpy(dtype=np.float64)
    yukawa_q = model.yukawa_eps.loc[aa_names, aa_names].to_numpy(dtype=np.float64)

    r_b = r_nm[None, None, :]
    yukawa = compute_yukawa(
        r_b,
        yukawa_q[:, :, None],
        model.yukawa_kappa,
        yukawa_r_cut=model.yukawa_r_cut,
    )
    ashbaugh = compute_ashbaugh_hatch(
        r_b,
        sigma[:, :, None],
        lam[:, :, None],
        cutoff=model.cutoff,
        lj_eps=model.lj_eps,
    )
    yukawa = np.where(r_b > model.yukawa_r_cut, 0.0, yukawa)
    ashbaugh = np.where(r_b > model.cutoff, 0.0, ashbaugh)
    kt = k_BOLTZMANN * float(temp)
    return normalize_pair_energy(yukawa / kt), normalize_pair_energy(ashbaugh / kt)


def _mpipi_energy_channels(centers_a: np.ndarray, temp: float) -> tuple[np.ndarray, np.ndarray]:
    """Coulomb and Wang-Frenkel, each asinh(E/kT), shape [20, 20, bins]."""
    params = load_mpipi_arrays(list(restypes))
    r_b = np.asarray(centers_a, dtype=np.float64)[None, None, :]
    electrostatic = coulomb(
        params["charge"][:, None, None],
        params["charge"][None, :, None],
        r_b,
    )
    lennard_jones = wang_frenkel(
        r_b,
        params["sigma"][:, :, None],
        params["epsilon"][:, :, None],
        params["mu"][:, :, None],
        params["nu"][:, :, None],
    )
    kt = k_BOLTZMANN * float(temp)
    return (
        normalize_pair_energy(electrostatic / kt),
        normalize_pair_energy(lennard_jones / kt),
    )


def _build_pair_bin_energy_table(
    min_dist: float,
    max_dist: float,
    dim: int,
    temp: float,
) -> torch.Tensor:
    """Four asinh(E/kT) channels at each shared distance-bin center.

    Channel order is Calvados Yukawa, Calvados Ashbaugh-Hatch, Mpipi Coulomb,
    Mpipi Wang-Frenkel. Shape [n_aa, n_aa, 4, bins]. Unknown residues stay zero.
    """
    centers_a = distance_bin_centers(min_dist, max_dist, dim).detach().cpu().numpy()
    channels = (
        *_calvados_energy_channels(centers_a, temp),
        *_mpipi_energy_channels(centers_a, temp),
    )
    table = np.zeros(
        (NUM_RESIDUE_TYPES, NUM_RESIDUE_TYPES, N_ENERGY_CHANNELS, int(dim)),
        dtype=np.float32,
    )
    n_canonical = len(restypes)
    for index, channel in enumerate(channels):
        table[:n_canonical, :n_canonical, index] = np.asarray(channel, dtype=np.float32)
    return torch.from_numpy(table)


@lru_cache(maxsize=8)
def _cached_pair_bin_energy_table(
    min_dist: float,
    max_dist: float,
    dim: int,
    temp: float,
) -> torch.Tensor:
    return _build_pair_bin_energy_table(min_dist, max_dist, dim, temp)


def build_pair_bin_energy_table(
    min_dist: float,
    max_dist: float,
    dim: int,
    *,
    temp: float = CALVADOS_TEMP,
) -> torch.Tensor:
    """Return a cloned residue-pair energy lookup. Physics is computed once per key."""
    return _cached_pair_bin_energy_table(
        float(min_dist),
        float(max_dist),
        int(dim),
        float(temp),
    ).clone()


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


def within_chain_index(chain_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sequential position 0..L-1 inside each chain, ignoring padding.

    PDB residue numbers have gaps and are not what inference provides.
    Both sides use this count of earlier valid residues on the same chain.
    """
    length = int(chain_index.shape[-1])
    position = torch.arange(length, device=chain_index.device)
    valid = mask.to(dtype=torch.bool)
    same_chain = chain_index[:, :, None] == chain_index[:, None, :]
    both_valid = valid[:, :, None] & valid[:, None, :]
    earlier = position.view(1, 1, length) < position.view(1, length, 1)
    return (same_chain & both_valid & earlier).to(dtype=torch.long).sum(dim=-1)


def relative_position_encoding(
    token_index: torch.Tensor,
    residue_index: torch.Tensor,
    chain_index: torch.Tensor,
    mask: torch.Tensor,
    r_max: int,
    row_start: int = 0,
    row_end: int | None = None,
):
    """Relative position features for query rows ``[row_start, row_end)``.

    The key axis stays the full sequence, so concatenating row chunks reproduces
    the full ``[B, L, L, C]`` encoding without building it in one allocation.
    """
    if row_end is None:
        row_end = int(token_index.shape[1])
    query_token = token_index[:, row_start:row_end]
    query_residue = residue_index[:, row_start:row_end]
    query_chain = chain_index[:, row_start:row_end]
    query_mask = mask[:, row_start:row_end].to(dtype=torch.bool)
    key_mask = mask.to(dtype=torch.bool)
    pair_mask = query_mask[..., None] & key_mask[..., None, :]

    same_chain_mask = (query_chain[:, :, None] == chain_index[:, None, :]).long()
    rel_pos_chain = F.one_hot(same_chain_mask.clamp(0, 1), 2)

    rel_pos_classes = 2 * (int(r_max) + 1)
    d_residue = torch.clip(
        input=query_residue[:, :, None] - residue_index[:, None, :] + r_max,
        min=0, max=2 * r_max,
    ) * same_chain_mask + (1 - same_chain_mask) * (2 * r_max + 1)
    rel_pos_residue = F.one_hot(
        d_residue.long().clamp(0, rel_pos_classes - 1), rel_pos_classes
    )

    # add a global position encoding
    d_token = torch.clip(
        input=query_token[:, :, None] - token_index[:, None, :] + r_max,
        min=0, max=2 * r_max,
    )
    rel_pos_token = F.one_hot(
        d_token.long().clamp(0, rel_pos_classes - 1), rel_pos_classes
    )

    rel_pos = torch.cat([rel_pos_token, rel_pos_residue, rel_pos_chain], dim=-1).float()
    return rel_pos * pair_mask[..., None].to(dtype=rel_pos.dtype)  # [b, rows, n, 2 + 4 * (r_max + 1)]


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
        pair_row_chunk_size: int = 32,
        **kwargs,
    ):
        # residue features contain:
        # 1. plm embedding (single)
        # 2. residue type (single)
        # 3. within-chain sequential index (single)
        # 4. chain break per residue (single)
        # 5. four binned pair energies (pair; sequence, available at test time):
        #    Calvados Yukawa, Calvados Ashbaugh-Hatch, Mpipi Coulomb, Mpipi Wang-Frenkel
        # 6. relative position encoding (pair), using the same sequential index

        super().__init__()
        self.dim_token = dim_token
        self.dim_pair = dim_pair
        if int(pair_row_chunk_size) < 1:
            raise ValueError(
                f"pair_row_chunk_size must be positive, got {pair_row_chunk_size}."
            )
        self.pair_row_chunk_size = int(pair_row_chunk_size)

        self.r_max = kwargs.get('r_max', 32)
        self.distance_bin_start = float(kwargs.get("distance_bin_start", 2.0))
        self.distance_bin_end = float(kwargs.get("distance_bin_end", 22.0))
        self.distance_bin_count = int(kwargs.get("distance_bin_count", 64))
        if self.distance_bin_count < 2:
            raise ValueError(
                f"distance_bin_count must be at least 2, got {self.distance_bin_count}."
            )
        self.plm_in_dim = kwargs.get('plm_in_dim', 1280)
        self.plm_out_dim = kwargs.get('plm_out_dim', 256)
        self.index_emb_dim = int(kwargs.get("index_emb_dim", 128))
        if self.index_emb_dim <= 0 or self.index_emb_dim % 2 != 0:
            raise ValueError(
                f"index_emb_dim must be a positive even integer, got {self.index_emb_dim}."
            )
        
        self.plm_embedder = nn.Linear(self.plm_in_dim, self.plm_out_dim)
        
        single_dim = self.plm_out_dim + NUM_RESIDUE_TYPES + 1 + self.index_emb_dim
        self.single_out = nn.Linear(single_dim, self.dim_token)

        pair_dim = N_ENERGY_CHANNELS * self.distance_bin_count + 2 + 4 * (self.r_max + 1)
        self.pair_out = nn.Linear(pair_dim, self.dim_pair)
        self.register_buffer(
            "pair_bin_energies",
            build_pair_bin_energy_table(
                self.distance_bin_start,
                self.distance_bin_end,
                self.distance_bin_count,
            ),
            persistent=False,
        )

    def pair_energies(
        self,
        residue_type: torch.Tensor,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> torch.Tensor:
        aa = residue_type.long().clamp(0, NUM_RESIDUE_TYPES - 1)
        if row_end is None:
            row_end = int(aa.shape[1])
        return self.pair_bin_energies[aa[:, row_start:row_end, None], aa[:, None, :]]

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
        if residue_index.shape[:2] != chain_index.shape[:2]:
            raise ValueError(
                f"residue_index shape {tuple(residue_index.shape)} does not match "
                f"chain_index shape {tuple(chain_index.shape)}."
            )

        residue_type = residue_type.long().clamp(0, NUM_RESIDUE_TYPES - 1)
        residue_type_emb = F.one_hot(residue_type, num_classes=NUM_RESIDUE_TYPES).to(
            dtype=plm_emb.dtype
        )

        chain_break_emb = (chain_index[:, 1:] != chain_index[:, :-1]).float()  # [b, n-1]
        chain_break_emb = F.pad(chain_break_emb, (0, 1), mode="constant", value=0.0)[..., None]  # [b, n, 1]

        seq_index = within_chain_index(chain_index, mask)
        index_emb = get_index_embedding(seq_index, self.index_emb_dim).to(dtype=plm_emb.dtype)
        token_index = torch.arange(
            seq_index.shape[1], device=seq_index.device, dtype=seq_index.dtype
        ).unsqueeze(0).expand_as(seq_index)

        single_repr = self.single_out(
            torch.cat([plm_emb, residue_type_emb, chain_break_emb, index_emb], dim=-1)
            * mask[..., None]
        )
        pair_repr = self._pair_repr_rows(
            residue_type,
            token_index,
            seq_index,
            chain_index,
            mask,
            dtype=residue_position.dtype,
        )
        return single_repr, pair_repr, mask

    def _pair_repr_rows(
        self,
        residue_type: torch.Tensor,
        token_index: torch.Tensor,
        seq_index: torch.Tensor,
        chain_index: torch.Tensor,
        mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        length = int(mask.shape[1])
        chunk = self.pair_row_chunk_size

        def rows(start: int, end: int) -> torch.Tensor:
            rel_pos_emb = relative_position_encoding(
                token_index,
                seq_index,
                chain_index,
                mask,
                self.r_max,
                row_start=start,
                row_end=end,
            )
            pairwise_dist_emb = self.pair_energies(
                residue_type,
                row_start=start,
                row_end=end,
            ).to(device=rel_pos_emb.device, dtype=dtype).flatten(start_dim=-2)
            row_mask = mask[:, start:end, None] * mask[:, None, :]
            return self.pair_out(
                torch.cat([pairwise_dist_emb, rel_pos_emb], dim=-1)
                * row_mask[..., None]
            )

        if chunk >= length:
            return rows(0, length)
        return torch.cat(
            [rows(start, min(start + chunk, length)) for start in range(0, length, chunk)],
            dim=1,
        )


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
