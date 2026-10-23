"""Residue and rectangular inter-chain pair embeddings.

The model never materializes a square pair-input feature tensor. Expensive
physics curves are projected on the 21x21 residue table first and only then
looked up for the L1xL2 inter-chain block.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

from src.common.calvados.calvados_functions import (
    compute_ashbaugh_hatch,
    compute_yukawa,
    load_calvados_model,
)
from src.common.mpipi.mpipi_functions import coulomb, load_mpipi_arrays, wang_frenkel
from src.common.residue_constants import restypes, restypes_with_x
from src.model.loss import distance_bin_centers
from src.model.components.transformer import SwiGLU, Transition

NUM_RESIDUE_TYPES = len(restypes_with_x)
N_ENERGY_CHANNELS = 4
K_BOLTZMANN = 0.0083144721
CALVADOS_TEMP = 298.15


def normalize_pair_energy(value: np.ndarray) -> np.ndarray:
    return np.arcsinh(np.asarray(value, dtype=np.float64))


def _calvados_channels(
    centers_a: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = load_calvados_model(temp=temperature)
    distance_nm = np.maximum(
        np.asarray(centers_a, dtype=np.float64) * 0.1,
        1e-6,
    )
    names = list(restypes)
    sigma = model.sigmamap.loc[names, names].to_numpy(dtype=np.float64)
    lam = model.lambdamap.loc[names, names].to_numpy(dtype=np.float64)
    charge = model.yukawa_eps.loc[names, names].to_numpy(dtype=np.float64)
    r = distance_nm[None, None, :]
    yukawa = compute_yukawa(
        r,
        charge[:, :, None],
        model.yukawa_kappa,
        yukawa_r_cut=model.yukawa_r_cut,
    )
    ashbaugh = compute_ashbaugh_hatch(
        r,
        sigma[:, :, None],
        lam[:, :, None],
        cutoff=model.cutoff,
        lj_eps=model.lj_eps,
    )
    yukawa = np.where(r > model.yukawa_r_cut, 0.0, yukawa)
    ashbaugh = np.where(r > model.cutoff, 0.0, ashbaugh)
    kt = K_BOLTZMANN * float(temperature)
    return normalize_pair_energy(yukawa / kt), normalize_pair_energy(ashbaugh / kt)


def _mpipi_channels(
    centers_a: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    params = load_mpipi_arrays(list(restypes))
    r = np.asarray(centers_a, dtype=np.float64)[None, None, :]
    electrostatic = coulomb(
        params["charge"][:, None, None],
        params["charge"][None, :, None],
        r,
    )
    contact = wang_frenkel(
        r,
        params["sigma"][:, :, None],
        params["epsilon"][:, :, None],
        params["mu"][:, :, None],
        params["nu"][:, :, None],
    )
    kt = K_BOLTZMANN * float(temperature)
    return normalize_pair_energy(electrostatic / kt), normalize_pair_energy(contact / kt)


@lru_cache(maxsize=8)
def _cached_energy_table(
    minimum: float,
    maximum: float,
    count: int,
    temperature: float,
) -> torch.Tensor:
    centers = distance_bin_centers(minimum, maximum, count).numpy()
    channels = (
        *_calvados_channels(centers, temperature),
        *_mpipi_channels(centers, temperature),
    )
    table = np.zeros(
        (NUM_RESIDUE_TYPES, NUM_RESIDUE_TYPES, N_ENERGY_CHANNELS, int(count)),
        dtype=np.float32,
    )
    canonical = len(restypes)
    for index, channel in enumerate(channels):
        table[:canonical, :canonical, index] = channel
    return torch.from_numpy(table)


def build_pair_bin_energy_table(
    minimum: float,
    maximum: float,
    count: int,
    *,
    temperature: float = CALVADOS_TEMP,
    allow_missing: bool = False,
) -> torch.Tensor:
    """Build the fixed residue-pair physics table.

    Missing external parameter files are an explicit error in production.
    Tests and lightweight installations may request a zero initialization.
    """
    try:
        return _cached_energy_table(
            float(minimum),
            float(maximum),
            int(count),
            float(temperature),
        ).clone()
    except FileNotFoundError:
        if not allow_missing:
            raise
        warnings.warn(
            "CALVADOS/Mpipi parameters are unavailable; physics features are "
            "zero-initialized.",
            RuntimeWarning,
        )
        return torch.zeros(
            NUM_RESIDUE_TYPES,
            NUM_RESIDUE_TYPES,
            N_ENERGY_CHANNELS,
            int(count),
        )


def sinusoidal_position_embedding(
    position: torch.Tensor,
    dimension: int,
    *,
    maximum_period: int = 4096,
) -> torch.Tensor:
    if dimension <= 0 or dimension % 2:
        raise ValueError("position embedding dimension must be a positive even integer")
    frequency = torch.arange(
        dimension // 2,
        device=position.device,
        dtype=torch.float32,
    )
    frequency = math.pi / (
        float(maximum_period) ** (2.0 * frequency / float(dimension))
    )
    angle = position.to(dtype=torch.float32)[..., None] * frequency
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)


def within_chain_index(
    chain_index: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return 0-based positions independently inside each chain."""
    length = int(chain_index.shape[-1])
    token = torch.arange(length, device=chain_index.device)
    valid = mask.to(dtype=torch.bool)
    same_chain = chain_index[:, :, None] == chain_index[:, None, :]
    earlier = token.view(1, 1, length) < token.view(1, length, 1)
    both_valid = valid[:, :, None] & valid[:, None, :]
    return (same_chain & earlier & both_valid).sum(dim=-1)


def _normalized_position(mask: torch.Tensor) -> torch.Tensor:
    length = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    index = torch.arange(mask.shape[-1], device=mask.device).view(1, -1)
    denominator = (length - 1).clamp_min(1)
    return (index / denominator).to(dtype=torch.float32) * mask


class ResidueEmbedder(nn.Module):
    """Embed residues and construct only the L1xL2 inter-chain pair block."""

    def __init__(
        self,
        dim_token: int,
        dim_pair: int,
        *,
        plm_in_dim: int = 2560,
        plm_out_dim: int = 384,
        index_emb_dim: int = 64,
        distance_bin_start: float = 2.0,
        distance_bin_end: float = 22.0,
        distance_bin_count: int = 16,
        allow_missing_physics: bool = False,
        **_: object,
    ):
        super().__init__()
        self.dim_token = int(dim_token)
        self.dim_pair = int(dim_pair)
        self.distance_bin_count = int(distance_bin_count)
        self.index_emb_dim = int(index_emb_dim)

        aa_dim = min(64, self.dim_token // 4)
        self.plm_projection = nn.Linear(int(plm_in_dim), int(plm_out_dim))
        self.residue_embedding = nn.Embedding(NUM_RESIDUE_TYPES, aa_dim)
        single_input = int(plm_out_dim) + aa_dim + self.index_emb_dim
        self.single_projection = nn.Sequential(
            nn.LayerNorm(single_input),
            nn.Linear(single_input, self.dim_token),
        )

        energy_table = build_pair_bin_energy_table(
            distance_bin_start,
            distance_bin_end,
            self.distance_bin_count,
            allow_missing=bool(allow_missing_physics),
        )
        self.register_buffer("pair_bin_energies", energy_table, persistent=False)
        energy_dimension = N_ENERGY_CHANNELS * self.distance_bin_count
        self.physics_projection = nn.Linear(
            energy_dimension,
            self.dim_pair,
            bias=False,
        )

        self.pair_left = nn.Linear(self.dim_token, self.dim_pair, bias=False)
        self.pair_product = nn.Linear(self.dim_token, self.dim_pair, bias=False)
        self.position_projection = nn.Sequential(
            nn.Linear(4, self.dim_pair * 2),
            SwiGLU(),
            nn.Linear(self.dim_pair, self.dim_pair),
        )
        self.inter_chain_embedding = nn.Parameter(torch.zeros(self.dim_pair))
        self.pair_transition = Transition(self.dim_pair)

    def forward(
        self,
        plm_emb: torch.Tensor,
        residue_type: torch.Tensor,
        chain_index: torch.Tensor,
        mask: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        residue_type = residue_type.long().clamp(0, NUM_RESIDUE_TYPES - 1)
        sequential_index = within_chain_index(chain_index.long(), mask)
        position = sinusoidal_position_embedding(
            sequential_index,
            self.index_emb_dim,
        ).to(dtype=plm_emb.dtype)
        plm_features = self.plm_projection(plm_emb)
        feature_dtype = plm_features.dtype
        features = torch.cat(
            (
                plm_features,
                self.residue_embedding(residue_type).to(dtype=feature_dtype),
                position.to(dtype=feature_dtype),
            ),
            dim=-1,
        )
        return self.single_projection(features) * mask[..., None]

    def cross_pair(
        self,
        chain_1: torch.Tensor,
        chain_2: torch.Tensor,
        type_1: torch.Tensor,
        type_2: torch.Tensor,
        mask_1: torch.Tensor,
        mask_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct an order-equivariant rectangular inter-chain map."""
        mask_1 = mask_1.to(dtype=torch.bool)
        mask_2 = mask_2.to(dtype=torch.bool)
        pair_mask = mask_1[:, :, None] & mask_2[:, None, :]

        left = self.pair_left(chain_1)
        right = self.pair_left(chain_2)
        left_product = self.pair_product(chain_1)
        right_product = self.pair_product(chain_2)
        pair = (
            left[:, :, None, :]
            + right[:, None, :, :]
            + left_product[:, :, None, :] * right_product[:, None, :, :]
        )

        symmetric_energy = 0.5 * (
            self.pair_bin_energies
            + self.pair_bin_energies.transpose(0, 1)
        )
        flat_energy = symmetric_energy.flatten(start_dim=-2)
        projected_energy = self.physics_projection(
            flat_energy.to(
                device=pair.device,
                dtype=self.physics_projection.weight.dtype,
            )
        )
        projected_energy = projected_energy.to(dtype=pair.dtype)
        type_1 = type_1.long().clamp(0, NUM_RESIDUE_TYPES - 1)
        type_2 = type_2.long().clamp(0, NUM_RESIDUE_TYPES - 1)
        pair = pair + projected_energy[type_1[:, :, None], type_2[:, None, :]]

        position_1 = _normalized_position(mask_1)
        position_2 = _normalized_position(mask_2)
        p1 = position_1[:, :, None].expand(-1, -1, position_2.shape[1])
        p2 = position_2[:, None, :].expand(-1, position_1.shape[1], -1)
        position_features = torch.stack(
            (
                p1 + p2,
                (p1 - p2).abs(),
                (1.0 - p1) + (1.0 - p2),
                ((1.0 - p1) - (1.0 - p2)).abs(),
            ),
            dim=-1,
        ).to(dtype=pair.dtype)
        pair = pair + self.position_projection(position_features)
        pair = pair + self.inter_chain_embedding.to(dtype=pair.dtype)
        # Transition returns an update; preserve the physics/position inputs.
        pair = pair + self.pair_transition(pair, pair_mask)
        return pair * pair_mask[..., None], pair_mask
