import os
import pickle
from typing import Sequence

import numpy as np

_PARAM_FILES = ("sigma", "epsilon", "mu", "nu", "charge")


def _parameter_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_mpipi_arrays(aa_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Load Mpipi pair parameters in ``aa_names`` order.

    ``sigma``, ``epsilon``, ``mu``, and ``nu`` are ``[n, n]``. ``charge`` is ``[n]``.
    Uses the residue pickles in this directory, not the ``*_ggv1`` copies.
    """
    data_dir = _parameter_dir()
    loaded = {}
    for name in _PARAM_FILES:
        path = os.path.join(data_dir, f"{name}.pickle")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Mpipi parameter file not found: {path}")
        with open(path, "rb") as handle:
            loaded[name] = pickle.load(handle)

    names = [str(name) for name in aa_names]
    missing = [name for name in names if name not in loaded["charge"]]
    if missing:
        raise KeyError(f"Mpipi charge table is missing residues: {missing}")

    count = len(names)
    sigma = np.zeros((count, count), dtype=np.float64)
    epsilon = np.zeros((count, count), dtype=np.float64)
    mu = np.zeros((count, count), dtype=np.float64)
    nu = np.zeros((count, count), dtype=np.float64)
    charge = np.zeros((count,), dtype=np.float64)
    for i, aa_i in enumerate(names):
        charge[i] = float(loaded["charge"][aa_i])
        row_sigma = loaded["sigma"][aa_i]
        row_epsilon = loaded["epsilon"][aa_i]
        row_mu = loaded["mu"][aa_i]
        row_nu = loaded["nu"][aa_i]
        for j, aa_j in enumerate(names):
            sigma[i, j] = float(row_sigma[aa_j])
            epsilon[i, j] = float(row_epsilon[aa_j])
            mu[i, j] = float(row_mu[aa_j])
            nu[i, j] = float(row_nu[aa_j])
    return {
        "sigma": sigma,
        "epsilon": epsilon,
        "mu": mu,
        "nu": nu,
        "charge": charge,
    }


def coulomb(qi, qj, rij, dielectric=80.0, salt=0.150):
    """Debye-Huckel Coulomb energy in kJ/mol. Distances are Angstroms."""
    rij = np.asarray(rij, dtype=np.float64)
    qi = np.asarray(qi, dtype=np.float64)
    qj = np.asarray(qj, dtype=np.float64)
    kappa = np.sqrt(float(salt)) / 3.06
    elementary = 1.602176634e-19
    avogadro = 6.023e23
    conversion = (elementary * elementary * avogadro) / (np.pi * 4 * 8.854187812799999e-12)
    rij_eval = np.maximum(rij, 1e-6)
    screened = np.exp(-kappa * rij_eval)
    energy_kj = (
        (conversion * qi * qj) / (float(dielectric) * rij_eval * 1e-10) * screened / 1000.0
    )
    return np.where(rij > 0.0, energy_kj, 0.0)


def wang_frenkel(r, sigma_ij, epsilon_ij, mu_ij=2, nu_ij=1):
    """Wang-Frenkel energy in kJ/mol. Zero at and beyond ``3 * sigma``."""
    r = np.asarray(r, dtype=np.float64)
    sigma_ij = np.asarray(sigma_ij, dtype=np.float64)
    epsilon_ij = np.asarray(epsilon_ij, dtype=np.float64)
    mu_ij = np.asarray(mu_ij, dtype=np.float64)
    nu_ij = np.asarray(nu_ij, dtype=np.float64)

    cutoff = 3.0 * sigma_ij
    ratio = cutoff / sigma_ij
    two_mu = 2.0 * mu_ij
    two_nu = 2.0 * nu_ij
    alpha_term1 = two_nu * np.power(ratio, two_mu)
    alpha_term2 = (two_nu + 1.0) / (two_nu * (np.power(ratio, two_mu) - 1.0))
    alpha = alpha_term1 * np.power(alpha_term2, two_nu + 1.0)

    r_eval = np.maximum(r, 1e-6)
    repulsive = np.power(sigma_ij / r_eval, two_mu) - 1.0
    attractive_base = np.maximum(np.power(cutoff / r_eval, two_mu) - 1.0, 0.0)
    energy = epsilon_ij * alpha * repulsive * np.power(attractive_base, two_nu)
    valid = (r > 0.0) & (r < cutoff)
    return np.where(valid, energy, 0.0)
