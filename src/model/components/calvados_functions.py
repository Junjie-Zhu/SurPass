import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CalvadosModel:
    sigmamap: pd.DataFrame
    lambdamap: pd.DataFrame
    yukawa_eps: pd.DataFrame
    yukawa_kappa: float
    cutoff: float = 2.0
    lj_eps: float = 4.184 * 0.2
    yukawa_r_cut: float = 4.0


def _default_residue_pickle_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    candidates = [
        os.path.join(here, "Calvados_data", "calvados_residues.pickle"),
        os.path.join(root, "Calvados_data", "calvados_residues.pickle"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    # Keep a deterministic fallback even when file is absent.
    return candidates[0]


def load_calvados_model(
    *,
    version: str = "CALVADOS2",
    salt: float = 0.150,
    pH: float = 7.4,
    temp: float = 298.15,
    residue_pickle_path: Optional[str] = None,
) -> CalvadosModel:
    data_path = residue_pickle_path or _default_residue_pickle_path()
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"CALVADOS residue file not found: {data_path}. "
            "Expected a calvados_residues.pickle file."
        )

    residues = pd.read_pickle(data_path).copy()
    if "one" not in residues.columns:
        raise ValueError("CALVADOS residue table must contain column 'one'.")
    if "sigmas" not in residues.columns:
        raise ValueError("CALVADOS residue table must contain column 'sigmas'.")
    if "q" not in residues.columns:
        raise ValueError("CALVADOS residue table must contain column 'q'.")
    if version not in residues.columns:
        raise ValueError(f"CALVADOS residue table missing version column: {version}")

    residues["one"] = residues["one"].astype(str).str.upper()
    residues = residues.drop_duplicates(subset=["one"], keep="first").set_index("one")
    residues["lambdas"] = residues[version].astype(float)

    yukawa_kappa, yukawa_eps, residues = _gen_params(residues, salt=salt, pH=pH, temp=temp)
    sigmamap = pd.DataFrame(
        (residues.sigmas.values + residues.sigmas.values.reshape(-1, 1)) / 2.0,
        index=residues.index,
        columns=residues.index,
    )
    lambdamap = pd.DataFrame(
        (residues.lambdas.values + residues.lambdas.values.reshape(-1, 1)) / 2.0,
        index=residues.index,
        columns=residues.index,
    )

    return CalvadosModel(
        sigmamap=sigmamap,
        lambdamap=lambdamap,
        yukawa_eps=yukawa_eps,
        yukawa_kappa=float(yukawa_kappa),
    )


def _gen_params(residues: pd.DataFrame, *, salt: float, pH: float, temp: float):
    r = residues.copy()
    r["q"] = r["q"].astype(float)
    if "H" in r.index:
        r.loc["H", "q"] = 1.0 / (1.0 + 10 ** (pH - 6.0))

    RT = 8.3145 * temp * 1e-3
    fepsw = lambda T: 5321.0 / T + 233.76 - 0.9297 * T + 0.1417e-2 * T * T - 0.8292e-6 * T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2 / (4 * np.pi * 8.854188 * epsw) * 6.022 * 1000 / RT
    yukawa_kappa = np.sqrt(8 * np.pi * lB * salt * 6.022 / 10)

    qq = pd.DataFrame(r.q.values * r.q.values.reshape(-1, 1), index=r.index, columns=r.index)
    yukawa_eps = qq * lB * RT
    return yukawa_kappa, yukawa_eps, r


def compute_ashbaugh_hatch(r_nm, sigma, lam, *, cutoff=2.0, lj_eps=4.184 * 0.2):
    r_nm = np.asarray(r_nm, dtype=float)
    rc = float(cutoff)
    eps = float(lj_eps)
    shift = (sigma / rc) ** 12 - (sigma / rc) ** 6
    switch = np.power(2.0, 1.0 / 6.0) * sigma

    y = 4.0 * eps * lam * ((sigma / r_nm) ** 12 - (sigma / r_nm) ** 6 - shift)
    z = 4.0 * eps * ((sigma / r_nm) ** 12 - (sigma / r_nm) ** 6 - lam * shift) + eps * (1.0 - lam)
    return np.where(r_nm < switch, z, y)


def compute_yukawa(r_nm, q, yukawa_kappa, *, yukawa_r_cut=4.0):
    r_nm = np.asarray(r_nm, dtype=float)
    shift = np.exp(-yukawa_kappa * yukawa_r_cut) / yukawa_r_cut
    return q * (np.exp(-yukawa_kappa * r_nm) / r_nm - shift)


def compute_calvados_energy(r_nm, sigma, lam, q, yukawa_kappa, *, cutoff=2.0, lj_eps=4.184 * 0.2, yukawa_r_cut=4.0):
    ah = compute_ashbaugh_hatch(r_nm, sigma, lam, cutoff=cutoff, lj_eps=lj_eps)
    yu = compute_yukawa(r_nm, q, yukawa_kappa, yukawa_r_cut=yukawa_r_cut)
    return ah + yu
