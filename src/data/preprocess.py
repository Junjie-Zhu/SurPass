### calculate surface, atom and residue features for protein/peptide
### the desired pipeline is: pdb --> xyzr --> msms 
###                          --> verts, faces --> subsample verts to 1A resolution 
###                          --> 

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import Optional
import time

from pdbfixer import PDBFixer
from openmm.app import PDBFile
import Bio.PDB as bio
import numpy as np
import torch
try:
    from pykeops.torch.cluster import grid_cluster as _grid_cluster_keops
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _grid_cluster_keops = None

from src.data.compute_charges import computeCharges
from src.data.compute_hydrophibicity import computeHydrophobicity
from src.data.compute_normals import compute_normals
from src.data.compute_apbs import computeAPBS
from src.data.compute_curvatures import curvatures as compute_curvatures
from src.common.global_vars import MSMS_BIN
from src.common.residue_constants import (
    restype_name_to_atom14_names,
    restype_3to1,
    restype_order,
    unk_restype_index,
)
from src.common.atom_constants import polarHydrogens

VDW_RADII_MSMS = {
    "N": 1.55,
    "O": 1.52,
    "C": 1.70,
    "H": 1.10,
    "S": 1.80,
    "P": 1.80,
    "SE": 1.90,
    "K": 2.75,
    "NA": 2.27,
    "MG": 1.73,
    "ZN": 1.39,
}
ATOMTYPES_DMASIF = ["C", "H", "O", "N", "S", "SE"]
ATOMTYPES_DMASIF_INDEX = {name: idx for idx, name in enumerate(ATOMTYPES_DMASIF)}
CONTACT_DISTANCE_THRESHOLD_A = 4.5
CONTACT_CHUNK_SIZE = 128
CPU_GPU_SCHEMA_VERSION = "v1"


def _grid_cluster(x: torch.Tensor, scale: float, use_keops: bool = True) -> torch.Tensor:
    """Grid clustering with optional KeOps acceleration."""
    if _grid_cluster_keops is not None and use_keops:
        return _grid_cluster_keops(x, scale).long()

    coords = torch.floor(x / scale).to(dtype=torch.long)
    _, labels = torch.unique(coords, dim=0, return_inverse=True)
    return labels.long()


def fix_pdb(pdb_path: Path, processed_pdb_path: Path) -> None:
    if processed_pdb_path.exists():
        return
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens()
    PDBFile.writeFile(fixer.topology, fixer.positions, str(processed_pdb_path))


def remove_hetatm_records(
    pdb_path: Path,
    cleaned_pdb_path: Path,
    overwrite: bool = False,
) -> Path:
    """Write a copy of pdb_path without HETATM records."""
    if cleaned_pdb_path.exists() and not overwrite:
        return cleaned_pdb_path

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as src, cleaned_pdb_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if line.startswith("HETATM"):
                continue
            dst.write(line)
    return cleaned_pdb_path


def prepare_fixed_pdb(
    pdb_path: Path,
    remove_hetatm: bool = False,
    overwrite_cleaned: bool = False,
) -> tuple[Path, Optional[Path]]:
    """Return fixed PDB path and optional cleaned (no-HETATM) path."""
    source_path = pdb_path
    cleaned_path: Optional[Path] = None
    if remove_hetatm:
        cleaned_path = pdb_path.with_suffix(".nohet.pdb")
        source_path = remove_hetatm_records(
            pdb_path,
            cleaned_path,
            overwrite=overwrite_cleaned,
        )

    fixed_pdb_path = source_path.with_suffix(".fixed.pdb")
    fix_pdb(source_path, fixed_pdb_path)
    return fixed_pdb_path, cleaned_path


def _load_model(pdb_path: Path):
    parser = bio.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(pdb_path))
    return next(structure.get_models())


def pdb_to_xyzr(pdb_path: Path, xyzr_path: Path, model=None) -> None:
    model = _load_model(pdb_path) if model is None else model
    with xyzr_path.open("w", encoding="utf-8") as handle:
        for atom in model.get_atoms():
            elem = atom.element.strip().upper()
            if elem not in VDW_RADII_MSMS:
                continue
            x, y, z = atom.get_coord()
            handle.write(f"{x:.04f} {y:.04f} {z:.04f} {VDW_RADII_MSMS[elem]:.02f}\n")


def pdb_to_xyzrn(pdb_path: Path, xyzrn_path: Path, model=None) -> None:
    """
    Convert a PDB structure to xyzrn format used by MSMS-style tools.

    Output line format:
        x y z radius 1 chain_resid_insertion_resname_atom_color
    """
    model = _load_model(pdb_path) if model is None else model
    with xyzrn_path.open("w", encoding="utf-8") as handle:
        for atom in model.get_atoms():
            elem = atom.element.strip().upper()
            if elem not in VDW_RADII_MSMS:
                continue

            residue = atom.get_parent()
            chain = residue.get_parent()
            chain_id = chain.get_id() if chain.get_id() != "" else " "
            resname = residue.get_resname().upper()
            atom_name = atom.get_name().strip()

            # Color tag follows the MaSIF convention used by downstream scripts.
            color = "Green"
            if elem == "O":
                color = "Red"
            elif elem == "N":
                color = "Blue"
            elif elem == "H" and atom_name in polarHydrogens.get(resname, []):
                color = "Blue"

            x, y, z = atom.get_coord()
            insertion = residue.get_id()[2] if residue.get_id()[2] != " " else "x"
            full_id = f"{chain_id}_{residue.get_id()[1]}_{insertion}_{resname}_{atom_name}_{color}"
            radius = VDW_RADII_MSMS[elem]
            handle.write(f"{x:.06f} {y:.06f} {z:.06f} {radius:.02f} 1 {full_id}\n")
    

def get_filtered_atom_metadata(model):
    atom_metadata = []
    for atom in model.get_atoms():
        elem = atom.element.strip().upper()
        if elem in VDW_RADII_MSMS:
            residue = atom.get_parent()
            chain = residue.get_parent()
            chain_id = chain.get_id() if chain.get_id() != "" else " "
            hetero_flag, resseq, icode = residue.get_id()
            if icode == "":
                icode = " "
            atom_metadata.append(
                {
                    "element": elem,
                    "vertex_name": f"{chain_id}_{resseq}_{icode}_{residue.get_resname()}_{atom.get_name()}_X",
                }
            )
    return atom_metadata


def parse_verts(vert_file):
    """
    Generate the vertices and faces (and optionally the normals) from .vert and .face files generated by MSMS
    :param vert_file:
    :param face_file:
    :param keep_normals:
    :return:
    """
    with open(vert_file, 'r', errors='ignore') as f:
        # Parse the file and ensure it looks sound
        lines = f.readlines()
        n_vert = int(lines[2].split()[0])
        no_header = lines[3:]
        assert len(no_header) == n_vert

        # Parse the info to retrieve vertices and mapped atom / residue ids.
        tokens = [line.split() for line in no_header]
        verts = np.array([vals[:3] for vals in tokens], dtype=np.float32)
        verts_atom_id = np.array([int(vals[7]) for vals in tokens], dtype=np.int64)
        verts_res_id = np.array([int(vals[9].split("_")[1]) for vals in tokens], dtype=np.int64)  # full_id assigned by pdb_to_xyzrn

    return verts, verts_atom_id, verts_res_id


def run_msms(
    xyzr_path: Path,
    out_prefix: Path,
    density: float = 3.0,
    timeout_s: int = 180,
    force_rerun: bool = False,
):
    msms_bin = os.environ.get("VISPEPO_MSMS_BIN", MSMS_BIN)
    command = [msms_bin, "-density", str(density), "-hdensity", "3.0", "-probe", "1.5", "-all_components", 
               "-if", str(xyzr_path), "-of", str(out_prefix), "-af", str(out_prefix)]
    vert_path = out_prefix.with_suffix(".msms.vert")
    face_path = out_prefix.with_suffix(".msms.face")

    if force_rerun:
        for p in (vert_path, face_path):
            if p.exists():
                p.unlink()

    if not vert_path.exists():
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"MSMS timed out after {timeout_s}s for {xyzr_path}.\n"
                f"Command: {' '.join(command)}\n"
                "Try increasing timeout_s or rerun with force_rerun=True."
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"MSMS failed ({result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        
    # parse the output
    verts, verts_atom_id, verts_res_id = parse_verts(vert_path)
    return verts, verts_atom_id, verts_res_id


def extract_surface_primitives_cpu(
    pdb_path: Path,
    model=None,
    msms_timeout_s: int = 180,
    force_rerun_msms: bool = False,
) -> dict:
    """
    CPU-only surface preparation:
    pdb -> xyzrn -> msms -> raw vertex metadata.
    """
    model = _load_model(pdb_path) if model is None else model
    xyzrn_path = pdb_path.with_suffix(".xyzrn")
    pdb_to_xyzrn(pdb_path, xyzrn_path, model=model)

    atom_metadata = get_filtered_atom_metadata(model)
    verts, verts_atom_id, verts_res_id = run_msms(
        xyzrn_path,
        pdb_path.with_suffix(".msms"),
        timeout_s=msms_timeout_s,
        force_rerun=force_rerun_msms,
    )

    n_atomtypes = len(ATOMTYPES_DMASIF)
    verts_atomtype = np.zeros((len(verts_atom_id), n_atomtypes), dtype=np.float32)
    verts_names = []
    for i, atom_id in enumerate(verts_atom_id):
        atom_idx = int(atom_id) - 1  # MSMS ids are 1-based
        if atom_idx < 0 or atom_idx >= len(atom_metadata):
            verts_names.append("A_0_ _GLY_C_X")
            continue
        meta = atom_metadata[atom_idx]
        elem = meta["element"]
        verts_names.append(meta["vertex_name"])
        if elem in ATOMTYPES_DMASIF_INDEX:
            verts_atomtype[i, ATOMTYPES_DMASIF_INDEX[elem]] = 1.0

    return {
        "verts": verts.astype(np.float32),
        "verts_res_id": verts_res_id.astype(np.int64),
        "verts_atomtype": verts_atomtype.astype(np.float32),
        "verts_names": verts_names,
    }


def subsample(x, x_res_id, x_atomtype, scale=1.0, use_keops=True):
    """Subsamples the point cloud using a grid (cubic) clustering scheme.

    The function returns one average sample per cell, as described in Fig. 3.e)
    of the paper.

    Args:
        x (Tensor): (N,3) point cloud.
        batch (integer Tensor, optional): (N,) batch vector, as in PyTorch_geometric.
            Defaults to None.
        scale (float, optional): side length of the cubic grid cells. Defaults to 1 (Angstrom).

    Returns:
        points (Tensor): (M,3) sub-sampled point cloud, with M <= N.
        atomtypes (Tensor): (M,6) representative one-hot atom types.
        res_ids (Tensor): (M,) representative residue id per sub-sampled point.
        rep_idx (Tensor): (M,) representative indices in the original vertices.
    """
    labels = _grid_cluster(x, scale, use_keops=use_keops)
    _, labels = torch.unique(labels, sorted=True, return_inverse=True)
    C = int(labels.max().item()) + 1

    x_1 = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
    D = x_1.shape[1]
    points = torch.zeros_like(x_1[:C])

    points.scatter_add_(0, labels[:, None].repeat(1, D), x_1)
    points = (points[:, :-1] / points[:, -1:]).contiguous()

    # Keep IDs aligned with subsampled points by selecting, in each voxel,
    # the original vertex nearest to that voxel centroid.
    atomtype_src = torch.as_tensor(x_atomtype, device=x.device, dtype=torch.float32)
    res_src = torch.as_tensor(x_res_id, device=x.device, dtype=torch.long)

    # Per-vertex distance to its own voxel centroid.
    dist2 = ((x - points[labels]) ** 2).sum(dim=-1)

    n = labels.shape[0]
    idx = torch.arange(n, device=x.device, dtype=torch.long)

    # IMPORTANT: with include_self=True, scatter_reduce uses current values too.
    # Initialize with sentinel "n" to avoid undefined indices from uninitialized memory.
    rep_idx = torch.full((C,), n, device=x.device, dtype=torch.long)
    min_dist2 = torch.full((C,), torch.finfo(dist2.dtype).max, device=x.device, dtype=dist2.dtype)
    min_dist2.scatter_reduce_(0, labels, dist2, reduce="amin", include_self=True)
    
    is_min = (dist2 == min_dist2[labels])
    candidate_idx = torch.where(is_min, idx, torch.full_like(idx, n))
    rep_idx.scatter_reduce_(0, labels, candidate_idx, reduce="amin", include_self=True)

    atomtypes = atomtype_src[rep_idx]
    res_ids = res_src[rep_idx]

    return points, atomtypes, res_ids, rep_idx


def build_surface_features_from_cpu_primitives(
    pdb_path: Path,
    surface_cpu: dict,
    normal_smoothness: float = 0.01,
    use_keops: bool = True,
    model=None,
) -> dict:
    """Build full surface features from CPU primitives (GPU/KeOps stage)."""
    model = _load_model(pdb_path) if model is None else model
    verts = torch.from_numpy(np.asarray(surface_cpu["verts"], dtype=np.float32)).float()
    verts_res_id = torch.from_numpy(np.asarray(surface_cpu["verts_res_id"], dtype=np.int64)).long()
    verts_atomtype = torch.from_numpy(np.asarray(surface_cpu["verts_atomtype"], dtype=np.float32)).float()
    verts_names = list(surface_cpu["verts_names"])

    points, atomtypes, res_ids, rep_idx = subsample(
        verts, verts_res_id, verts_atomtype, use_keops=use_keops
    )
    normals = compute_normals(
        points,
        smoothness=normal_smoothness,
        atomtypes=atomtypes,
        use_keops=use_keops,
    )

    subsampled_names = [verts_names[i] for i in rep_idx.tolist()]
    points_np = points.detach().cpu().numpy()
    hbond = torch.as_tensor(
        computeCharges(str(pdb_path), points_np, subsampled_names, structure=model)
    ).float()
    hydrophobicity = torch.as_tensor(computeHydrophobicity(subsampled_names)).float()
    curvatures = compute_curvatures(
        points,
        triangles=None,
        normals=normals,
        scales=[1.0, 2.0, 3.0, 5.0, 10.0],
        batch=None,
        use_keops=use_keops,
    )

    return {
        "xyzs": points,
        "normals": normals,
        "res_ids": res_ids,
        "hbond": hbond,
        "hydrophobicity": hydrophobicity,
        "curvatures": curvatures,
    }


def get_surface_features(
    pdb_path: Path,
    normal_smoothness: float = 0.01,
    use_keops: bool = True,
    model=None,
    enable_timing: bool = False,
    msms_timeout_s: int = 180,
    force_rerun_msms: bool = False,
):
    t0 = time.perf_counter()
    model = _load_model(pdb_path) if model is None else model

    surface_cpu = extract_surface_primitives_cpu(
        pdb_path,
        model=model,
        msms_timeout_s=msms_timeout_s,
        force_rerun_msms=force_rerun_msms,
    )
    t_surface_cpu = time.perf_counter()
    surface_features = build_surface_features_from_cpu_primitives(
        pdb_path=pdb_path,
        surface_cpu=surface_cpu,
        normal_smoothness=normal_smoothness,
        use_keops=use_keops,
        model=model,
    )
    t_feat = time.perf_counter()
    if enable_timing:
        print(
            f"[timer][surface] {pdb_path.name} "
            f"cpu_surface={t_surface_cpu - t0:.2f}s "
            f"gpu_surface={t_feat - t_surface_cpu:.2f}s "
            f"total={t_feat - t0:.2f}s"
        )
    return surface_features


def _encode_ascii_padded_4(name: str) -> np.ndarray:
    """Encode a token as ord(c)-32, padded/truncated to length 4."""
    encoded = np.zeros(4, dtype=np.int64)
    padded = str(name)[:4].ljust(4)
    for i, c in enumerate(padded):
        encoded[i] = ord(c) - 32
    return encoded


def get_residue_and_atom_features(
    pdb_path: Path,
    model=None,
    enable_timing: bool = False,
):
    t0 = time.perf_counter()
    model = _load_model(pdb_path) if model is None else model

    residue_type = []
    residue_index = []
    chain_index = []
    chain_break = []

    atom14_positions = []
    atom14_mask = []
    atom14_name = []

    chain_to_idx = {}
    prev_chain_id = None

    for chain in model:
        chain_id = chain.get_id() if chain.get_id() != "" else " "
        if chain_id not in chain_to_idx:
            chain_to_idx[chain_id] = len(chain_to_idx)
        chain_idx = chain_to_idx[chain_id]

        for res in chain:
            hetero_flag, resseq, icode = res.get_id()
            if hetero_flag.strip() != "":
                continue

            resname = res.get_resname().upper()
            atom14_names = restype_name_to_atom14_names.get(
                resname, restype_name_to_atom14_names["UNK"]
            )

            atom_name_to_atom = {atom.get_name().strip(): atom for atom in res.get_atoms()}

            atom_pos = np.zeros((14, 3), dtype=np.float32)
            atom_mask = np.zeros((14,), dtype=np.float32)
            atom_name_enc = np.zeros((14, 4), dtype=np.int64)

            for i, atom_name in enumerate(atom14_names):
                if atom_name == "":
                    continue
                atom_name_enc[i] = _encode_ascii_padded_4(atom_name)
                atom_obj = atom_name_to_atom.get(atom_name)
                if atom_obj is None:
                    continue
                atom_mask[i] = 1.0
                atom_pos[i] = atom_obj.get_coord().astype(np.float32)

            one_letter = restype_3to1.get(resname, "X")
            residue_type.append(restype_order.get(one_letter, unk_restype_index))
            residue_index.append(int(resseq))
            chain_index.append(chain_idx)
            chain_break.append(0 if prev_chain_id is None or prev_chain_id == chain_id else 1)
            prev_chain_id = chain_id

            atom14_positions.append(atom_pos)
            atom14_mask.append(atom_mask)
            atom14_name.append(atom_name_enc)

    residue_features = {
        "residue_position": torch.as_tensor(np.array(atom14_positions)[:, 1, :], dtype=torch.float32),  # CA atom position
        "residue_type": torch.as_tensor(np.array(residue_type), dtype=torch.long),
        "residue_index": torch.as_tensor(np.array(residue_index), dtype=torch.long),
        "chain_index": torch.as_tensor(np.array(chain_index), dtype=torch.long),
        "chain_break": torch.as_tensor(np.array(chain_break), dtype=torch.float32),
        "mask": torch.ones(len(residue_type), dtype=torch.float32),
    }

    atom_features = {
        "atom14_position": torch.as_tensor(np.array(atom14_positions), dtype=torch.float32),
        "atom14_mask": torch.as_tensor(np.array(atom14_mask), dtype=torch.float32),
        "atom14_name": torch.as_tensor(np.array(atom14_name), dtype=torch.long),
    }
    if enable_timing:
        t1 = time.perf_counter()
        print(f"[timer][residue_atom] {pdb_path.name} total={t1 - t0:.2f}s")

    return {"residue_features": residue_features, "atom_features": atom_features}


def calculate_atom14_contact_labels(
    protein_atom14_position: torch.Tensor,
    protein_atom14_mask: torch.Tensor,
    peptide_atom14_position: torch.Tensor,
    peptide_atom14_mask: torch.Tensor,
    threshold: float = CONTACT_DISTANCE_THRESHOLD_A,
    chunk_size: int = CONTACT_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute residue-level and pairwise contacts from atom14 geometry."""
    protein_atom14_position = protein_atom14_position.to(dtype=torch.float32)
    peptide_atom14_position = peptide_atom14_position.to(dtype=torch.float32)
    protein_atom14_mask = protein_atom14_mask.to(dtype=torch.bool)
    peptide_atom14_mask = peptide_atom14_mask.to(dtype=torch.bool)

    n_protein = protein_atom14_position.shape[0]
    n_peptide = peptide_atom14_position.shape[0]
    pair_contact = torch.zeros((n_protein, n_peptide), dtype=torch.bool)
    if n_protein == 0 or n_peptide == 0:
        return pair_contact.any(dim=1), pair_contact.any(dim=0), pair_contact

    threshold_sq = float(threshold) ** 2
    peptide_mask = peptide_atom14_mask[None, :, None, :]

    for start in range(0, n_protein, max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), n_protein)
        protein_chunk_pos = protein_atom14_position[start:end]
        protein_chunk_mask = protein_atom14_mask[start:end, None, :, None]
        valid_mask = protein_chunk_mask & peptide_mask
        if not valid_mask.any():
            continue

        diff = protein_chunk_pos[:, None, :, None, :] - peptide_atom14_position[None, :, None, :, :]
        dist_sq = (diff * diff).sum(dim=-1)
        pair_contact[start:end] = ((dist_sq <= threshold_sq) & valid_mask).any(dim=(-1, -2))

    protein_site_contact = pair_contact.any(dim=1)
    peptide_site_contact = pair_contact.any(dim=0)
    return protein_site_contact, peptide_site_contact, pair_contact


def build_contact_label(
    protein_features: dict,
    peptide_features: dict,
    threshold: float = CONTACT_DISTANCE_THRESHOLD_A,
) -> dict:
    protein_atom14 = protein_features["atom_features"]["atom14_position"]
    protein_mask = protein_features["atom_features"]["atom14_mask"]
    peptide_atom14 = peptide_features["atom_features"]["atom14_position"]
    peptide_mask = peptide_features["atom_features"]["atom14_mask"]

    protein_site_contact, peptide_site_contact, pair_contact = calculate_atom14_contact_labels(
        protein_atom14_position=protein_atom14,
        protein_atom14_mask=protein_mask,
        peptide_atom14_position=peptide_atom14,
        peptide_atom14_mask=peptide_mask,
        threshold=threshold,
    )
    return {
        "protein_contact_1d": protein_site_contact.to(dtype=torch.float32),
        "peptide_contact_1d": peptide_site_contact.to(dtype=torch.float32),
        "pair_contact_2d": pair_contact.to(dtype=torch.float32),
        "contact_1d": torch.cat([protein_site_contact, peptide_site_contact], dim=0).to(dtype=torch.float32),
        "distance_threshold": torch.tensor(float(threshold), dtype=torch.float32),
    }


def _count_polymer_residues(model) -> int:
    """Count non-hetero residues in a Bio.PDB model."""
    count = 0
    for chain in model:
        for res in chain:
            hetero_flag, _, _ = res.get_id()
            if hetero_flag.strip() == "":
                count += 1
    return count


def validate_cpu_intermediate_payload(payload: dict, context: str = "cpu_intermediate") -> None:
    required_top = {"meta", "surface_cpu", "residue_features", "atom_features"}
    missing_top = required_top - set(payload.keys())
    if missing_top:
        raise ValueError(f"{context}: missing top-level keys: {sorted(missing_top)}")

    meta = payload["meta"]
    for key in ("schema_version", "fixed_pdb_path"):
        if key not in meta:
            raise ValueError(f"{context}: missing meta key '{key}'")
    if meta["schema_version"] != CPU_GPU_SCHEMA_VERSION:
        raise ValueError(
            f"{context}: incompatible schema version '{meta['schema_version']}', "
            f"expected '{CPU_GPU_SCHEMA_VERSION}'"
        )

    surface_cpu = payload["surface_cpu"]
    for key in ("verts", "verts_res_id", "verts_atomtype", "verts_names"):
        if key not in surface_cpu:
            raise ValueError(f"{context}: missing surface_cpu key '{key}'")


def build_cpu_intermediate_features(
    pdb_path: Path,
    normal_smoothness: float = 0.01,
    remove_hetatm: bool = True,
    enable_timing: bool = False,
    msms_timeout_s: int = 180,
    force_rerun_msms: bool = False,
    max_residues: int = 1024,
) -> dict:
    """
    CPU stage entry:
      remove HETATM -> fix -> xyzrn/msms/cpu primitives + residue/atom features.
    """
    del normal_smoothness  # reserved for compatibility with monolithic API knobs
    t0 = time.perf_counter()
    fixed_pdb_path, cleaned_pdb_path = prepare_fixed_pdb(
        pdb_path, remove_hetatm=remove_hetatm
    )
    t_fix = time.perf_counter()
    model = _load_model(fixed_pdb_path)
    residue_count = _count_polymer_residues(model)
    if residue_count > max_residues:
        raise ValueError(
            f"skip_large_protein: residue_count={residue_count} > max_residues={max_residues}"
        )

    surface_cpu = extract_surface_primitives_cpu(
        fixed_pdb_path,
        model=model,
        msms_timeout_s=msms_timeout_s,
        force_rerun_msms=force_rerun_msms,
    )
    residue_atom_features = get_residue_and_atom_features(
        fixed_pdb_path, model=model, enable_timing=enable_timing
    )
    t_end = time.perf_counter()
    if enable_timing:
        print(
            f"[timer][cpu_stage] {pdb_path.name} "
            f"prepare={t_fix - t0:.2f}s "
            f"features={t_end - t_fix:.2f}s "
            f"total={t_end - t0:.2f}s"
        )

    payload = {
        "meta": {
            "schema_version": CPU_GPU_SCHEMA_VERSION,
            "source_pdb_path": str(pdb_path),
            "cleaned_pdb_path": str(cleaned_pdb_path) if cleaned_pdb_path else "",
            "fixed_pdb_path": str(fixed_pdb_path),
            "remove_hetatm": bool(remove_hetatm),
        },
        "surface_cpu": surface_cpu,
        **residue_atom_features,
    }
    validate_cpu_intermediate_payload(payload, context=f"cpu_intermediate:{pdb_path.name}")
    return payload


def build_surface_features_from_cpu_payload(
    cpu_payload: dict,
    normal_smoothness: float = 0.01,
    use_keops: bool = True,
) -> dict:
    """GPU stage: complete surface features from saved CPU intermediate payload."""
    validate_cpu_intermediate_payload(cpu_payload)
    fixed_pdb_path = Path(cpu_payload["meta"]["fixed_pdb_path"])
    model = _load_model(fixed_pdb_path)
    return build_surface_features_from_cpu_primitives(
        pdb_path=fixed_pdb_path,
        surface_cpu=cpu_payload["surface_cpu"],
        normal_smoothness=normal_smoothness,
        use_keops=use_keops,
        model=model,
    )


def compose_final_features(
    cpu_payload: dict,
    surface_features: dict,
) -> dict:
    """Compose final feature payload format consumed by training/inference."""
    validate_cpu_intermediate_payload(cpu_payload)
    return {
        "surface_features": surface_features,
        "residue_features": cpu_payload["residue_features"],
        "atom_features": cpu_payload["atom_features"],
    }


def get_all_features(
    pdb_path: Path,
    normal_smoothness: float = 0.01,
    use_keops: bool = True,
    enable_timing: bool = False,
    msms_timeout_s: int = 180,
    force_rerun_msms: bool = False,
    max_residues: int = 1024,
):
    """
    Unified entry point that parses the PDB once and computes both
    surface and residue/atom features.
    """
    # first fix and add hydrogens
    t0 = time.perf_counter()
    fixed_pdb_path = pdb_path.with_suffix(".fixed.pdb")
    fix_pdb(pdb_path, fixed_pdb_path)
    t_fix = time.perf_counter()

    model = _load_model(fixed_pdb_path)
    residue_count = _count_polymer_residues(model)
    if residue_count > max_residues:
        raise ValueError(
            f"skip_large_protein: residue_count={residue_count} > max_residues={max_residues}"
        )
    surface_features = get_surface_features(
        fixed_pdb_path,
        normal_smoothness=normal_smoothness,
        use_keops=use_keops,
        model=model,
        enable_timing=enable_timing,
        msms_timeout_s=msms_timeout_s,
        force_rerun_msms=force_rerun_msms,
    )
    t_surface = time.perf_counter()
    residue_atom_features = get_residue_and_atom_features(
        fixed_pdb_path, model=model, enable_timing=enable_timing
    )
    t_resatom = time.perf_counter()
    if enable_timing:
        print(
            f"[timer][all] {pdb_path.name} "
            f"fix={t_fix - t0:.2f}s "
            f"surface={t_surface - t_fix:.2f}s "
            f"res_atom={t_resatom - t_surface:.2f}s "
            f"total={t_resatom - t0:.2f}s"
        )
    return {
        "surface_features": surface_features,
        **residue_atom_features,
    }


def _process_pair_row(
    row_dict,
    output_dir: Path,
    use_keops: bool,
    torch_threads: int,
    enable_timing: bool = False,
    msms_timeout_s: int = 180,
    force_rerun_msms: bool = False,
    max_residues: int = 1024,
):
    """Worker for one pair row (picklable for multiprocessing)."""
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    pdb_id = str(row_dict["group_id"])
    row_output_dir = output_dir / pdb_id
    row_output_dir.mkdir(parents=True, exist_ok=True)

    protein_path = Path(row_dict["query_pdb"])
    peptide_path = Path(row_dict["positive_pdb"])

    if not protein_path.exists() or not peptide_path.exists():
        return pdb_id, False, "missing_input"

    try:
        protein_features = get_all_features(
            protein_path,
            use_keops=use_keops,
            enable_timing=enable_timing,
            msms_timeout_s=msms_timeout_s,
            force_rerun_msms=force_rerun_msms,
            max_residues=max_residues,
        )
        peptide_features = get_all_features(
            peptide_path,
            use_keops=use_keops,
            enable_timing=enable_timing,
            msms_timeout_s=msms_timeout_s,
            force_rerun_msms=force_rerun_msms,
            max_residues=max_residues,
        )
        label_data = build_contact_label(protein_features, peptide_features)
        torch.save(protein_features, row_output_dir / "receptor.pt")
        torch.save(peptide_features, row_output_dir / "peptide.pt")
        torch.save(label_data, row_output_dir / "label.pt")
        return pdb_id, True, ""
    except Exception as e:
        return pdb_id, False, str(e)


def _process_row_subset(
    rows_subset,
    output_dir: Path,
    use_keops: bool,
    torch_threads: int,
    enable_timing: bool = False,
    msms_timeout_s: int = 180,
    force_rerun_msms: bool = False,
    max_residues: int = 1024,
    gpu_id: int = 0,
):
    """
    Process one subset sequentially inside a single worker process.
    This avoids per-row process spawn overhead and is friendlier to GPU/KeOps flows.
    """
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    if use_keops and torch.cuda.is_available():
        try:
            torch.cuda.set_device(gpu_id)
            if enable_timing:
                print(f"[worker] using cuda:{gpu_id}")
        except Exception as e:
            if enable_timing:
                print(f"[worker] failed to set cuda:{gpu_id}, fallback default device ({e})")

    results = []
    for row_dict in tqdm(rows_subset, desc="Processing rows", leave=False):
        results.append(
            _process_pair_row(
                row_dict=row_dict,
                output_dir=output_dir,
                use_keops=use_keops,
                torch_threads=torch_threads,
                enable_timing=enable_timing,
                msms_timeout_s=msms_timeout_s,
                force_rerun_msms=force_rerun_msms,
                max_residues=max_residues,
            )
        )
    return results


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm 

    # process all structures under a given directory
    dataset_root = Path('/data1/home/zhujunjie/datasets/hybrid')
    dataset_info = pd.read_csv(dataset_root / 'train_manifest_hard.csv')

    output_dir = dataset_root / 'processed_pepo'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Runtime knobs (environment variables):
    # - VISPEPO_USE_KEOPS: "1" or "0" (default: "1")
    # - VISPEPO_NUM_WORKERS: process count (default: "1" -> sequential)
    # - VISPEPO_TORCH_THREADS: torch threads per process (default: "8")
    # - VISPEPO_MSMS_TIMEOUT_S: timeout in seconds per MSMS call (default: "180")
    # - VISPEPO_FORCE_RERUN_MSMS: "1" to ignore cached .vert/.face and rerun MSMS
    # - VISPEPO_MAX_RESIDUES: skip structures with residues above this threshold
    # - VISPEPO_GPU_IDS: comma-separated GPU ids for workers, e.g. "0" or "0,1"
    use_keops = os.environ.get("VISPEPO_USE_KEOPS", "1") != "0"
    num_workers = int(os.environ.get("VISPEPO_NUM_WORKERS", "1"))
    torch_threads = int(os.environ.get("VISPEPO_TORCH_THREADS", "8"))
    msms_timeout_s = int(os.environ.get("VISPEPO_MSMS_TIMEOUT_S", "20"))
    force_rerun_msms = os.environ.get("VISPEPO_FORCE_RERUN_MSMS", "0") == "1"
    max_residues = int(os.environ.get("VISPEPO_MAX_RESIDUES", "1024"))
    gpu_ids = [int(x) for x in os.environ.get("VISPEPO_GPU_IDS", "0").split(",") if x.strip() != ""]
    if len(gpu_ids) == 0:
        gpu_ids = [0]
    enable_timing = True

    rows = [row.to_dict() for _, row in dataset_info.iterrows()]

    # if processed files already exist, skip
    for row_dict in rows:
        pdb_id = str(row_dict["group_id"])
        if (
            os.path.exists(output_dir / pdb_id / "receptor.pt")
            and os.path.exists(output_dir / pdb_id / "peptide.pt")
            # and os.path.exists(output_dir / pdb_id / "label.pt")
        ):
            rows.remove(row_dict)

    print(f"{len(rows)} rows to process")

    if num_workers <= 1:
        for row_dict in tqdm(rows):
            pdb_id, ok, err = _process_pair_row(
                row_dict,
                output_dir=output_dir,
                use_keops=use_keops,
                torch_threads=torch_threads,
                enable_timing=enable_timing,
                msms_timeout_s=msms_timeout_s,
                force_rerun_msms=force_rerun_msms,
                max_residues=max_residues,
            )
            if not ok:
                print(f"Error processing {pdb_id}: {err}")
    else:
        # GPU/KeOps-friendly parallel mode: each worker processes one shard sequentially.
        # Using spawn context avoids CUDA/KeOps issues with forked processes.
        row_subsets = [rows[i::num_workers] for i in range(num_workers)]
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as executor:
            futures = [
                executor.submit(
                    _process_row_subset,
                    row_subsets[worker_id],
                    output_dir,
                    use_keops,
                    torch_threads,
                    enable_timing,
                    msms_timeout_s,
                    force_rerun_msms,
                    max_residues,
                    gpu_ids[worker_id % len(gpu_ids)],
                )
                for worker_id in range(num_workers)
            ]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                subset_results = fut.result()
                for pdb_id, ok, err in subset_results:
                    if not ok:
                        print(f"Error processing {pdb_id}: {err}")
        