import copy
import os
from math import prod
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Scipy_Rotation
from torch.utils.data import DataLoader, Dataset

DISTANCE_BIN_START_A = 2.0
DISTANCE_BIN_END_A = 20.0
DISTANCE_BIN_NUM = 36
ATOM14_CA_INDEX = 1
ATOM14_CB_INDEX = 4


def _resolve_atom14_tensors(complex_features: dict) -> tuple[torch.Tensor, torch.Tensor]:
    atom_features = complex_features["atom_features"]
    atom_position = atom_features.get("atom14_position", atom_features.get("atom_position"))
    atom_mask = atom_features.get("atom14_mask", atom_features.get("mask"))
    if atom_position is None or atom_mask is None:
        raise KeyError("atom14_position/atom14_mask (or atom_position/mask) are required.")
    return atom_position, atom_mask


def _representative_cb_or_ca(
    atom14_position: torch.Tensor,
    atom14_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    atom14_position = atom14_position.to(dtype=torch.float32)
    atom14_mask = atom14_mask.to(dtype=torch.bool)

    cb_pos = atom14_position[:, ATOM14_CB_INDEX, :]
    ca_pos = atom14_position[:, ATOM14_CA_INDEX, :]
    cb_mask = atom14_mask[:, ATOM14_CB_INDEX]
    ca_mask = atom14_mask[:, ATOM14_CA_INDEX]

    # Use CB when present; fallback to CA (e.g. GLY or missing CB).
    rep_pos = torch.where(cb_mask[:, None], cb_pos, ca_pos)
    rep_mask = cb_mask | ca_mask
    return rep_pos, rep_mask


def build_pair_distance_label(
    protein_features: dict,
    peptide_features: dict,
) -> dict:
    protein_atom14_position, protein_atom14_mask = _resolve_atom14_tensors(protein_features)
    peptide_atom14_position, peptide_atom14_mask = _resolve_atom14_tensors(peptide_features)

    protein_rep_pos, protein_rep_mask = _representative_cb_or_ca(
        protein_atom14_position, protein_atom14_mask
    )
    peptide_rep_pos, peptide_rep_mask = _representative_cb_or_ca(
        peptide_atom14_position, peptide_atom14_mask
    )

    n_protein = int(protein_rep_pos.shape[0])
    n_peptide = int(peptide_rep_pos.shape[0])
    total_len = n_protein + n_peptide

    cross_mask = protein_rep_mask[:, None] & peptide_rep_mask[None, :]

    if n_protein == 0 or n_peptide == 0:
        cross_bins = protein_rep_pos.new_zeros((n_protein, n_peptide, DISTANCE_BIN_NUM))
    else:
        dist = torch.cdist(protein_rep_pos, peptide_rep_pos, p=2)
        bin_edges = torch.linspace(
            DISTANCE_BIN_START_A,
            DISTANCE_BIN_END_A,
            DISTANCE_BIN_NUM + 1,
            dtype=dist.dtype,
            device=dist.device,
        )
        bin_index = torch.bucketize(dist, bin_edges[1:-1])
        cross_bins = F.one_hot(bin_index, num_classes=DISTANCE_BIN_NUM).to(dtype=dist.dtype)
        cross_bins = cross_bins * cross_mask[..., None].to(dtype=dist.dtype)

    full_bins = cross_bins.new_zeros((total_len, total_len, DISTANCE_BIN_NUM))
    full_mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=cross_mask.device)

    full_bins[:n_protein, n_protein:, :] = cross_bins
    full_bins[n_protein:, :n_protein, :] = cross_bins.transpose(0, 1)
    full_mask[:n_protein, n_protein:] = cross_mask
    full_mask[n_protein:, :n_protein] = cross_mask.transpose(0, 1)

    return {
        "pair_distance_bins_2d": full_bins.to(dtype=torch.float32),
        "pair_distance_mask_2d": full_mask,
    }


def collate_fn(batch):
    def _pad_value(t: torch.Tensor):
        if t.dtype == torch.bool:
            return False
        if torch.is_floating_point(t):
            return 0.0
        return 0

    def _pad_first_dim(tensors: list[torch.Tensor]) -> torch.Tensor:
        max_n = max(t.shape[0] for t in tensors)
        out_shape = (len(tensors), max_n, *tensors[0].shape[1:])
        out = tensors[0].new_full(out_shape, fill_value=_pad_value(tensors[0]))
        for i, t in enumerate(tensors):
            out[i, : t.shape[0]] = t
        return out

    def _pad_first_two_dims(tensors: list[torch.Tensor]) -> torch.Tensor:
        max_n = max(t.shape[0] for t in tensors)
        max_m = max(t.shape[1] for t in tensors)
        out_shape = (len(tensors), max_n, max_m, *tensors[0].shape[2:])
        out = tensors[0].new_full(out_shape, fill_value=_pad_value(tensors[0]))
        for i, t in enumerate(tensors):
            out[i, : t.shape[0], : t.shape[1]] = t
        return out

    def _collate_complex(samples: list[dict]) -> dict:
        batch_out = {}
        feature_samples = [s["residue_features"] for s in samples]
        feature_batch = {}
        for key in feature_samples[0].keys():
            values = [fs[key] for fs in feature_samples]
            if not isinstance(values[0], torch.Tensor):
                feature_batch[key] = values
                continue
            feature_batch[key] = _pad_first_dim(values)
        batch_out["residue_features"] = feature_batch
        return batch_out

    proteins, peptides, labels_2d_bins, labels_2d_mask = zip(*batch)
    protein_batch = _collate_complex(list(proteins))
    peptide_batch = _collate_complex(list(peptides))

    label_batch = {
        "label_2d_bins": _pad_first_two_dims([x.to(dtype=torch.float32) for x in labels_2d_bins]),
        "label_2d_mask": _pad_first_two_dims([x.to(dtype=torch.bool) for x in labels_2d_mask]),
    }
    return protein_batch, peptide_batch, label_batch


def get_dataloader(
    dataset,
    collate_fn,
    batch_size,
    shuffle=True,
    num_workers=4,
    sampler=None,
    pin_memory=True,
    drop_last=False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


class PepoTrainDataset(Dataset):
    def __init__(
        self,
        cluster_csv_path=None,
        transform=None,
        center_coordinates: bool = True,
        random_rotation: bool = True,
    ):
        if metadata is None:
            if csv_path is None:
                raise ValueError("Either `csv_path` or `metadata` must be provided.")
            self.metadata = pd.read_csv(csv_path)
        else:
            self.metadata = metadata.reset_index(drop=True).copy()

        self.transform = transform
        self.center_coordinates = center_coordinates
        self.random_rotation = random_rotation

        if "protein_path" in self.metadata.columns:
            self.protein_path_column = "protein_path"
        elif "receptor_path" in self.metadata.columns:
            self.protein_path_column = "receptor_path"
        else:
            raise KeyError("CSV must contain either `protein_path` or `receptor_path`.")

        if "peptide_path" not in self.metadata.columns:
            raise KeyError("CSV must contain `peptide_path`.")
        self.peptide_path_column = "peptide_path"

        self.items = [
            (protein_path, peptide_path)
            for protein_path, peptide_path in zip(
                self.metadata[self.protein_path_column],
                self.metadata[self.peptide_path_column],
            )
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        protein_path, peptide_path = self.items[idx]
        protein = torch.load(protein_path)
        peptide = torch.load(peptide_path)
        label_2d_bins, label_2d_mask = self._ensure_pair_distance_label_file(
            protein_path=protein_path,
            peptide_path=peptide_path,
            protein_features=protein,
            peptide_features=peptide,
        )

        protein = self._process_complex(protein)
        peptide = self._process_complex(peptide)

        item = (protein, peptide, label_2d_bins, label_2d_mask)
        if self.transform is not None:
            item = self.transform(item)
        return item

    @staticmethod
    def _infer_label_path(
        protein_path: str | Path,
        peptide_path: str | Path,
    ) -> Path | None:
        protein_parent = Path(protein_path).parent
        peptide_parent = Path(peptide_path).parent
        if protein_parent == peptide_parent:
            return protein_parent / "label.pt"
        return None

    @staticmethod
    def _safe_save_label(label_path: Path, label_data: dict) -> None:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = label_path.with_suffix(label_path.suffix + ".tmp")
        torch.save(label_data, tmp_path)
        os.replace(tmp_path, label_path)

    @staticmethod
    def _is_label_shape_consistent(
        protein_len: int,
        peptide_len: int,
        pair_distance_bins_2d: torch.Tensor,
        pair_distance_mask_2d: torch.Tensor,
    ) -> bool:
        total_len = protein_len + peptide_len
        return (
            tuple(pair_distance_bins_2d.shape) == (total_len, total_len, DISTANCE_BIN_NUM)
            and tuple(pair_distance_mask_2d.shape) == (total_len, total_len)
        )

    def _ensure_pair_distance_label_file(
        self,
        protein_path: str | Path,
        peptide_path: str | Path,
        protein_features: dict,
        peptide_features: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        label_path = self._infer_label_path(
            protein_path=protein_path,
            peptide_path=peptide_path,
        )

        protein_len = int(_resolve_atom14_tensors(protein_features)[0].shape[0])
        peptide_len = int(_resolve_atom14_tensors(peptide_features)[0].shape[0])

        if label_path is not None and label_path.exists():
            label_dict = torch.load(label_path)
            bins_key = "pair_distance_bins_2d"
            mask_key = "pair_distance_mask_2d"
            if bins_key in label_dict and mask_key in label_dict:
                pair_distance_bins_2d = label_dict[bins_key]
                pair_distance_mask_2d = label_dict[mask_key]
                if self._is_label_shape_consistent(
                    protein_len,
                    peptide_len,
                    pair_distance_bins_2d,
                    pair_distance_mask_2d,
                ):
                    return pair_distance_bins_2d, pair_distance_mask_2d

        label_data = build_pair_distance_label(protein_features, peptide_features)
        if label_path is not None:
            self._safe_save_label(label_path, label_data)
        return label_data["pair_distance_bins_2d"], label_data["pair_distance_mask_2d"]

    @staticmethod
    def sample_uniform_rotation(shape=(), dtype=None, device=None) -> torch.Tensor:
        return torch.tensor(
            Scipy_Rotation.random(prod(shape)).as_matrix(),
            device=device,
            dtype=dtype,
        ).reshape(*shape, 3, 3)

    @staticmethod
    def _masked_center(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask.to(dtype=torch.bool)
        mask_float = mask_bool.to(dtype=coords.dtype)
        denom = mask_float.sum().clamp_min(1.0)
        return (coords * mask_float[..., None]).sum(dim=0) / denom

    @staticmethod
    def _apply_rigid(
        coords: torch.Tensor,
        center: torch.Tensor,
        rot: torch.Tensor,
        translate: bool = True,
    ) -> torch.Tensor:
        shifted = coords - center if translate else coords
        return torch.matmul(shifted, rot)

    def _process_complex(self, residue_features: dict) -> dict:
        # use CA to center the proteins
        residue_position = residue_features["atom14_positions"][:, 1, :]
        residue_mask = residue_features["atom14_mask"][:, 1]
        center = (
            self._masked_center(residue_position, residue_mask)
            if self.center_coordinates
            else torch.zeros(3, dtype=residue_position.dtype, device=residue_position.device)
        )

        if self.random_rotation:
            rot = self.sample_uniform_rotation(
                dtype=residue_position.dtype,
                device=residue_position.device,
            )
        else:
            rot = torch.eye(3, dtype=residue_position.dtype, device=residue_position.device)

        # apply rigid transform
        return residue_features
