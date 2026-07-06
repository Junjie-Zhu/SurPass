import pickle
import random
from math import prod
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from scipy.spatial.transform import Rotation as Scipy_Rotation
from torch.utils.data import DataLoader, Dataset

DISTANCE_BIN_START_A = 2.0
DISTANCE_BIN_WIDTH_A = 0.5
DISTANCE_BIN_NUM = 36
ATOM14_CA_INDEX = 1
ATOM14_CB_INDEX = 4


def distance_to_bins(
    distance: torch.Tensor,
    bin_start: float = DISTANCE_BIN_START_A,
    bin_width: float = DISTANCE_BIN_WIDTH_A,
    bin_count: int = DISTANCE_BIN_NUM,
) -> torch.Tensor:
    bin_index = torch.floor((distance.to(dtype=torch.float32) - bin_start) / bin_width)
    return bin_index.clamp(min=0, max=bin_count - 1).to(dtype=torch.long)


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
        feature_batch = {}
        for key in samples[0].keys():
            values = [sample[key] for sample in samples]
            if not isinstance(values[0], torch.Tensor):
                feature_batch[key] = values
                continue
            feature_batch[key] = _pad_first_dim(values)
        return feature_batch

    proteins, peptides, labels_2d_bins, labels_2d_mask = zip(*batch)
    protein_batch = _collate_complex(list(proteins))
    peptide_batch = _collate_complex(list(peptides))

    label_batch = {
        "label_2d_bins": _pad_first_two_dims([x.to(dtype=torch.long) for x in labels_2d_bins]),
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
        root_dir: str,
        cluster_tsv_path=None,
        transform=None,
        center_coordinates: bool = True,
        random_rotation: bool = True,
        distance_bin_start: float = DISTANCE_BIN_START_A,
        distance_bin_width: float = DISTANCE_BIN_WIDTH_A,
        distance_bin_count: int = DISTANCE_BIN_NUM,
    ):
        self.root_dir = Path(root_dir)
        self.cluster_tsv_path = Path(cluster_tsv_path)
        self.transform = transform
        self.center_coordinates = center_coordinates
        self.random_rotation = random_rotation
        self.distance_bin_start = float(distance_bin_start)
        self.distance_bin_width = float(distance_bin_width)
        self.distance_bin_count = int(distance_bin_count)

        if self.distance_bin_count <= 0:
            raise ValueError("distance_bin_count must be positive.")

        self.metadata = pd.read_csv(self.cluster_tsv_path, sep="\t")
        required_columns = {"chain1", "chain2", "label_path", "ppi_cluster_id"}
        missing_columns = required_columns.difference(self.metadata.columns)
        if missing_columns:
            raise ValueError(
                f"Cluster TSV is missing required columns: {sorted(missing_columns)}"
            )

        self.items = sorted(self.metadata["ppi_cluster_id"].dropna().unique().tolist())
        self.cluster_to_rows = {
            cluster_id: group.reset_index(drop=True)
            for cluster_id, group in self.metadata.groupby("ppi_cluster_id", sort=False)
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.get_positive_item(idx)

    def _sample_row(self, idx):
        cluster_id = self.items[idx]
        cluster_rows = self.cluster_to_rows[cluster_id]
        return cluster_rows.iloc[random.randrange(len(cluster_rows))]

    def _load_pair_features(self, row) -> tuple[dict, dict]:
        p1_features = self._process_complex(self._load_chain_features(row["chain1"]))
        p2_features = self._process_complex(self._load_chain_features(row["chain2"]))
        return p1_features, p2_features

    def get_positive_item(self, idx):
        row = self._sample_row(idx)
        p1_features, p2_features = self._load_pair_features(row)
        label_path = self.root_dir / "labels" / Path(row["label_path"]).name
        label_bins, label_mask = self._load_pair_labels(
            label_path, p1_features["mask"].shape[0], p2_features["mask"].shape[0]
        )

        sample = (p1_features, p2_features, label_bins, label_mask)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_negative_item(self, p1_idx: int, p2_idx: int):
        if self.items[p1_idx] == self.items[p2_idx]:
            raise ValueError("Negative samples must use chains from different PPI clusters.")

        p1_row = self._sample_row(p1_idx)
        p2_row = self._sample_row(p2_idx)
        p1_features = self._process_complex(self._load_chain_features(p1_row["chain1"]))
        p2_features = self._process_complex(self._load_chain_features(p2_row["chain2"]))

        label_mask = p1_features["mask"][:, None].to(dtype=torch.bool) & p2_features[
            "mask"
        ][None, :].to(dtype=torch.bool)
        label_bins = torch.full(
            label_mask.shape,
            fill_value=self.distance_bin_count - 1,
            dtype=torch.long,
        )
        sample = (p1_features, p2_features, label_bins, label_mask)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def _to_tensor(value, dtype=None) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    @staticmethod
    def _encode_chain_index(chain_index) -> torch.Tensor:
        if isinstance(chain_index, torch.Tensor):
            if chain_index.dtype.is_floating_point:
                return chain_index.to(dtype=torch.long)
            return chain_index.long()

        values = list(chain_index)
        mapping = {value: i for i, value in enumerate(sorted(set(values)))}
        return torch.tensor([mapping[value] for value in values], dtype=torch.long)

    def _load_pickle(self, path: Path) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_chain_features(self, chain_name: str) -> dict:
        path = self.root_dir / "processed" / f"{chain_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing chain feature pickle: {path}")
        return self._load_pickle(path)

    def _load_pair_labels(
        self,
        label_path: str,
        protein_len: int,
        peptide_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        label_data = self._load_pickle(label_path)
        pairwise_dist = self._to_tensor(label_data["pairwise_dist"], dtype=torch.float32)
        pairwise_mask = self._to_tensor(label_data["pairwise_mask"], dtype=torch.bool)
        expected_shape = (protein_len, peptide_len)
        if tuple(pairwise_dist.shape) != expected_shape:
            raise ValueError(
                f"pairwise_dist shape {tuple(pairwise_dist.shape)} does not match "
                f"feature lengths {expected_shape}."
            )
        if tuple(pairwise_mask.shape) != expected_shape:
            raise ValueError(
                f"pairwise_mask shape {tuple(pairwise_mask.shape)} does not match "
                f"feature lengths {expected_shape}."
            )

        cross_bins = distance_to_bins(
            pairwise_dist,
            bin_start=self.distance_bin_start,
            bin_width=self.distance_bin_width,
            bin_count=self.distance_bin_count,
        )

        return cross_bins, pairwise_mask

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
        residue_features = {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in residue_features.items()
        }
        residue_features["atom14_positions"] = self._to_tensor(
            residue_features["atom14_positions"], dtype=torch.float32
        )
        residue_features["atom14_mask"] = self._to_tensor(
            residue_features["atom14_mask"], dtype=torch.bool
        )
        residue_features["cb_positions"] = self._to_tensor(
            residue_features["cb_positions"], dtype=torch.float32
        )
        residue_features["cb_mask"] = self._to_tensor(
            residue_features["cb_mask"], dtype=torch.bool
        )
        residue_features["residue_type"] = self._to_tensor(
            residue_features["residue_type"], dtype=torch.long
        )
        residue_features["residue_index"] = self._to_tensor(
            residue_features["residue_index"], dtype=torch.long
        )
        residue_features["chain_index"] = self._encode_chain_index(
            residue_features["chain_index"]
        )

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

        residue_features["atom14_positions"] = self._apply_rigid(
            residue_features["atom14_positions"], center, rot
        )
        residue_features["cb_positions"] = self._apply_rigid(
            residue_features["cb_positions"], center, rot
        )

        return {
            "plm_emb": torch.zeros(
                residue_features["residue_type"].shape[0],
                1280,
                dtype=torch.float32,
            ),
            "residue_type": residue_features["residue_type"],
            "residue_index": residue_features["residue_index"],
            "residue_position": residue_features["cb_positions"],
            "chain_index": residue_features["chain_index"],
            "mask": residue_features["cb_mask"],
            "atom14_positions": residue_features["atom14_positions"],
            "atom14_mask": residue_features["atom14_mask"],
        }


class BalancedClusterDataset(Dataset):
    def __init__(
        self,
        dataset: PepoTrainDataset,
        indices: Sequence[int] | None = None,
        distance_bin_count: int = DISTANCE_BIN_NUM,
        negative_ratio: int = 1,
    ):
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else list(indices)
        self.distance_bin_count = int(distance_bin_count)
        self.negative_ratio = int(negative_ratio)
        if self.negative_ratio < 0:
            raise ValueError("negative_ratio must be non-negative.")
        if self.negative_ratio > 0 and len(self.indices) < 2:
            raise ValueError("At least two PPI clusters are required to create negatives.")

    def __len__(self):
        return len(self.indices) * (1 + self.negative_ratio)

    def __getitem__(self, idx):
        positive_count = len(self.indices)
        if idx < positive_count:
            return self.dataset[self.indices[idx]]

        negative_idx = idx - positive_count
        source_pos = negative_idx % positive_count
        source_idx = self.indices[source_pos]
        candidate_indices = [i for i in self.indices if i != source_idx]
        target_idx = random.choice(candidate_indices)
        return self.dataset.get_negative_item(source_idx, target_idx)
