import pickle
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.cropping import (
    DEFAULT_CONTACT_THRESHOLD_A,
    DEFAULT_CROP_CONTIGUOUS_PROBABILITY,
    DEFAULT_CROP_NEIGHBORHOOD_SIZE,
    DEFAULT_CROP_SIZE,
    crop_negative_pair,
    crop_positive_pair,
)
from src.data.labels import (
    DISTANCE_BIN_END_A,
    DISTANCE_BIN_NUM,
    DISTANCE_BIN_START_A,
    build_negative_pair_labels,
    distance_to_bins,
)


class PepoTrainDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        cluster_tsv_path=None,
        transform=None,
        distance_bin_start: float = DISTANCE_BIN_START_A,
        distance_bin_end: float = DISTANCE_BIN_END_A,
        distance_bin_count: int = DISTANCE_BIN_NUM,
        crop_size: int | None = DEFAULT_CROP_SIZE,
        crop_neighborhood_size: int = DEFAULT_CROP_NEIGHBORHOOD_SIZE,
        crop_contiguous_probability: float = DEFAULT_CROP_CONTIGUOUS_PROBABILITY,
        contact_threshold: float = DEFAULT_CONTACT_THRESHOLD_A,
        require_labels: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.cluster_tsv_path = Path(cluster_tsv_path)
        self.transform = transform
        self.require_labels = bool(require_labels)
        self.distance_bin_start = float(distance_bin_start)
        self.distance_bin_end = float(distance_bin_end)
        self.distance_bin_count = int(distance_bin_count)
        self.crop_size = None if crop_size is None else int(crop_size)
        self.crop_neighborhood_size = int(crop_neighborhood_size)
        self.crop_contiguous_probability = float(crop_contiguous_probability)
        self.contact_threshold = float(contact_threshold)

        if self.distance_bin_count <= 0:
            raise ValueError("distance_bin_count must be positive.")
        if self.crop_size is not None and self.crop_size <= 0:
            raise ValueError("crop_size must be positive when provided.")
        if self.crop_neighborhood_size <= 0:
            raise ValueError("crop_neighborhood_size must be positive.")
        if not 0.0 <= self.crop_contiguous_probability <= 1.0:
            raise ValueError("crop_contiguous_probability must be between 0 and 1.")

        self.metadata = pd.read_csv(self.cluster_tsv_path, sep="\t")
        required_columns = {"chain1", "chain2", "ppi_cluster_id"}
        if self.require_labels:
            required_columns.add("label_path")
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
        cluster_rows = self.cluster_to_rows[self.items[idx]]
        order = list(range(len(cluster_rows)))
        random.shuffle(order)
        for row_index in order:
            sample = self._positive_sample_from_row(cluster_rows.iloc[row_index])
            if sample is not None:
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample
        raise RuntimeError(
            f"Cluster {self.items[idx]} has no structure with an inter-chain contact."
        )

    def _positive_sample_from_row(self, row):
        p1_features, p2_features = self._load_pair_features(row)
        label_path = self.root_dir / "labels" / Path(row["label_path"]).name
        pairwise_dist, pairwise_mask = self._load_pair_distance_and_mask(
            label_path, p1_features["mask"].shape[0], p2_features["mask"].shape[0]
        )
        cropped = crop_positive_pair(
            p1_features,
            p2_features,
            pairwise_dist,
            pairwise_mask,
            crop_size=self.crop_size,
            crop_contiguous_probability=self.crop_contiguous_probability,
            crop_neighborhood_size=self.crop_neighborhood_size,
            contact_threshold=self.contact_threshold,
        )
        if cropped is None:
            return None
        p1_features, p2_features, pairwise_dist, pairwise_mask = cropped
        label_bins = distance_to_bins(
            pairwise_dist,
            bin_start=self.distance_bin_start,
            bin_end=self.distance_bin_end,
            bin_count=self.distance_bin_count,
        )
        return (p1_features, p2_features, label_bins, pairwise_mask, True)

    def get_negative_item(self, p1_idx: int, p2_idx: int):
        if self.items[p1_idx] == self.items[p2_idx]:
            raise ValueError("Negative samples must use chains from different PPI clusters.")

        p1_row = self._sample_row(p1_idx)
        p2_row = self._sample_row(p2_idx)
        p1_features = self._process_complex(self._load_chain_features(p1_row["chain1"]))
        p2_features = self._process_complex(self._load_chain_features(p2_row["chain2"]))
        return self._finalize_negative_sample(p1_features, p2_features)

    def get_static_negative_item(self, idx):
        row = self._sample_row(idx)
        p1_features, p2_features = self._load_pair_features(row)
        return self._finalize_negative_sample(p1_features, p2_features)

    def _finalize_negative_sample(self, p1_features: dict, p2_features: dict):
        p1_features, p2_features = crop_negative_pair(
            p1_features,
            p2_features,
            crop_size=self.crop_size,
        )
        label_bins, label_mask = build_negative_pair_labels(
            p1_features["residue_position"],
            p1_features["mask"],
            p2_features["residue_position"],
            p2_features["mask"],
            bin_start=self.distance_bin_start,
            bin_end=self.distance_bin_end,
            bin_count=self.distance_bin_count,
        )
        sample = (p1_features, p2_features, label_bins, label_mask, False)
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
        path = self.root_dir / "embedded" / f"{chain_name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing chain feature file: {path}")
        return torch.load(path, weights_only=False)

    def _load_pair_distance_and_mask(
        self,
        label_path: str,
        protein_len: int,
        peptide_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        label_data = self._load_pickle(label_path)
        pairwise_dist = self._to_tensor(label_data["pairwise_dist"], dtype=torch.float32)
        pairwise_mask = self._to_tensor(label_data["pairwise_mask"], dtype=torch.bool)
        expected_shape = (protein_len + peptide_len, protein_len + peptide_len)
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

        return pairwise_dist, pairwise_mask

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
        residue_features["plm_emb"] = self._to_tensor(
            residue_features["plm_emb"], dtype=torch.float32
        )

        return {
            "plm_emb": residue_features["plm_emb"],
            "residue_type": residue_features["residue_type"],
            "residue_index": residue_features["residue_index"],
            "residue_position": residue_features["cb_positions"],
            "chain_index": residue_features["chain_index"],
            "mask": residue_features["cb_mask"],
            "atom14_positions": residue_features["atom14_positions"],
            "atom14_mask": residue_features["atom14_mask"],
        }
