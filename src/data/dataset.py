import gzip
import io
import pickle
import random
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.cropping import (
    DEFAULT_CONTACT_THRESHOLD_A,
    DEFAULT_CROP_SIZE,
    crop_negative_pair,
    crop_positive_pair,
)
from src.data.labels import (
    DISTANCE_BIN_END_A,
    DISTANCE_BIN_NUM,
    DISTANCE_BIN_START_A,
    distance_to_bins,
)


class PairSample(NamedTuple):
    p1_features: dict
    p2_features: dict
    label_inter_bins: torch.Tensor
    label_inter_mask: torch.Tensor
    crop_is_positive: bool
    source_is_positive: bool
    metadata: dict[str, Any]


def _load_embedded_file(path: Path) -> dict:
    """Load a plain torch archive, or a gzip-compressed PDB/AFDB one, in one read."""
    with path.open("rb") as handle:
        payload = handle.read()
    if payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)
    return torch.load(io.BytesIO(payload), weights_only=False, mmap=False)


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
        contact_threshold: float = DEFAULT_CONTACT_THRESHOLD_A,
        require_labels: bool = True,
        distance_bin_convention: str = "legacy",
    ):
        self.root_dir = Path(root_dir)
        self.cluster_tsv_path = Path(cluster_tsv_path)
        self.transform = transform
        self.require_labels = bool(require_labels)
        self.distance_bin_convention = distance_bin_convention
        self.distance_bin_start = float(distance_bin_start)
        self.distance_bin_end = float(distance_bin_end)
        self.distance_bin_count = int(distance_bin_count)
        self.crop_size = None if crop_size is None else int(crop_size)
        self.contact_threshold = float(contact_threshold)

        if self.distance_bin_count <= 0:
            raise ValueError("distance_bin_count must be positive.")
        if self.crop_size is not None and self.crop_size < 2:
            raise ValueError("crop_size must be at least 2 when provided.")

        self.metadata = pd.read_csv(self.cluster_tsv_path, sep="\t").drop(
            columns=["sample_confidence", "confidence"], errors="ignore",
        )
        if "ppi_cluster_id" not in self.metadata.columns:
            if "pair_id" not in self.metadata.columns:
                raise ValueError(
                    "Pair manifest must contain ppi_cluster_id or pair_id."
                )
            self.metadata["ppi_cluster_id"] = self.metadata["pair_id"].astype(str)
        else:
            missing_cluster = self.metadata["ppi_cluster_id"].isna()
            if missing_cluster.any() and "pair_id" in self.metadata.columns:
                self.metadata.loc[missing_cluster, "ppi_cluster_id"] = (
                    self.metadata.loc[missing_cluster, "pair_id"].astype(str)
                )
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
        return self.sample(idx)

    def sample(self, idx, *, rng=random, row_index=None):
        cluster_rows = self.cluster_to_rows[self.items[idx]]
        row_index = rng.randrange(len(cluster_rows)) if row_index is None else row_index
        row = cluster_rows.iloc[row_index]
        if "source_is_positive" not in self.metadata.columns:
            sample = self._positive_sample_from_row(row, rng=rng)
            return self.transform(sample) if self.transform is not None else sample
        if self._as_bool(row["source_is_positive"]):
            sample = self._positive_sample_from_row(row, rng=rng)
            return self.transform(sample) if self.transform is not None else sample
        p1_features, p2_features = self._load_pair_features(row)
        return self._finalize_negative_sample(
            p1_features,
            p2_features,
            negative_kind=str(self._row_value(row, "negative_kind", "curated")),
            rng=rng,
        )

    def _sample_row(self, idx):
        cluster_id = self.items[idx]
        cluster_rows = self.cluster_to_rows[cluster_id]
        return cluster_rows.iloc[random.randrange(len(cluster_rows))]

    def _load_pair_features(self, row) -> tuple[dict, dict]:
        p1_features = self._process_complex(self._load_chain_features(row["chain1"]))
        p2_features = self._process_complex(self._load_chain_features(row["chain2"]))
        return p1_features, p2_features

    def _positive_sample_from_row(self, row, *, rng=random):
        p1_features, p2_features = self._load_pair_features(row)
        label_path = self.root_dir / "labels" / Path(row["label_path"]).name
        pairwise_dist, pairwise_mask = self._load_pair_distance_and_mask(
            label_path, p1_features["mask"].shape[0], p2_features["mask"].shape[0]
        )
        (
            p1_features,
            p2_features,
            pairwise_dist,
            pairwise_mask,
            crop_is_positive,
            crop_metadata,
        ) = crop_positive_pair(
            p1_features,
            p2_features,
            pairwise_dist,
            pairwise_mask,
            crop_size=self.crop_size,
            contact_threshold=self.contact_threshold,
            rng=rng,
        )
        p1_length = int(p1_features["mask"].shape[0])
        label_inter_dist = pairwise_dist[:p1_length, p1_length:]
        label_inter_mask = pairwise_mask[:p1_length, p1_length:].to(dtype=torch.bool)
        label_inter_bins = distance_to_bins(
            label_inter_dist,
            bin_start=self.distance_bin_start,
            bin_end=self.distance_bin_end,
            bin_count=self.distance_bin_count,
            convention=self.distance_bin_convention,
        )
        crop_metadata.update(
            {
                "source_kind": "positive",
            }
        )
        return PairSample(
            p1_features,
            p2_features,
            label_inter_bins,
            label_inter_mask,
            crop_is_positive,
            True,
            crop_metadata,
        )

    @staticmethod
    def _row_value(row, key: str, default):
        value = row.get(key, default)
        return default if pd.isna(value) else value

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not pd.isna(value):
            return bool(value)
        lowered = str(value).strip().lower()
        if lowered in {"true", "1", "yes", "positive", "pos"}:
            return True
        if lowered in {"false", "0", "no", "negative", "neg"}:
            return False
        raise ValueError(f"Unsupported source_is_positive value: {value!r}")

    def _finalize_negative_sample(
        self,
        p1_features: dict,
        p2_features: dict,
        *,
        negative_kind: str,
        rng=random,
    ):
        p1_features, p2_features, crop_metadata = crop_negative_pair(
            p1_features,
            p2_features,
            crop_size=self.crop_size,
            rng=rng,
        )
        p1_mask = p1_features["mask"].to(dtype=torch.bool)
        p2_mask = p2_features["mask"].to(dtype=torch.bool)
        label_inter_bins = torch.full(
            (int(p1_mask.shape[0]), int(p2_mask.shape[0])),
            self.distance_bin_count - 1,
            dtype=torch.long,
            device=p1_mask.device,
        )
        label_inter_mask = p1_mask[:, None] & p2_mask[None, :]
        crop_metadata.update(
            {
                "source_kind": str(negative_kind),
            }
        )
        sample = PairSample(
            p1_features,
            p2_features,
            label_inter_bins,
            label_inter_mask,
            False,
            False,
            crop_metadata,
        )
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
        return _load_embedded_file(path)

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
            "chain_index": residue_features["chain_index"],
            "mask": residue_features["cb_mask"],
        }
