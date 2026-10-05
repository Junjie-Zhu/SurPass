import random
from typing import Sequence

import torch
from torch.utils.data import Dataset, Subset, random_split

from src.data.dataset import PepoTrainDataset
from src.data.labels import DISTANCE_BIN_NUM


def split_dataset(
    dataset: Dataset,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if len(dataset) == 0:
        raise ValueError("Cannot split an empty dataset.")
    if len(dataset) == 1:
        return dataset, dataset

    test_len = max(1, int(round(len(dataset) * test_fraction)))
    train_len = len(dataset) - test_len
    if train_len == 0:
        train_len, test_len = 1, len(dataset) - 1
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, test_len], generator=generator)


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
            source_idx = self.indices[idx]
            return self.dataset.get_positive_item(source_idx)

        negative_idx = idx - positive_count
        source_pos = negative_idx % positive_count
        source_idx = self.indices[source_pos]
        candidate_indices = [i for i in self.indices if i != source_idx]
        target_idx = random.choice(candidate_indices)
        return self.dataset.get_negative_item(source_idx, target_idx)


def _as_balanced_cluster_dataset(
    dataset: Dataset,
    negative_ratio: int,
    distance_bin_count: int,
) -> Dataset:
    if negative_ratio <= 0:
        return dataset
    if isinstance(dataset, Subset) and isinstance(dataset.dataset, PepoTrainDataset):
        if len(dataset.indices) < 2:
            return dataset
        return BalancedClusterDataset(
            dataset.dataset,
            indices=list(dataset.indices),
            distance_bin_count=distance_bin_count,
            negative_ratio=negative_ratio,
        )
    if isinstance(dataset, PepoTrainDataset):
        if len(dataset) < 2:
            return dataset
        return BalancedClusterDataset(
            dataset,
            distance_bin_count=distance_bin_count,
            negative_ratio=negative_ratio,
        )
    raise TypeError(f"Unsupported dataset type for negative sampling: {type(dataset)}")


def create_balanced_split_datasets(
    dataset: PepoTrainDataset,
    test_fraction: float,
    seed: int,
    negative_ratio: int,
    distance_bin_count: int,
) -> tuple[Dataset, Dataset]:
    train_dataset, test_dataset = split_dataset(dataset, test_fraction=test_fraction, seed=seed)
    return (
        _as_balanced_cluster_dataset(train_dataset, negative_ratio, distance_bin_count),
        _as_balanced_cluster_dataset(test_dataset, negative_ratio, distance_bin_count),
    )
