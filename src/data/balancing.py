import math
import random
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler, Subset, random_split

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


class StaticClusterDataset(Dataset):
    def __init__(
        self,
        dataset: PepoTrainDataset,
        indices: Sequence[int],
        is_positive: bool,
    ):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.is_positive = bool(is_positive)
        if len(self.indices) == 0:
            raise ValueError("Static cluster dataset has no clusters.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        source_idx = self.indices[idx]
        if self.is_positive:
            return self.dataset.get_positive_item(source_idx)
        return self.dataset.get_static_negative_item(source_idx)


class DistributedWeightedSampler(Sampler[int]):
    """Draw a fixed epoch with replacement and give each DDP rank an equal share."""

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        replacement: bool = True,
    ):
        if int(num_replicas) <= 0:
            raise ValueError("num_replicas must be positive.")
        if int(rank) < 0 or int(rank) >= int(num_replicas):
            raise ValueError("rank must be in [0, num_replicas).")
        if int(num_samples) <= 0:
            raise ValueError("num_samples must be positive.")
        super().__init__()

        weight_tensor = torch.as_tensor(weights, dtype=torch.double)
        if weight_tensor.ndim != 1 or weight_tensor.numel() == 0:
            raise ValueError("weights must be a non-empty 1D tensor.")
        if torch.any(weight_tensor < 0) or float(weight_tensor.sum()) <= 0:
            raise ValueError("weights must be non-negative and have a positive sum.")

        self.weights = weight_tensor
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.replacement = bool(replacement)
        self.total_size = int(math.ceil(int(num_samples) / self.num_replicas) * self.num_replicas)
        self.num_samples_per_rank = self.total_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples_per_rank

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        draws = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        )
        return iter(draws[self.rank :: self.num_replicas].tolist())


def _mapping_get(source: Mapping, key: str, default=None):
    if key not in source or source[key] is None:
        return default
    return source[key]


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return True
        if lowered in {"0", "false", "no"}:
            return False
        raise ValueError(f"Cannot interpret boolean value: {value!r}")
    return bool(value)


def _cluster_indices(dataset: Dataset) -> tuple[PepoTrainDataset, list[int]]:
    if isinstance(dataset, Subset):
        if not isinstance(dataset.dataset, PepoTrainDataset):
            raise TypeError(f"Unsupported subset dataset type: {type(dataset.dataset)}")
        return dataset.dataset, [int(index) for index in dataset.indices]
    if isinstance(dataset, PepoTrainDataset):
        return dataset, list(range(len(dataset)))
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def _wrap_source_split(
    split: Dataset,
    *,
    assemble_negatives: bool,
    is_positive: bool,
    negative_ratio: int,
    distance_bin_count: int,
) -> Dataset:
    if assemble_negatives:
        return _as_balanced_cluster_dataset(split, negative_ratio, distance_bin_count)
    base, indices = _cluster_indices(split)
    return StaticClusterDataset(base, indices, is_positive=is_positive)


def source_sample_weights(
    lengths: Sequence[int],
    proportions: Sequence[float],
) -> tuple[torch.Tensor, list[float]]:
    """Turn per-dataset percentages into per-sample draw weights.

    Each proportion is the percent of an epoch taken from that dataset, and the
    proportions must sum to 100. A value of 50 means half of the sampled data.
    """
    if len(lengths) == 0:
        raise ValueError("At least one dataset source is required.")
    if len(lengths) != len(proportions):
        raise ValueError("Each dataset source needs one weight.")
    if any(int(length) <= 0 for length in lengths):
        raise ValueError("Each dataset source must contain at least one training sample.")

    percentages = [float(value) for value in proportions]
    if any(value <= 0.0 for value in percentages):
        raise ValueError("Dataset weights must be positive percentages.")
    total = float(sum(percentages))
    if abs(total - 100.0) > 1e-3:
        raise ValueError(f"Dataset weights must sum to 100, got {total:g}.")

    probabilities = [value / 100.0 for value in percentages]
    pieces = [
        torch.full((int(length),), value / int(length), dtype=torch.double)
        for length, value in zip(lengths, percentages)
    ]
    return torch.cat(pieces), probabilities


def build_multi_source_split(
    sources: Sequence[Mapping],
    *,
    root_dir: str,
    test_fraction: float,
    seed: int,
    negative_ratio: int,
    dataset_kwargs: Mapping | None = None,
) -> tuple[Dataset, Dataset, torch.Tensor, list[dict]]:
    if len(sources) == 0:
        raise ValueError("At least one dataset source is required.")

    shared_kwargs = dict(dataset_kwargs or {})
    shared_kwargs.pop("require_labels", None)
    shared_kwargs.pop("root_dir", None)
    shared_kwargs.pop("cluster_tsv_path", None)
    distance_bin_count = int(shared_kwargs.get("distance_bin_count", DISTANCE_BIN_NUM))
    parent = Path(root_dir)

    train_views: list[Dataset] = []
    test_views: list[Dataset] = []
    manual_weights: list[float] = []
    descriptions: list[dict] = []

    for index, source in enumerate(sources):
        name = str(_mapping_get(source, "name", f"source_{index}"))
        subdir = str(_mapping_get(source, "subdir", "") or "")
        cluster_tsv = _mapping_get(source, "cluster_tsv", None)
        if not cluster_tsv:
            raise ValueError(f"Dataset source {name!r} is missing cluster_tsv.")
        assemble_negatives = _as_bool(_mapping_get(source, "assemble_negatives", False), False)
        is_positive = _as_bool(_mapping_get(source, "is_positive", True), True)
        weight = float(_mapping_get(source, "weight", 1.0))

        source_root = parent / subdir if subdir else parent
        cluster_path = Path(str(cluster_tsv))
        if not cluster_path.is_absolute():
            cluster_path = source_root / cluster_path

        dataset = PepoTrainDataset(
            root_dir=str(source_root),
            cluster_tsv_path=str(cluster_path),
            require_labels=assemble_negatives or is_positive,
            **shared_kwargs,
        )
        train_split, test_split = split_dataset(
            dataset,
            test_fraction=test_fraction,
            seed=seed,
        )
        train_view = _wrap_source_split(
            train_split,
            assemble_negatives=assemble_negatives,
            is_positive=is_positive,
            negative_ratio=negative_ratio,
            distance_bin_count=distance_bin_count,
        )
        test_view = _wrap_source_split(
            test_split,
            assemble_negatives=assemble_negatives,
            is_positive=is_positive,
            negative_ratio=negative_ratio,
            distance_bin_count=distance_bin_count,
        )
        if len(train_view) == 0 or len(test_view) == 0:
            raise ValueError(f"Dataset source {name!r} produced an empty split.")

        train_views.append(train_view)
        test_views.append(test_view)
        manual_weights.append(weight)
        descriptions.append(
            {
                "name": name,
                "train_length": len(train_view),
                "test_length": len(test_view),
                "assemble_negatives": assemble_negatives,
                "is_positive": is_positive,
            }
        )

    if len(train_views) == 1:
        descriptions[0]["probability"] = 1.0
        weights = torch.ones(int(descriptions[0]["train_length"]), dtype=torch.double)
    else:
        weights, probabilities = source_sample_weights(
            [item["train_length"] for item in descriptions],
            manual_weights,
        )
        for item, probability in zip(descriptions, probabilities):
            item["probability"] = probability

    train_dataset = train_views[0] if len(train_views) == 1 else ConcatDataset(train_views)
    test_dataset = test_views[0] if len(test_views) == 1 else ConcatDataset(test_views)
    return train_dataset, test_dataset, weights, descriptions
