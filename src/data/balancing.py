"""Manifest-backed multi-source sampling."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from src.data.dataset import PepoTrainDataset


class DistributedWeightedSampler(Sampler[int]):
    """Draw a fixed weighted epoch and shard it deterministically across ranks."""

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ):
        if int(num_replicas) <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= int(rank) < int(num_replicas):
            raise ValueError("rank must be in [0, num_replicas)")
        weights = torch.as_tensor(weights, dtype=torch.double)
        if weights.ndim != 1 or weights.numel() == 0 or float(weights.sum()) <= 0:
            raise ValueError("weights must be a non-empty positive 1D tensor")
        self.weights = weights
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.total_size = int(
            math.ceil(int(num_samples) / self.num_replicas) * self.num_replicas
        )
        self.num_samples_per_rank = self.total_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples_per_rank

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        return iter(draws[self.rank :: self.num_replicas].tolist())


def source_sample_weights(
    lengths: Sequence[int],
    proportions: Sequence[float],
) -> tuple[torch.Tensor, list[float]]:
    if not lengths or len(lengths) != len(proportions):
        raise ValueError("Each source requires a length and sampling proportion")
    if any(int(length) <= 0 for length in lengths):
        raise ValueError("Dataset sources cannot be empty")
    proportions = [float(value) for value in proportions]
    if any(value <= 0.0 for value in proportions):
        raise ValueError("Source proportions must be positive")
    total = sum(proportions)
    probabilities = [value / total for value in proportions]
    weights = torch.cat(
        [
            torch.full(
                (int(length),),
                probability / int(length),
                dtype=torch.double,
            )
            for length, probability in zip(lengths, probabilities)
        ]
    )
    return weights, probabilities


def _source_value(source: Mapping, key: str, default=None):
    value = source.get(key, default)
    return default if value is None else value


def _resolve_manifest(
    source_root: Path,
    source: Mapping,
    split: str,
    split_dir: Path,
) -> Path:
    value = _source_value(
        source,
        f"{split}_tsv",
        _source_value(source, f"{split}_cluster_tsv"),
    )
    if not value:
        name = str(source["name"])
        path = split_dir / "sources" / f"{name}.{split}.tsv"
    else:
        path = Path(str(value))
        if not path.is_absolute():
            path = source_root / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} manifest: {path}")
    return path


def build_multi_source_split(
    sources: Sequence[Mapping],
    *,
    root_dir: str,
    dataset_kwargs: Mapping | None = None,
    split_dir: str | None = None,
) -> tuple[Dataset, Dataset, torch.Tensor, list[dict]]:
    """Load immutable train/validation manifests for every data source."""
    if not sources:
        raise ValueError("At least one dataset source is required")
    shared_kwargs = dict(dataset_kwargs or {})
    parent = Path(root_dir)
    manifest_root = Path(split_dir) if split_dir else parent / "splits"
    train_datasets: list[Dataset] = []
    validation_datasets: list[Dataset] = []
    proportions: list[float] = []
    descriptions: list[dict] = []

    for index, source in enumerate(sources):
        name = str(_source_value(source, "name", f"source_{index}"))
        subdir = str(_source_value(source, "subdir", "") or "")
        source_root = parent / subdir if subdir else parent
        train_path = _resolve_manifest(source_root, source, "train", manifest_root)
        validation_path = _resolve_manifest(source_root, source, "validation", manifest_root)
        require_labels = bool(_source_value(source, "require_labels", True))
        train_dataset = PepoTrainDataset(
            root_dir=str(source_root),
            cluster_tsv_path=str(train_path),
            require_labels=require_labels,
            **shared_kwargs,
        )
        validation_dataset = PepoTrainDataset(
            root_dir=str(source_root),
            cluster_tsv_path=str(validation_path),
            require_labels=require_labels,
            **shared_kwargs,
        )
        if not train_dataset:
            raise ValueError(f"Source {name!r} has an empty train split; check preprocessing stats.json")
        train_datasets.append(train_dataset)
        validation_datasets.append(validation_dataset)
        proportions.append(float(_source_value(source, "weight", 1.0)))
        descriptions.append(
            {
                "name": name,
                "train_length": len(train_dataset),
                "validation_length": len(validation_dataset),
            }
        )

    weights, probabilities = source_sample_weights(
        [len(dataset) for dataset in train_datasets],
        proportions,
    )
    for description, probability in zip(descriptions, probabilities):
        description["probability"] = probability
    train = (
        train_datasets[0]
        if len(train_datasets) == 1
        else ConcatDataset(train_datasets)
    )
    validation = (
        validation_datasets[0]
        if len(validation_datasets) == 1
        else ConcatDataset(validation_datasets)
    )
    if not validation:
        raise ValueError("The combined validation split is empty; check preprocessing stats.json")
    return train, validation, weights, descriptions
