"""Small, class-balanced validation draws from immutable split manifests."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from torch.utils.data import ConcatDataset, Dataset, Sampler


@dataclass
class ValidationPool:
    source: str
    positive: bool
    weight: float
    indices: list[int]


def validation_pools(dataset: Dataset, sources: Sequence[Mapping]) -> list[ValidationPool]:
    """Inspect cluster labels once, without loading features or calling __getitem__."""
    datasets = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    if len(datasets) != len(sources):
        raise ValueError("Validation datasets and source configurations must correspond")
    pools = []
    offset = 0
    for data, source in zip(datasets, sources):
        weight = float(source.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Validation source weights must be finite and positive")
        labels = {}
        if "source_is_positive" in data.metadata:
            for cluster, value in data.metadata[["ppi_cluster_id", "source_is_positive"]].itertuples(index=False, name=None):
                positive = data._as_bool(value)
                if cluster in labels and labels[cluster] != positive:
                    raise ValueError(f"Validation cluster {cluster!r} mixes positive and negative rows")
                labels[cluster] = positive
        by_class = {True: [], False: []}
        for index, cluster in enumerate(data.items):
            by_class[labels.get(cluster, True)].append(offset + index)
        for positive, indices in by_class.items():
            if indices:
                pools.append(ValidationPool(str(source["name"]), positive, weight, indices))
        offset += len(data)
    return pools


def _pool_quotas(pools: list[ValidationPool], count: int) -> list[int]:
    """Allocate a class budget by source weight, redistributing exhausted pools."""
    counts = [0] * len(pools)
    remaining = count
    while remaining:
        active = [i for i, pool in enumerate(pools) if counts[i] < len(pool.indices)]
        total_weight = sum(pools[i].weight for i in active)
        shares = {i: remaining * pools[i].weight / total_weight for i in active}
        added = 0
        for i in active:
            take = min(int(shares[i]), len(pools[i].indices) - counts[i])
            counts[i] += take
            added += take
        remaining -= added
        if not added:
            for i in sorted(active, key=lambda i: (-shares[i], pools[i].source))[:remaining]:
                counts[i] += 1
                remaining -= 1
    return counts


class DistributedValidationSampler(Sampler[int]):
    """Resample unique clusters per epoch, then shard without duplicate padding.

    num_samples is the global budget across ranks. None selects the full split.
    negative_ratio means negatives per positive, independently of source sizes.
    """

    def __init__(
        self, pools: Sequence[ValidationPool], num_samples: int | None,
        *, negative_ratio: float = 1.0, num_replicas: int = 1, rank: int = 0, seed: int = 42,
    ):
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid validation rank or replica count")
        if num_samples is not None and (isinstance(num_samples, bool) or int(num_samples) != num_samples or num_samples <= 0):
            raise ValueError("validation_samples must be a positive integer or null for the full split")
        ratio = float(negative_ratio)
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError("validation_negative_ratio must be finite and non-negative")
        self.pools = list(pools)
        self.num_replicas, self.rank, self.seed = int(num_replicas), int(rank), int(seed)
        self.epoch = 0
        self.requested_samples = num_samples
        self.full_split = num_samples is None
        self._indices: list[int] | None = None
        positive_capacity = sum(len(pool.indices) for pool in pools if pool.positive)
        negative_capacity = sum(len(pool.indices) for pool in pools if not pool.positive)
        if self.full_split:
            self.positive_samples, self.negative_samples = positive_capacity, negative_capacity
        else:
            budget = min(int(num_samples), positive_capacity + negative_capacity)
            if ratio == 0:
                budget = min(budget, positive_capacity)
            else:
                budget = min(budget, math.floor(positive_capacity * (1 + ratio)),
                             math.floor(negative_capacity * (1 + ratio) / ratio))
            while budget > 0:
                negative = round(budget * ratio / (1 + ratio))
                positive = budget - negative
                if positive <= positive_capacity and negative <= negative_capacity:
                    break
                budget -= 1
            if budget <= 0:
                raise ValueError("Validation has insufficient eligible clusters for the requested negative ratio")
            self.positive_samples, self.negative_samples = positive, negative
        self.total_size = self.positive_samples + self.negative_samples
        if not self.total_size:
            raise ValueError("The validation split is empty")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._indices = None

    def __len__(self) -> int:
        return max(0, (self.total_size - self.rank + self.num_replicas - 1) // self.num_replicas)

    def __iter__(self):
        if self._indices is None:
            rng = random.Random(self.seed + self.epoch)
            indices = []
            for positive, count in ((True, self.positive_samples), (False, self.negative_samples)):
                pools = [pool for pool in self.pools if pool.positive == positive]
                if self.full_split:
                    indices.extend(index for pool in pools for index in pool.indices)
                else:
                    for pool, quota in zip(pools, _pool_quotas(pools, count)):
                        indices.extend(rng.sample(pool.indices, quota))
            if not self.full_split:
                rng.shuffle(indices)
            self._indices = indices[self.rank :: self.num_replicas]
        return iter(self._indices)
