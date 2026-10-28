"""A persisted validation panel with lazy, deterministic member/crop replay."""
import bisect
import hashlib
import json
import random
from pathlib import Path

from torch.utils.data import Dataset, ConcatDataset, Sampler
from src.data.validation_sampling import DistributedValidationSampler, validation_pools


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class UnpaddedShardSampler(Sampler):
    def __init__(self, dataset, rank=0, world_size=1):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class FrozenValidationDataset(Dataset):
    def __init__(self, dataset, manifest):
        self.datasets = list(dataset.datasets) if isinstance(dataset, ConcatDataset) else [dataset]
        expected = self.split_identity(self.datasets)
        if manifest["split_identity"] != expected:
            raise ValueError("Fixed validation manifest does not match current split/crop settings")
        if manifest.get("identity") != digest({k: v for k, v in manifest.items() if k != "identity"}):
            raise ValueError("Corrupt fixed validation manifest")
        self.manifest = manifest
        self.identity = manifest["identity"]
        self.entries = manifest["entries"]
        self.total_size = len(self.entries)
        self.positive_samples = sum(entry["positive"] for entry in self.entries)
        self.negative_samples = self.total_size - self.positive_samples
        self.requested_samples = manifest["requested_samples"]

    @staticmethod
    def split_identity(datasets):
        return digest([{
            "manifest_sha256": hashlib.sha256(Path(data.cluster_tsv_path).read_bytes()).hexdigest(),
            "clusters": [str(item) for item in data.items], "crop_size": data.crop_size,
            "contact_threshold": data.contact_threshold,
        } for data in datasets])

    @classmethod
    def build(cls, dataset, sources, num_samples, negative_ratio, seed):
        datasets = list(dataset.datasets) if isinstance(dataset, ConcatDataset) else [dataset]
        sampler = DistributedValidationSampler(validation_pools(dataset, sources), num_samples,
                                                negative_ratio=negative_ratio, seed=seed)
        cumulative = []
        for data in datasets:
            cumulative.append(len(data) + (cumulative[-1] if cumulative else 0))
        entries = []
        for panel_index, global_index in enumerate(sampler):
            source = bisect.bisect_right(cumulative, global_index)
            local_index = global_index - (cumulative[source-1] if source else 0)
            data = datasets[source]
            rng = random.Random(seed + panel_index)
            rows = data.cluster_to_rows[data.items[local_index]]
            row_index = rng.randrange(len(rows))
            crop_seed = rng.randrange(2**63)
            sample = data.sample(local_index, row_index=row_index, rng=random.Random(crop_seed))
            entries.append({"source": source, "source_name": str(sources[source]["name"]),
                            "index": local_index, "cluster": str(data.items[local_index]),
                            "row_index": row_index, "row": json.loads(rows.iloc[row_index].to_json()),
                            "crop_seed": crop_seed, "crop": sample.metadata,
                            "positive": bool(sample.crop_is_positive)})
        manifest = {"version": 1, "seed": seed, "requested_samples": num_samples,
                    "negative_ratio": negative_ratio, "split_identity": cls.split_identity(datasets),
                    "entries": entries}
        # Normalize tuples (crop contact coordinates) to their persisted representation.
        manifest = json.loads(canonical(manifest))
        manifest["identity"] = digest(manifest)
        return cls(dataset, manifest)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        sample = self.datasets[entry["source"]].sample(
            entry["index"], row_index=entry["row_index"], rng=random.Random(entry["crop_seed"]))
        if canonical(sample.metadata) != canonical(entry["crop"]) or bool(sample.crop_is_positive) != entry["positive"]:
            raise ValueError("Validation crop/label changed from the saved panel")
        return sample
