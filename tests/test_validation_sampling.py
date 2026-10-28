"""Bounded validation draws, exact class budgets, and unpadded distributed shards."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import ConcatDataset

from src.data.dataset import PepoTrainDataset
from src.data.splits import write_tsv
from src.data.validation_sampling import DistributedValidationSampler, ValidationPool, validation_pools
from src.train import evaluate_epoch


class ValidationSamplingTest(unittest.TestCase):
    def _pools(self):
        return [
            ValidationPool("pdb_ppi", True, 25, list(range(10))),
            ValidationPool("afdb_pos", True, 25, list(range(10, 110))),
            ValidationPool("pdb_neg", False, 25, list(range(110, 120))),
            ValidationPool("afdb_neg", False, 25, list(range(120, 220))),
        ]

    def test_exact_class_budget_source_balance_and_epoch_resampling(self):
        pools = self._pools()
        sampler = DistributedValidationSampler(pools, 20, negative_ratio=1, seed=7)
        sampler.set_epoch(1)
        first = list(sampler)
        self.assertEqual(len(first), 20)
        self.assertEqual(len(set(first)), 20)
        self.assertEqual(sampler.positive_samples, 10)
        self.assertEqual(sampler.negative_samples, 10)
        self.assertEqual([len(set(first) & set(pool.indices)) for pool in pools], [5, 5, 5, 5])
        self.assertEqual(list(sampler), first)
        sampler.set_epoch(2)
        self.assertNotEqual(list(sampler), first)
        sampler.set_epoch(1)
        self.assertEqual(list(sampler), first)

    def test_negative_ratio_and_source_weights(self):
        pools = self._pools()
        pools[0].weight, pools[1].weight = 1, 3
        sampler = DistributedValidationSampler(pools, 24, negative_ratio=2)
        indices = set(sampler)
        self.assertEqual(sampler.positive_samples, 8)
        self.assertEqual(sampler.negative_samples, 16)
        self.assertEqual(len(indices & set(pools[0].indices)), 2)
        self.assertEqual(len(indices & set(pools[1].indices)), 6)

    def test_insufficient_class_capacity_caps_budget_without_duplicates(self):
        pools = [ValidationPool("pos", True, 1, list(range(100))),
                 ValidationPool("neg1", False, 1, [100]),
                 ValidationPool("neg2", False, 1, [101, 102, 103])]
        sampler = DistributedValidationSampler(pools, 50, negative_ratio=1)
        self.assertEqual(sampler.total_size, 8)
        self.assertEqual(sampler.positive_samples, 4)
        self.assertEqual(sampler.negative_samples, 4)
        self.assertEqual(len(set(sampler)), 8)
        self.assertTrue({100, 101, 102, 103}.issubset(set(sampler)))

    def test_distributed_ranks_partition_one_global_draw_without_padding(self):
        pools = self._pools()
        expected = list(DistributedValidationSampler(pools, 23, seed=9))
        shards = [DistributedValidationSampler(pools, 23, seed=9, num_replicas=3, rank=i) for i in range(3)]
        self.assertEqual([len(shard) for shard in shards], [8, 8, 7])
        for rank, shard in enumerate(shards):
            self.assertEqual(list(shard), expected[rank::3])
        self.assertEqual(len(set(index for shard in shards for index in shard)), 23)
        empty = DistributedValidationSampler(pools, 2, num_replicas=4, rank=3)
        self.assertEqual(len(empty), 0)
        self.assertEqual(list(empty), [])

    def test_null_selects_full_split_and_zero_ratio_selects_only_positives(self):
        pools = self._pools()
        full = DistributedValidationSampler(pools, None)
        self.assertEqual(set(full), set(range(220)))
        positive = DistributedValidationSampler(pools, 20, negative_ratio=0)
        self.assertEqual(positive.negative_samples, 0)
        self.assertTrue(all(index < 110 for index in positive))

    def test_validation_pool_index_uses_metadata_without_loading_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = []
            for name, label in (("pos", True), ("neg", False)):
                path = write_tsv(root / f"{name}.tsv", [
                    {"chain1": "a", "chain2": "b", "ppi_cluster_id": "c1", "source_is_positive": label},
                    {"chain1": "a2", "chain2": "b2", "ppi_cluster_id": "c1", "source_is_positive": label},
                    {"chain1": "c", "chain2": "d", "ppi_cluster_id": "c2", "source_is_positive": label},
                ])
                datasets.append(PepoTrainDataset(str(root), path, require_labels=False))
            with patch.object(PepoTrainDataset, "__getitem__", side_effect=AssertionError("Features must not load")):
                pools = validation_pools(ConcatDataset(datasets), [{"name": "pos"}, {"name": "neg"}])
            self.assertEqual([(p.positive, p.indices) for p in pools], [(True, [0, 1]), (False, [2, 3])])
            datasets[0].metadata.loc[0, "source_is_positive"] = False
            with self.assertRaisesRegex(ValueError, "mixes positive and negative"):
                validation_pools(ConcatDataset(datasets), [{"name": "pos"}, {"name": "neg"}])

    def test_invalid_budgets_ratios_and_missing_classes_are_rejected(self):
        pools = self._pools()
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                DistributedValidationSampler(pools, size)
        for ratio in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                DistributedValidationSampler(pools, 20, negative_ratio=ratio)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            DistributedValidationSampler(pools[:2], 20, negative_ratio=1)

    def test_evaluation_counts_samples_and_weights_partial_batches(self):
        class Model(torch.nn.Module):
            def forward(self, batch):
                values = batch["value"]
                return SimpleNamespace(pair_logits=torch.stack((values, -values), dim=-1)[:, None, None, :],
                                       pair_mask=torch.ones((len(values), 1, 1), dtype=torch.bool))
        class Loss(torch.nn.Module):
            contact = SimpleNamespace(contact_bins=1)
            def forward(self, output, labels):
                loss = labels["loss"].float().mean()
                return loss, {"total": loss}
        batches = [
            ({"value": torch.tensor([2., -2.])}, {"crop_is_positive": torch.tensor([True, False]), "loss": torch.tensor([1., 1.])}),
            ({"value": torch.tensor([3.])}, {"crop_is_positive": torch.tensor([True]), "loss": torch.tensor([3.])}),
        ]
        metrics = evaluate_epoch(Model(), batches, Loss(), torch.device("cpu"), precision="float32", positive_weight=1)
        self.assertAlmostEqual(metrics["total"], 5 / 3)
        self.assertEqual((metrics["samples"], metrics["positive_samples"], metrics["negative_samples"]), (3, 2, 1))
        self.assertEqual(metrics["pair_auroc"], 1)


if __name__ == "__main__":
    unittest.main()
