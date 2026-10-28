"""Bound CPU embedding retention without changing inference inputs."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.inference import InferencePairDataset, load_plm_embedding


class InferenceMemoryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.identifiers = [f"p{i}" for i in range(40)]
        (self.root / "test.fasta").write_text("".join(
            f">{identifier}\nACDE\n" for identifier in self.identifiers
        ))
        (self.root / "test.tsv").write_text(
            "Protein pairs\tCategory\n" + "".join(
                f"p0_{identifier}\t1\n" for identifier in self.identifiers[1:]
            )
        )
        for index, identifier in enumerate(self.identifiers):
            torch.save({"plm_emb": torch.full((4, 8), float(index))},
                       self.root / f"{identifier}.pt")
        # Four rows: eight floats plus residue/index/chain longs and one bool.
        self.entry_bytes = 4 * (8 * 4 + 3 * 8 + 1)

    def dataset(self, budget_bytes):
        return InferencePairDataset(
            self.root / "test.tsv", self.root / "test.fasta", self.root,
            embedding_cache_mb=budget_bytes / 1024**2,
        )

    def test_cache_remains_bounded_across_many_distinct_proteins(self):
        dataset = self.dataset(2 * self.entry_bytes)
        for identifier in self.identifiers:
            dataset._features(identifier, "ACDE")
            self.assertLessEqual(dataset._feature_cache_bytes, 2 * self.entry_bytes)
        self.assertEqual(len(dataset._feature_cache), 2)
        self.assertEqual(dataset._feature_cache_bytes, 2 * self.entry_bytes)

    def test_hits_refresh_recency_and_evicted_entries_reload(self):
        dataset = self.dataset(2 * self.entry_bytes)
        with patch("src.inference.load_plm_embedding", wraps=load_plm_embedding) as load:
            first = dataset._features("p0", "ACDE")
            dataset._features("p1", "ACDE")
            self.assertIs(first, dataset._features("p0", "ACDE"))
            self.assertEqual(load.call_count, 2)
            dataset._features("p2", "ACDE")
            self.assertNotIn("p1", dataset._feature_cache)
            self.assertIn("p0", dataset._feature_cache)
            reloaded = dataset._features("p1", "ACDE")
            self.assertEqual(load.call_count, 4)
            torch.testing.assert_close(reloaded["plm_emb"], torch.ones((4, 8)))

    def test_disabled_and_undersized_cache_do_not_retain_embeddings(self):
        for budget in (0, self.entry_bytes - 1):
            with self.subTest(budget=budget):
                dataset = self.dataset(budget)
                with patch("src.inference.load_plm_embedding", wraps=load_plm_embedding) as load:
                    for _ in range(2):
                        dataset[0]
                    self.assertEqual(load.call_count, 4)
                self.assertEqual(dataset._feature_cache_bytes, 0)
                self.assertFalse(dataset._feature_cache)

    def test_cached_and_uncached_inputs_are_identical_after_evictions(self):
        cached, uncached = self.dataset(2 * self.entry_bytes), self.dataset(0)
        for index in [0, 1, 2, 0, len(cached) - 1]:
            actual, expected = cached[index], uncached[index]
            self.assertEqual(actual[2], expected[2])
            for first, second in zip(actual[:2], expected[:2]):
                for key in first:
                    torch.testing.assert_close(first[key], second[key])

    def test_loaded_view_releases_oversized_backing_storage(self):
        backing = torch.arange(1024 * 8, dtype=torch.float32).reshape(1024, 8)
        expected = backing[10:14]
        path = self.root / "view.pt"
        torch.save({"plm_emb": expected}, path)
        loaded = load_plm_embedding(path)
        torch.testing.assert_close(loaded, expected)
        self.assertEqual(loaded.untyped_storage().nbytes(),
                         loaded.numel() * loaded.element_size())

    def test_invalid_cache_budget_is_rejected(self):
        for budget in (-1, float("inf"), float("nan")):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "embedding_cache_mb"):
                InferencePairDataset(
                    self.root / "test.tsv", self.root / "test.fasta", self.root,
                    embedding_cache_mb=budget,
                )


if __name__ == "__main__":
    unittest.main()
