"""Focused tests for SurPass cropping and rectangular collation."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collate import collate_fn  # noqa: E402
from src.data.cropping import (  # noqa: E402
    crop_negative_pair,
    crop_positive_pair,
)
from src.data.dataset import PairSample  # noqa: E402


def _features(length: int, offset: int = 0) -> dict:
    return {
        "mask": torch.ones(length, dtype=torch.bool),
        "residue_type": torch.arange(offset, offset + length),
        "residue_index": torch.arange(length),
        "chain_index": torch.zeros(length, dtype=torch.long),
        "plm_emb": torch.arange(length, dtype=torch.float32).unsqueeze(-1),
    }


def _pair_maps(
    p1_length: int,
    p2_length: int,
    contact: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    total = p1_length + p2_length
    distances = torch.full((total, total), 20.0)
    masks = torch.ones((total, total), dtype=torch.bool)
    p1_index, p2_index = contact
    distances[p1_index, p1_length + p2_index] = 4.0
    distances[p1_length + p2_index, p1_index] = 4.0
    return distances, masks


class DataSideContractTest(unittest.TestCase):
    def test_interface_crop_uses_one_total_budget(self):
        distances, masks = _pair_maps(10, 10, (8, 2))
        result = crop_positive_pair(
            _features(10),
            _features(10),
            distances,
            masks,
            crop_size=8,
            contact_threshold=8.0,
            rng=random.Random(7),
        )
        p1, p2, _, _, contains_contact, metadata = result
        self.assertEqual(len(p1["mask"]) + len(p2["mask"]), 8)
        self.assertTrue(contains_contact)
        self.assertEqual(metadata["mode"], "interface")
        self.assertEqual(metadata["selected_contact"], (8, 2))

    def test_positive_crops_always_retain_contact_at_chain_boundaries(self):
        for contact in ((0, 0), (9, 9), (0, 9), (9, 0)):
            distances, masks = _pair_maps(10, 10, contact)
            for seed in range(10):
                result = crop_positive_pair(
                    _features(10), _features(10), distances, masks,
                    crop_size=2, rng=random.Random(seed),
                )
                self.assertTrue(result[4])
                self.assertEqual(result[5]["mode"], "interface")

    def test_positive_without_valid_contact_is_rejected_even_without_crop(self):
        distances, masks = _pair_maps(10, 10, (9, 9))
        masks[:] = False
        for budget in (8, None):
            with self.assertRaisesRegex(ValueError, "no valid inter-chain contact"):
                crop_positive_pair(
                    _features(10), _features(10), distances, masks,
                    crop_size=budget,
                )

    def test_negative_crop_reallocates_short_chain_budget(self):
        p1, p2, metadata = crop_negative_pair(
            _features(20),
            _features(2),
            crop_size=10,
            rng=random.Random(3),
        )
        self.assertEqual((len(p1["mask"]), len(p2["mask"])), (8, 2))
        self.assertEqual(len(p1["mask"]) + len(p2["mask"]), 10)
        self.assertEqual(metadata["total_budget"], 10)

    def test_collate_pads_only_tails_and_uses_rectangular_labels(self):
        sample1 = PairSample(
            _features(2, 1),
            _features(1, 9),
            torch.tensor([[1], [2]]),
            torch.ones((2, 1), dtype=torch.bool),
            False,
            True,
            {"mode": "random"},
        )
        sample2 = PairSample(
            _features(1, 3),
            _features(3, 7),
            torch.tensor([[3, 4, 5]]),
            torch.ones((1, 3), dtype=torch.bool),
            True,
            True,
            {"mode": "interface"},
        )
        residues, labels = collate_fn([sample1, sample2])

        self.assertEqual(tuple(labels["label_inter_bins"].shape), (2, 2, 3))
        self.assertEqual(tuple(labels["label_inter_mask"].shape), (2, 2, 3))
        self.assertEqual(labels["crop_is_positive"].tolist(), [False, True])
        self.assertEqual(labels["source_is_positive"].tolist(), [True, True])
        self.assertEqual(
            residues["residue_type"][0].tolist(),
            [1, 2, 9, 0],
        )
        self.assertEqual(
            residues["residue_type"][1].tolist(),
            [3, 7, 8, 9],
        )
        self.assertNotIn("sample_confidence", labels)
        self.assertEqual(residues["p1_length"].tolist(), [2, 1])
        self.assertEqual(residues["p2_length"].tolist(), [1, 3])


if __name__ == "__main__":
    unittest.main()
