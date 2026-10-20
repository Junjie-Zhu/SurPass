"""Focused tests for deterministic, family-safe dataset manifests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import (  # noqa: E402
    build_dataset_splits,
    canonical_family_pair,
    generate_hard_negative_candidates,
    write_split_manifests,
)


class SplitManifestTest(unittest.TestCase):
    def setUp(self):
        self.mapping = [
            {"entity_id": "a", "family_id": "A", "length": 100},
            {"entity_id": "b", "family_id": "B", "length": 110},
            {"entity_id": "c", "family_id": "C", "length": 105},
            {"entity_id": "d", "family_id": "D", "length": 115},
            {"entity_id": "e", "family_id": "E", "length": 90},
            {"entity_id": "f", "family_id": "F", "length": 95},
        ]
        self.pairs = [
            {
                "pair_id": "ab",
                "chain1": "a",
                "chain2": "b",
                "source_is_positive": True,
            },
            {
                "pair_id": "cd",
                "chain1": "c",
                "chain2": "d",
                "source_is_positive": True,
            },
            {
                "pair_id": "ef",
                "chain1": "e",
                "chain2": "f",
                "source_is_positive": True,
            },
            {
                "pair_id": "ab-static",
                "chain1": "a",
                "chain2": "b",
                "source_is_positive": False,
                "negative_kind": "static",
                "confidence": 1.0,
            },
        ]

    def test_component_split_is_deterministic_and_excludes_benchmark(self):
        first = build_dataset_splits(
            self.mapping,
            [self.pairs],
            benchmark_families={"E"},
            validation_fraction=0.34,
            seed=19,
        )
        second = build_dataset_splits(
            self.mapping,
            [self.pairs],
            benchmark_families={"E"},
            validation_fraction=0.34,
            seed=19,
        )
        self.assertEqual(first, second)
        self.assertTrue(first["leakage"]["passed"])
        self.assertEqual(
            {row["pair_id"] for row in first["excluded"]},
            {"ef", "ab-static"},
        )

        family_split = {
            row["family_id"]: row["split"] for row in first["families"]
        }
        self.assertEqual(family_split["A"], family_split["B"])
        self.assertEqual(family_split["C"], family_split["D"])
        self.assertEqual(family_split["E"], "excluded")
        static = next(
            row
            for split in ("excluded",)
            for row in first[split]
            if row["pair_id"] == "ab-static"
        )
        self.assertEqual(static["negative_kind"], "static")
        self.assertEqual(static["sample_confidence"], 1.0)
        self.assertEqual(
            static["exclusion_reason"],
            "negative_conflicts_known_positive_family_pair",
        )

    def test_hard_negatives_exclude_known_positive_families(self):
        entity_to_family = {
            row["entity_id"]: row["family_id"] for row in self.mapping
        }
        lengths = {row["entity_id"]: row["length"] for row in self.mapping}
        annotated = [
            {
                **row,
                "family1": entity_to_family[row["chain1"]],
                "family2": entity_to_family[row["chain2"]],
                "length1": lengths[row["chain1"]],
                "length2": lengths[row["chain2"]],
                "split": "train",
            }
            for row in self.pairs
        ]
        negatives = generate_hard_negative_candidates(
            annotated,
            entity_to_family,
            count=3,
            entity_lengths=lengths,
            family_splits={family: "train" for family in entity_to_family.values()},
            seed=5,
            synthetic_confidence=0.4,
        )
        known_positive = {
            canonical_family_pair("A", "B"),
            canonical_family_pair("C", "D"),
            canonical_family_pair("E", "F"),
        }
        self.assertEqual(len(negatives), 3)
        for row in negatives:
            self.assertNotIn(
                canonical_family_pair(row["family1"], row["family2"]),
                known_positive,
            )
            self.assertEqual(row["negative_kind"], "synthetic")
            self.assertEqual(row["sample_confidence"], 0.4)
            self.assertFalse(row["source_is_positive"])

    def test_writes_all_required_manifests(self):
        manifests = build_dataset_splits(
            self.mapping,
            [self.pairs],
            benchmark_families={"E"},
            validation_fraction=0.34,
            seed=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_split_manifests(temp_dir, manifests)
            self.assertEqual(
                set(paths),
                {
                    "train",
                    "validation",
                    "excluded",
                    "families",
                    "components",
                    "leakage",
                    "stats",
                },
            )
            self.assertTrue(all(path.is_file() for path in paths.values()))
            self.assertTrue(
                (Path(temp_dir) / "sources" / "table_0.train.tsv").is_file()
            )
            self.assertTrue((Path(temp_dir) / "leakage_report.tsv").is_file())


if __name__ == "__main__":
    unittest.main()
