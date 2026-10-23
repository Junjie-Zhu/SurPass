"""Focused tests for deterministic, family-safe dataset manifests."""

from __future__ import annotations

import sys
import tempfile
import unittest
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import (  # noqa: E402
    build_dataset_splits,
    canonical_family_pair,
    generate_hard_negative_candidates,
    write_split_manifests,
    read_records,
)
from src.data.balancing import build_multi_source_split
from scripts.build_dataset_splits import main as build_splits_cli
import torch


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
        self.assertNotIn("sample_confidence", static)
        self.assertNotIn("confidence", static)
        self.assertEqual(
            static["exclusion_reason"],
            "negative_conflicts_known_positive_entity_pair",
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
            self.assertNotIn("sample_confidence", row)
            self.assertFalse(row["source_is_positive"])

    def test_benchmark_exclusion_does_not_spread_through_partners_or_negatives(self):
        pairs = [
            {"chain1": "a", "chain2": "b", "source_is_positive": True},
            {"chain1": "c", "chain2": "d", "source_is_positive": True},
            {"chain1": "b", "chain2": "c", "source_is_positive": False},
            {"chain1": "d", "chain2": "e", "source_is_positive": False},
        ]
        manifests = build_dataset_splits(
            self.mapping, [pairs], benchmark_entities={"a"}, validation_fraction=0,
        )
        self.assertEqual(len(manifests["excluded"]), 1)
        self.assertEqual(manifests["excluded"][0]["exclusion_reason"], "sequence_matches_benchmark")
        self.assertEqual(sum(not row["source_is_positive"] for row in manifests["train"]), 2)
        self.assertTrue(manifests["leakage"]["passed"])
        components = {row["entity_id"]: row["component_id"] for row in manifests["families"]}
        self.assertNotEqual(components["b"], components["c"])
        self.assertNotEqual(components["d"], components["e"])
        negatives = generate_hard_negative_candidates(
            manifests["train"] + manifests["excluded"],
            {row["entity_id"]: row["family_id"] for row in self.mapping}, count=10,
            entity_lengths={row["entity_id"]: row["length"] for row in self.mapping},
            family_splits={row["family_id"]: "train" for row in self.mapping},
        )
        self.assertTrue(all("a" not in (row["chain1"], row["chain2"]) for row in negatives))

    def test_raw_source_clusters_and_labels_are_preserved(self):
        manifests = build_dataset_splits(self.mapping, [[
            {"chain1": "a", "chain2": "b", "CLUSTER": "17"},
            {"chain1": "c", "chain2": "d", "CLUSTER": "17"},
            {"chain1": "e", "chain2": "f", "CLUSTER": "18"},
        ]], source_names=["afdb_neg"], validation_fraction=0)
        self.assertTrue(all(not row["source_is_positive"] for row in manifests["train"]))
        self.assertEqual(manifests["stats"]["source_cluster_counts"]["train"]["afdb_neg"], 2)
        self.assertEqual(manifests["stats"]["source_counts"]["train"]["afdb_neg"], 3)

    def test_raw_pdb_pairs_group_by_unordered_30_percent_families(self):
        mapping = self.mapping + [{"entity_id": "a2", "family_id": "A", "length": 100}]
        manifests = build_dataset_splits(mapping, [[
            {"chain1": "a", "chain2": "b"}, {"chain1": "b", "chain2": "a2"},
        ]], source_names=["pdb_ppi"], validation_fraction=0, pdb_negative_ratio=0)
        self.assertEqual(manifests["stats"]["source_cluster_counts"]["train"]["pdb_ppi"], 1)
        self.assertEqual(manifests["stats"]["pdb_negative_counts"]["train"]["positive_clusters"], 1)

    def test_negative_edges_and_clusters_do_not_merge_positive_components(self):
        mapping, pairs = self._pdb_fixture()
        negatives = [{"chain1": f"p{i}", "chain2": f"p{i+2}", "CLUSTER": "shared"}
                     for i in range(0, 16, 2)]
        manifests = build_dataset_splits(
            mapping, [pairs, negatives], source_names=["pdb_ppi", "afdb_neg"],
            validation_fraction=0.5, pdb_negative_ratio=0,
        )
        self.assertEqual(manifests["stats"]["component_count"], 9)
        self.assertGreater(sum(row["source_is_positive"] for row in manifests["validation"]), 0)
        self.assertTrue(manifests["leakage"]["passed"])
        kept_splits = {row["split"] for split in ("train", "validation")
                       for row in manifests[split] if row["source_table"] == "afdb_neg"}
        self.assertLessEqual(len(kept_splits), 1)

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

    def _pdb_fixture(self):
        mapping = [
            {"entity_id": f"p{i}", "family_id": f"F{i}", "length": 4}
            for i in range(18)
        ]
        pairs = [
            {
                "pair_id": f"pdb{i}", "chain1": f"p{i}", "chain2": f"p{i+1}",
                "source_is_positive": True, "label_path": f"pdb{i}.pkl",
            }
            for i in range(0, 18, 2)
        ]
        return mapping, pairs

    def test_pdb_negatives_are_generated_per_split_and_loaded_separately(self):
        mapping, pairs = self._pdb_fixture()
        manifests = build_dataset_splits(
            mapping, [pairs], source_names=["pdb_ppi"],
            benchmark_families={"F16"}, validation_fraction=0.5,
        )
        self.assertTrue(manifests["leakage"]["passed"])
        known_positive = {canonical_family_pair(f"F{i}", f"F{i+1}") for i in range(0, 18, 2)}
        entity_split = {row["entity_id"]: row["split"] for row in manifests["families"]}
        for split in ("train", "validation"):
            negatives = [row for row in manifests[split] if row["source_table"] == "pdb_neg"]
            positives = [row for row in manifests[split] if row["source_is_positive"]]
            self.assertEqual(len(negatives), len(positives))
            self.assertGreater(len(negatives), 0)
            for row in negatives:
                self.assertEqual(row["negative_kind"], "synthetic")
                self.assertNotIn("sample_confidence", row)
                self.assertNotIn("label_path", row)
                self.assertEqual(entity_split[row["chain1"]], split)
                self.assertEqual(entity_split[row["chain2"]], split)
                self.assertNotIn(canonical_family_pair(row["family1"], row["family2"]), known_positive)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_split_manifests(root / "splits", manifests)
            for split in ("train", "validation", "excluded"):
                positive_rows = read_records(root / "splits/sources" / f"pdb_ppi.{split}.tsv")
                negative_rows = read_records(root / "splits/sources" / f"pdb_neg.{split}.tsv")
                self.assertTrue(all(row["source_is_positive"] == "true" for row in positive_rows))
                self.assertTrue(all(row["source_is_positive"] == "false" for row in negative_rows))
            source_root = root / "PDB_PPI"
            (source_root / "embedded").mkdir(parents=True)
            (source_root / "labels").mkdir()
            for entity in mapping:
                torch.save({
                    "cb_mask": torch.ones(4).bool(), "residue_type": torch.zeros(4).long(),
                    "residue_index": torch.arange(4), "chain_index": torch.zeros(4).long(),
                    "plm_emb": torch.randn(4, 8),
                }, source_root / "embedded" / f"{entity['entity_id']}.pt")
            for pair in pairs:
                distances = torch.full((8, 8), 20.0)
                distances[0, 4] = distances[4, 0] = 4
                with (source_root / "labels" / pair["label_path"]).open("wb") as handle:
                    pickle.dump({"pairwise_dist": distances, "pairwise_mask": torch.ones(8, 8).bool()}, handle)
            train, validation, weights, descriptions = build_multi_source_split([
                {"name": "pdb_ppi", "subdir": "PDB_PPI", "require_labels": True},
                {"name": "pdb_neg", "subdir": "PDB_PPI", "require_labels": False},
            ], root_dir=str(root), dataset_kwargs={"crop_size": 4})
            self.assertEqual(len(train), 8)
            self.assertEqual(len(validation), 8)
            self.assertEqual(len(weights), len(train))
            positive, negative = train[0], train[len(train) - 1]
            self.assertTrue(positive.crop_is_positive)
            self.assertFalse(negative.source_is_positive)
            self.assertFalse(negative.crop_is_positive)
            self.assertEqual(negative.metadata["source_kind"], "synthetic")

    def test_cli_defaults_produce_separate_pdb_negatives(self):
        from src.data.splits import write_tsv
        from contextlib import redirect_stderr, redirect_stdout
        import io

        mapping, pairs = self._pdb_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tsv(root / "families.tsv", mapping)
            write_tsv(root / "PDB_PPI/verified_pdb_pairs.tsv", pairs)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                result = build_splits_cli([
                    "--family-mapping", str(root / "families.tsv"),
                    "--pair-table", str(root / "PDB_PPI/verified_pdb_pairs.tsv"),
                    "--validation-fraction", "0.5", "--output-dir", str(root / "splits"),
                ])
            self.assertEqual(result, 0)
            stats = json.loads(output.getvalue())["stats"]
            self.assertEqual(stats["generated_pdb_negatives"], 9)
            self.assertTrue((root / "splits/sources/pdb_neg.train.tsv").is_file())

    def test_optional_synthetic_sources_are_not_merged_into_positives(self):
        manifests = build_dataset_splits(
            self.mapping, [self.pairs], validation_fraction=0,
            hard_negative_count=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_split_manifests(root, manifests)
            rows = read_records(root / "sources/table_0.synthetic.train.tsv")
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["source_is_positive"] == "false" for row in rows))
            self.assertTrue((root / "sources/table_0.synthetic.validation.tsv").is_file())

    def test_pdb_negative_shortage_is_reported_without_crossing_splits(self):
        mapping, pairs = self._pdb_fixture()
        with self.assertWarnsRegex(RuntimeWarning, "generated 0/1"):
            manifests = build_dataset_splits(
                mapping[:2], [pairs[:1]], source_names=["pdb_ppi"], validation_fraction=0,
            )
        self.assertEqual(manifests["stats"]["pdb_negative_counts"]["train"], {"positive_pairs": 1, "positive_clusters": 1, "requested": 1, "generated": 0, "negative_clusters": 0})
        self.assertEqual(manifests["stats"]["generated_pdb_negatives"], 0)

    def test_empty_validation_source_has_loadable_headers(self):
        from src.data.dataset import PepoTrainDataset

        manifests = build_dataset_splits(
            self.mapping, [self.pairs], validation_fraction=0.34,
        )
        manifests["train"].append({**manifests["train"][0], "source_table": "extra"})
        with tempfile.TemporaryDirectory() as directory:
            write_split_manifests(Path(directory) / "splits", manifests)
            # Source table_0 contains no structural labels in this fixture.
            train, validation, _, descriptions = build_multi_source_split([
                {"name": "table_0", "require_labels": False},
                {"name": "extra", "require_labels": False},
            ], root_dir=directory)
            self.assertGreater(len(train), 0)
            self.assertGreater(len(validation), 0)
            self.assertEqual(descriptions[1]["validation_length"], 0)
            self.assertEqual(len(PepoTrainDataset(
                root_dir=directory,
                cluster_tsv_path=Path(directory) / "splits/sources/extra.validation.tsv",
                require_labels=True,
            )), 0)


if __name__ == "__main__":
    unittest.main()
