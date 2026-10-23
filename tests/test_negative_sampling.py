"""Exact matching, deterministic process workers, and CLI progress reporting."""

import hashlib
import io
import json
import multiprocessing
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_dataset_splits import main
from src.data.negative_sampling import CandidateIndex, NegativeQuery
from src.data.splits import build_dataset_splits, generate_hard_negative_candidates, write_tsv


class NegativeSamplingTest(unittest.TestCase):
    def test_indexed_best_matches_exhaustive_search_with_ties_and_exclusions(self):
        rng = random.Random(31)
        families = {f'e{i}': f'F{i // 2}' for i in range(80)}
        lengths = {entity: rng.choice((50, 60, 100, 101, 300)) for entity in families}
        degrees = {family: rng.choice((0, 1, 2, 5, 9)) for family in families.values()}
        train, validation = list(families)[:60], list(families)[60:]
        pools = {}
        for split, entities in (("train", train), ("validation", validation)):
            grouped = {}
            for entity in entities:
                degree = degrees[families[entity]]
                grouped.setdefault(degree, {}).setdefault(lengths[entity], []).append(entity)
            pools[("pdb_ppi", split)] = tuple(
                (degree, tuple(sorted(buckets)), tuple(buckets[length] for length in sorted(buckets)))
                for degree, buckets in sorted(grouped.items())
            )
        index = CandidateIndex(
            families, lengths, degrees, {"F0": {"F1", "F3"}}, {"e0": {"e8", "e9"}}, pools, 23,
        )
        for target in families:
            query = NegativeQuery("e0", target, "pdb_ppi", "train", "template", False, frozenset(("e10", "e11")))
            ranked = []
            for entity in train:
                if families[entity] in {"F0", "F1", "F3"} or entity in {"e8", "e9", "e10", "e11"}:
                    continue
                score = abs(lengths[entity] - lengths[target]) / lengths[target]
                score += abs(degrees[families[entity]] - degrees[families[target]]) / max(degrees[families[target]], 1)
                tie = hashlib.sha256(f"23|template|False|{entity}".encode()).hexdigest()
                ranked.append((score, tie, entity))
            best = min(ranked)
            self.assertEqual(index.best(query), (best[0], best[2]))

    def test_search_skips_worse_buckets_instead_of_hashing_all_candidates(self):
        families = {f'e{i}': f'F{i}' for i in range(1000)}
        lengths = {entity: 100 + i for i, entity in enumerate(families)}
        degrees = {family: 1 for family in families.values()}
        pool = ((1, tuple(lengths.values()), tuple([entity] for entity in families)),)
        index = CandidateIndex(families, lengths, degrees, {}, {}, {("pdb_ppi", "train"): pool}, 42)
        with patch("src.data.negative_sampling.stable_tie_breaker", return_value="tie") as tie:
            match = index.best(NegativeQuery("e0", "e500", "pdb_ppi", "train", "p", False, frozenset()))
        self.assertEqual(match, (0.0, "e500"))
        self.assertEqual(tie.call_count, 1)

    def _fixture(self):
        mapping = [{"entity_id": f'p{i}', "family_id": f'F{i}', "length": 100} for i in range(12)]
        rows = [{
            "pair_id": f"pair{i}", "chain1": f"p{i}", "chain2": f"p{i+1}",
            "family1": f"F{i}", "family2": f"F{i+1}", "source_table": "pdb_ppi",
            "source_is_positive": True, "split": "train",
        } for i in range(0, 12, 2)]
        return mapping, rows

    def test_process_workers_match_serial_through_conflicts_reversals_and_exhaustion(self):
        mapping, rows = self._fixture()
        families = {row["entity_id"]: row["family_id"] for row in mapping}
        lengths = {row["entity_id"]: row["length"] for row in mapping}
        kwargs = dict(count=1000, entity_lengths=lengths, seed=11)
        serial = generate_hard_negative_candidates(rows, families, **kwargs)
        parallel = generate_hard_negative_candidates(rows, families, workers=2, **kwargs)
        self.assertEqual(serial, parallel)
        # All 12 choose 2 unordered pairs except six known positives.
        self.assertEqual(len(serial), 60)
        self.assertEqual(len({tuple(sorted((r["chain1"], r["chain2"]))) for r in parallel}), 60)

    def test_parallel_build_preserves_every_manifest_and_leakage_check(self):
        mapping, rows = self._fixture()
        kwargs = dict(source_names=["pdb_ppi"], validation_fraction=0.5, seed=7)
        serial = build_dataset_splits(mapping, [rows], **kwargs)
        parallel = build_dataset_splits(mapping, [rows], workers=2, **kwargs)
        self.assertEqual(serial, parallel)
        self.assertTrue(parallel["leakage"]["passed"])

    def test_worker_index_is_picklable_with_spawn_start_method(self):
        mapping, rows = self._fixture()
        families = {row["entity_id"]: row["family_id"] for row in mapping}
        lengths = {row["entity_id"]: row["length"] for row in mapping}
        kwargs = dict(count=16, entity_lengths=lengths, seed=19)
        serial = generate_hard_negative_candidates(rows, families, **kwargs)
        with patch("src.data.splits.multiprocessing.Pool", multiprocessing.get_context("spawn").Pool):
            parallel = generate_hard_negative_candidates(rows, families, workers=2, **kwargs)
        self.assertEqual(serial, parallel)

    def test_cli_progress_stays_on_stderr_and_can_be_disabled(self):
        mapping, rows = self._fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tsv(root / "mapping.tsv", mapping)
            write_tsv(root / "verified_pdb_pairs.tsv", rows)
            args = [
                "--family-mapping", str(root / "mapping.tsv"),
                "--pair-table", str(root / "verified_pdb_pairs.tsv"),
                "--validation-fraction", "0.5", "--output-dir", str(root / "splits"),
            ]
            for quiet in (False, True):
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    main(args + (["--no-progress"] if quiet else []))
                self.assertTrue(json.loads(out.getvalue())["leakage"]["passed"])
                if quiet:
                    self.assertEqual(err.getvalue(), "")
                else:
                    self.assertIn("Negatives pdb_neg/train", err.getvalue())
                    self.assertIn("Write train.tsv", err.getvalue())
                    self.assertIn("pairs/s", err.getvalue())

    def test_invalid_worker_count_is_rejected_even_when_no_negatives_requested(self):
        with self.assertRaisesRegex(ValueError, "workers"):
            generate_hard_negative_candidates([], {}, count=0, workers=0)


if __name__ == "__main__":
    unittest.main()
