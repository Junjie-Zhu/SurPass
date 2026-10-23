"""Direct sequence filtering, real identity settings, and safe MMseqs reuse."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_mmseqs_families import main, read_benchmark_hits, read_clusters, read_fasta
from src.data.splits import build_dataset_splits, read_records


class MMseqsFamiliesTest(unittest.TestCase):
    def test_fasta_multiline_case_and_conflicting_duplicate_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.fasta"
            path.write_text(">a description\nac\nde\n>a\nACDE\n>b\nGG\n")
            self.assertEqual(read_fasta(path), {"a": "ACDE", "b": "GG"})
            for content in (">a\nAC\n>a\nGG\n", ">\nAC\n", ">a\n", "AC\n"):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_fasta(path)

    def test_exact_identity_boundary_and_optional_bidirectional_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hits.tsv"
            path.write_text("a\ttest\t0.3\t0.8\t0.8\nb\ttest\t0.299\t1\t1\nc\ttest\t0.8\t0.79\t1\n")
            self.assertEqual(read_benchmark_hits(path, {"a", "b", "c"}, {"test"}, 0), {"a", "c"})
            self.assertEqual(read_benchmark_hits(path, {"a", "b", "c"}, {"test"}, 0.8), {"a"})
            path.write_text("a\ttest\t30\t1\t1\n")
            with self.assertRaisesRegex(ValueError, "fractional"):
                read_benchmark_hits(path, {"a"}, {"test"}, 0)
            path.write_text("wrong-id\ttest\t0.3\t1\t1\n")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                read_benchmark_hits(path, {"a"}, {"test"}, 0)

    def test_incomplete_clusters_fail_instead_of_silent_singleton_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clusters.tsv"
            path.write_text("a\ta\n")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_clusters(path, {"a": "AC", "b": "DE"})

    def test_coverage_fallback_preserves_identity_and_surviving_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.fa").write_text(">a\nACDE\n>b\nGGGG\n")
            (root / "test.fa").write_text(">test\nACDEGGGG\n")
            (root / "neg.tsv").write_text("chain1\tchain2\na\tb\n")
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                if command[1] == "easy-search":
                    # A short local match removes the only negative at c=0,
                    # but does not meet bidirectional 80% coverage.
                    Path(command[4]).write_text("a\ttest\t0.5\t1\t0.5\n" if command[command.index("-c") + 1] == "0.0" else "")
                else:
                    Path(command[3] + "_cluster.tsv").write_text("a\ta\nb\tb\n")
            argv = ["--fasta", str(root / "train.fa"), "--benchmark-fasta", str(root / "test.fa"),
                    "--negative-pair-table", str(root / "neg.tsv"), "--output-dir", str(root / "out")]
            with patch("scripts.build_mmseqs_families.shutil.which", return_value="/fake/mmseqs"), patch(
                "scripts.build_mmseqs_families.subprocess.check_output", return_value="test-version"
            ), patch("scripts.build_mmseqs_families.subprocess.run", side_effect=run), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 0)
                self.assertEqual([cmd[cmd.index("-c") + 1] for cmd in commands], ["0.0", "0.8", "0.8"])
                self.assertTrue(all(cmd[cmd.index("--min-seq-id") + 1] == "0.3" for cmd in commands))
                stats = json.loads((root / "out/filter_stats.json").read_text())
                self.assertEqual(stats["effective_coverage"], 0.8)
                self.assertEqual(stats["negative_pair_counts"][str(root / "neg.tsv")]["kept_after_sequence_filter"], 1)
                self.assertEqual(main(argv), 0)
                self.assertEqual(len(commands), 3)

    def test_direct_filtering_pipeline_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "train.fasta"
            benchmark = root / "benchmark.fasta"
            fasta.write_text(">a\nACDE\n>b\nACDF\n>c\nGGGG\n>d\nGGGA\n")
            # Same ID, different sequence: benchmark namespace is independent.
            benchmark.write_text(">a\nTTTT\n")
            negatives = root / "negatives.tsv"
            negatives.write_text("chain1\tchain2\na\tc\nb\td\n")
            output = root / "output"
            argv = ["--fasta", str(fasta), "--benchmark-fasta", str(benchmark),
                    "--negative-pair-table", str(negatives), "--output-dir", str(output)]
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                if command[1] == "easy-search":
                    Path(command[4]).write_text("a\ta\t0.3\t0.8\t0.8\n")
                elif command[1] == "easy-cluster":
                    # b shares a family with the hit, but has NO direct hit.
                    Path(command[3] + "_cluster.tsv").write_text("a\ta\na\tb\nc\tc\nd\td\n")
                else:
                    raise AssertionError(command)

            with patch("scripts.build_mmseqs_families.shutil.which", return_value="/fake/mmseqs"), patch(
                "scripts.build_mmseqs_families.subprocess.check_output", return_value="test-version\n"
            ), patch("scripts.build_mmseqs_families.subprocess.run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 0)
                for command in calls:
                    for flag, value in (("--min-seq-id", "0.3"), ("--alignment-mode", "3"), ("--seq-id-mode", "0"), ("-c", "0.0")):
                        self.assertEqual(command[command.index(flag) + 1], value)
                self.assertEqual(main(argv), 0)
                self.assertEqual(len(calls), 2)
                stats = json.loads((output / "filter_stats.json").read_text())
                self.assertEqual(stats["effective_coverage"], 0.0)
                self.assertEqual(stats["negative_pair_counts"][str(negatives)]["kept_after_sequence_filter"], 1)
                mapping = read_records(output / "family_mapping.tsv")
                excluded = read_records(output / "benchmark_entities.tsv")
                self.assertEqual(excluded, [{"entity_id": "a"}])
                manifests = build_dataset_splits(mapping, [[
                    {"chain1": "a", "chain2": "c"},
                    {"chain1": "b", "chain2": "d", "source_is_positive": False, "CLUSTER": "1"},
                ]], benchmark_entities={row["entity_id"] for row in excluded}, validation_fraction=0)
                self.assertEqual(len(manifests["train"]), 1)
                self.assertFalse(manifests["train"][0]["source_is_positive"])
                self.assertTrue(manifests["leakage"]["passed"])
                with self.assertRaisesRegex(SystemExit, "changed"):
                    main(argv + ["--coverage", "0.8"])
                self.assertEqual(main(argv + ["--coverage", "0.8", "--overwrite"]), 0)
                self.assertEqual(len(calls), 4)
                self.assertEqual(json.loads((output / "run_metadata.json").read_text())["effective_coverage"], 0.8)
                fasta.write_text(fasta.read_text().replace("GGGA", "GGGT"))
                with self.assertRaisesRegex(SystemExit, "changed"):
                    main(argv + ["--coverage", "0.8"])


if __name__ == "__main__":
    unittest.main()
