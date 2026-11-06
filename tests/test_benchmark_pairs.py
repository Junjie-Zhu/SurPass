"""Length cap and out-of-memory skips for the external PPI benchmarks."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

import common  # noqa: E402


def _load_rf2():
    spec = importlib.util.spec_from_file_location(
        "run_rf2_ppi",
        BENCHMARK / "run_rf2-ppi.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _item(pair: str, sequence_1: str, sequence_2: str, label: bool = True) -> dict:
    protein_1, protein_2 = pair.split("_")
    return {
        "pair": pair,
        "protein_1": protein_1,
        "protein_2": protein_2,
        "label": label,
        "sequence_1": sequence_1,
        "sequence_2": sequence_2,
    }


class ScorePairsTest(unittest.TestCase):
    def test_long_pairs_and_out_of_memory_are_skipped(self):
        items = [
            _item("ok_one", "AC", "DE", True),
            _item("too_long", "A" * 2000, "C" * 100, False),
            _item("oom_pair", "AAAA", "CCCC", True),
            _item("ok_two", "GG", "TT", False),
        ]
        seen = []

        def score(sequence_1: str, sequence_2: str) -> float:
            seen.append(sequence_1 + sequence_2)
            if sequence_1 == "AAAA":
                raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")
            return 0.8 if sequence_1 == "AC" else 0.2

        rows, skipped = common.score_test_pairs(
            items,
            score,
            threshold=0.5,
            desc="test",
            maximum_combined_length=2048,
        )

        self.assertEqual([row["Protein pairs"] for row in rows], ["ok_one", "ok_two"])
        self.assertEqual(rows[0]["pred_interaction"], True)
        self.assertEqual(rows[1]["pred_interaction"], False)
        self.assertEqual(
            [(row["Protein pairs"], row["status"]) for row in skipped],
            [("too_long", "unsupported_length"), ("oom_pair", "out_of_memory")],
        )
        self.assertNotIn("A" * 2000, "".join(seen))

    def test_other_failures_still_stop_the_run(self):
        items = [_item("bad_pair", "AA", "CC")]

        def score(sequence_1: str, sequence_2: str) -> float:
            raise RuntimeError("missing checkpoint")

        with self.assertRaisesRegex(RuntimeError, "missing checkpoint"):
            common.score_test_pairs(items, score, threshold=0.5, desc="test")

    def test_default_limit_is_2048(self):
        self.assertEqual(common.MAX_COMBINED_LENGTH, 2048)


class RoseTTAFoldAttemptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rf2 = _load_rf2()

    def test_attempt_classification(self):
        classify = self.rf2.classify_rf2_attempt
        self.assertEqual(
            classify(0, "", log_written=True),
            "complete",
        )
        self.assertEqual(
            classify(1, "CUDA out of memory", log_written=True),
            "skip_oom",
        )
        self.assertEqual(
            classify(137, "", log_written=True),
            "skip_oom",
        )
        self.assertEqual(
            classify(1, "CUDA out of memory", log_written=False),
            "fatal",
        )
        self.assertEqual(
            classify(1, "ValueError: bad MSA", log_written=True),
            "fatal",
        )

    def _workspace(self, root: Path):
        rf2_root = root / "RoseTTAFold2-PPI"
        model_dir = rf2_root / "src" / "models"
        model_dir.mkdir(parents=True)
        (model_dir / "RF2-PPI.pt").write_text("checkpoint")
        (rf2_root / "src" / "predict_list_PPI.py").write_text("print('unused')\n")
        msa_dir = root / "msas"
        msa_dir.mkdir()
        fasta = [">b1\nAAAA\n>b2\nCCCC\n>a1\nGG\n>a2\nTT\n>c1\n", "A" * 2000, "\n>c2\n", "C" * 2000, "\n"]
        (root / "pairs.fasta").write_text("".join(fasta))
        (root / "pairs.tsv").write_text(
            "Protein pairs\tCategory\n"
            "b1_b2\t1\n"
            "a1_a2\t0\n"
            "c1_c2\t1\n"
        )
        for pair in ("b1_b2", "a1_a2"):
            (msa_dir / f"{pair}.i90.a3m").write_text(f">{pair}\nAC\n")
        return rf2_root, msa_dir

    def _argv(self, root: Path, rf2_root: Path, msa_dir: Path) -> list[str]:
        return [
            "--test-set-tsv",
            str(root / "pairs.tsv"),
            "--fasta",
            str(root / "pairs.fasta"),
            "--rf2-root",
            str(rf2_root),
            "--paired-msa-dir",
            str(msa_dir),
            "--logging-dir",
            str(root / "logs"),
            "--gpu-poll-seconds",
            "10",
        ]

    def test_oom_pair_is_skipped_and_the_rest_are_scored(self):
        calls = []

        def fake_run(command, cwd, env, stdout_path, poll_seconds):
            if len(calls) > 3:
                raise AssertionError("RF2 retry did not make progress")
            list_path = Path(command[command.index("-list_fn") + 1])
            rows = list_path.read_text(encoding="utf-8").splitlines()
            calls.append(rows)
            log_path = Path(f"{list_path}.log")
            if len(calls) == 1:
                log_path.write_text(
                    "Input_MSA\tInteraction_probability\tCompute_time\n",
                    encoding="utf-8",
                )
                stdout_path.write_text(
                    "RuntimeError: CUDA out of memory\n",
                    encoding="utf-8",
                )
                return 1, 1.0, 10.0
            body = "".join(f"{row.split(chr(9))[0]}\t0.7\t0.2\n" for row in rows)
            log_path.write_text(
                "Input_MSA\tInteraction_probability\tCompute_time\n" + body,
                encoding="utf-8",
            )
            stdout_path.write_text("done\n", encoding="utf-8")
            return 0, 2.0, 8.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rf2_root, msa_dir = self._workspace(root)
            with patch.object(self.rf2, "_run_rf2", side_effect=fake_run):
                status = self.rf2.main(self._argv(root, rf2_root, msa_dir))
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(calls[0]), 2)
            self.assertEqual(len(calls[1]), 1)
            self.assertIn("a1_a2", calls[1][0])
            skipped = (root / "logs").rglob("skipped.tsv")
            skipped_text = next(skipped).read_text(encoding="utf-8")
            self.assertIn("b1_b2", skipped_text)
            self.assertIn("out_of_memory", skipped_text)
            self.assertIn("c1_c2", skipped_text)
            self.assertIn("unsupported_length", skipped_text)
            predictions = next((root / "logs").rglob("predictions.tsv")).read_text(
                encoding="utf-8"
            )
            self.assertIn("a1_a2", predictions)
            self.assertNotIn("b1_b2", predictions)
            self.assertNotIn("c1_c2", predictions)

    def test_startup_out_of_memory_does_not_retry_forever(self):
        calls = []

        def fake_run(command, cwd, env, stdout_path, poll_seconds):
            calls.append(command)
            if len(calls) > 1:
                raise AssertionError("model-load OOM was treated as a pair failure")
            stdout_path.write_text("CUDA out of memory\n", encoding="utf-8")
            return 1, 0.1, None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rf2_root, msa_dir = self._workspace(root)
            with patch.object(self.rf2, "_run_rf2", side_effect=fake_run):
                with self.assertRaises(SystemExit) as raised:
                    self.rf2.main(self._argv(root, rf2_root, msa_dir))
            self.assertIn("out of memory", str(raised.exception).lower())
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
