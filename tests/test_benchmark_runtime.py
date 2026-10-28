"""Benchmark the selected architecture and retain useful timing/OOM reports."""

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import benchmark_runtime as benchmark
from src.runtime.config import create_model


class RuntimeBenchmarkTest(unittest.TestCase):
    def test_defaults_select_composed_twelve_layer_respair_at_2048(self):
        args = benchmark.build_arg_parser().parse_args([])
        benchmark._validate_args(args)
        cfg = benchmark._load_config(args)
        self.assertEqual(benchmark.DEFAULT_LENGTHS, (2048,))
        self.assertEqual(cfg.model.name, "ResPair")
        self.assertEqual(cfg.model.n_layers, 12)
        self.assertEqual(cfg.model.pair_update_every, 2)
        self.assertEqual(cfg.model.recycle_passes, 1)
        self.assertEqual(cfg.model.plm_in_dim, 2560)
        self.assertEqual(cfg.model.dim_token, 384)
        self.assertEqual(cfg.data.maximum_length, 2048)

    def test_default_and_unbalanced_2048_inputs_have_correct_layout(self):
        for first in (None, 512):
            batch = benchmark._synthetic_batch(2048, 8, 42, torch.device("cpu"), torch, batch_size=2, p1_length=first)
            self.assertEqual(batch["plm_emb"].shape, (2, 2048, 8))
            self.assertEqual(batch["p1_length"].tolist(), [first or 1024] * 2)
            self.assertEqual(batch["p2_length"].tolist(), [2048 - (first or 1024)] * 2)
            self.assertTrue(batch["padding_mask"].all())
            self.assertEqual(batch["chain_index"][0].sum().item(), 2048 - (first or 1024))

    def test_smoke_reduces_size_but_keeps_a_real_pair_update(self):
        args = benchmark.build_arg_parser().parse_args(["--cpu-smoke"])
        cfg = benchmark._load_config(args)
        model = create_model(cfg)
        self.assertEqual(model.n_layers, 2)
        self.assertEqual(list(model.pair_blocks), ["1"])

    def _run(self, extra, training=False, fail=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ["--device", "cpu", "--warmup", "1", "--repeats", "2",
                    "--output-json", str(root / "result.json"), "--output-csv", str(root / "result.csv"),
                    "--override", "data.distance_bin_count=16",
                    "--override", "data.distance_bin_convention=legacy", *extra]
            output = io.StringIO()
            with redirect_stdout(output):
                if fail:
                    with patch.object(benchmark, "_run_iteration", side_effect=RuntimeError("CUDA out of memory")):
                        code = benchmark.main(args)
                else:
                    code = benchmark.main(args)
            report = json.loads((root / "result.json").read_text())
            with (root / "result.csv").open() as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 1)
            return code, report, output.getvalue()

    def test_inference_smoke_reports_architecture_latency_and_shapes(self):
        code, report, output = self._run(["--cpu-smoke"])
        row = report["results"][0]
        self.assertEqual(code, 0)
        self.assertTrue(row["success"])
        self.assertEqual(row["model_name"], "ResPair")
        self.assertEqual(row["architecture"], "res_pair_v1")
        self.assertEqual(row["pair_update_layers"], [2])
        self.assertEqual(row["output_shapes"]["pair_logits"], [1, 8, 8, 16])
        self.assertEqual(len(row["iteration_times_seconds"]), 2)
        self.assertEqual(len(row["warmup_times_seconds"]), 1)
        self.assertGreater(row["wall_time_seconds"], 0)
        self.assertIsNone(row["peak_allocated_cuda_bytes"])
        self.assertIn("mean=", output)
        self.assertIn("starting", output)

    def test_training_keeps_twelve_layers_and_measures_real_optimizer_steps(self):
        overrides = {
            "model.dim_token": 24, "model.dim_pair": 12, "model.n_heads": 3,
            "model.plm_in_dim": 8, "model.plm_out_dim": 12, "model.index_emb_dim": 8,
            "model.triangle_hidden_dim": 8, "model.triangle_chunk_size": 2,
            "model.allow_missing_physics": "true",
        }
        extra = ["--mode", "training", "--batch-size", "2", "--lengths", "8"]
        for key, value in overrides.items():
            extra += ["--override", f"{key}={value}"]
        code, report, _ = self._run(extra)
        row = report["results"][0]
        self.assertEqual(code, 0)
        self.assertEqual(row["n_layers"], 12)
        self.assertEqual(row["pair_update_layers"], [2, 4, 6, 8, 10, 12])
        self.assertEqual(row["mode"], "training")
        self.assertEqual(row["output_shapes"]["pair_logits"], [2, 4, 4, 16])
        self.assertGreater(row["last_loss"], 0)
        self.assertAlmostEqual(row["seconds_per_sample"], row["wall_time_seconds"] / 2)
        self.assertIn("optimizer step", report["metadata"]["measurement_scope"])

    def test_oom_retains_failure_stage_and_reports_nonzero_exit(self):
        code, report, output = self._run(["--cpu-smoke"], fail=True)
        row = report["results"][0]
        self.assertEqual(code, 1)
        self.assertFalse(row["success"])
        self.assertIn("out of memory", row["error"])
        self.assertEqual(row["error_stage"], "warmup iteration 1/1")
        self.assertIn("FAIL", output)

    def test_cuda_peak_stats_and_synchronization_use_the_selected_device(self):
        cuda = Mock()
        cuda.max_memory_allocated.return_value = 123
        cuda.max_memory_reserved.return_value = 456
        device = SimpleNamespace(type="cuda")
        row = {}
        benchmark._synchronize(device, SimpleNamespace(cuda=cuda))
        benchmark._capture_memory(row, device, SimpleNamespace(cuda=cuda))
        cuda.synchronize.assert_called_once_with(device)
        self.assertEqual(row["peak_allocated_cuda_bytes"], 123)
        self.assertEqual(row["peak_reserved_cuda_bytes"], 456)

    def test_invalid_chain_split_and_batch_size_are_rejected(self):
        for extra in (["--p1-length", "2048"], ["--batch-size", "0"]):
            args = benchmark.build_arg_parser().parse_args(extra)
            with self.assertRaises(SystemExit):
                benchmark._validate_args(args)


if __name__ == "__main__":
    unittest.main()
