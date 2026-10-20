#!/usr/bin/env python3
"""Run the unmodified RoseTTAFold2-PPI predictor on the SurPass test set.

This adapter resolves precomputed paired MSAs, writes RF2-PPI's native input
list, invokes the sibling repository in a subprocess, then converts its output
to the shared SurPass benchmark prediction and metric files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATA_ROOT = "/groups/sbinlab/clschf/datasets/PDB_PPI"
DEFAULT_RF2_ROOT = Path(__file__).resolve().parents[2] / "RoseTTAFold2-PPI"
RUNTIME_FIELDS = (
    "Protein pairs",
    "paired_msa",
    "p1_length",
    "preprocessing_latency_seconds",
    "model_latency_seconds",
    "process_peak_gpu_memory_mib",
    "output_shape",
    "success",
    "error",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RoseTTAFold2-PPI adapter for the SurPass test TSV/FASTA. "
            "Precomputed paired MSAs are required."
        )
    )
    parser.add_argument(
        "--test-set-tsv",
        default=str(Path(DEFAULT_DATA_ROOT) / "test_set.tsv"),
    )
    parser.add_argument(
        "--fasta",
        default=str(Path(DEFAULT_DATA_ROOT) / "test_set.fasta"),
    )
    parser.add_argument(
        "--rf2-root",
        type=Path,
        default=DEFAULT_RF2_ROOT,
        help=f"RoseTTAFold2-PPI checkout (default: {DEFAULT_RF2_ROOT}).",
    )
    parser.add_argument(
        "--paired-msa-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing <protein1>_<protein2>.i90.a3m files. "
            "RF2-PPI requires paired MSAs; this argument is required."
        ),
    )
    parser.add_argument(
        "--msa-suffix",
        default=".i90.a3m",
        help="Suffix appended to each 'Protein pairs' value (default: .i90.a3m).",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="RF2-PPI checkpoint (default: <rf2-root>/src/models/RF2-PPI.pt).",
    )
    parser.add_argument(
        "--predict-script",
        type=Path,
        default=None,
        help="RF2 predictor (default: <rf2-root>/src/predict_list_PPI.py).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the RF2-PPI environment.",
    )
    parser.add_argument("--number-seqs", type=int, default=5000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--positive-weight", type=float, default=0.01)
    parser.add_argument(
        "--maximum-combined-length",
        type=int,
        default=2048,
        help="Use the same dense-length subset as SurPass (default: 2048).",
    )
    parser.add_argument("--logging-dir", default="logs")
    parser.add_argument(
        "--require-complete-coverage",
        action="store_true",
        help="Fail instead of benchmarking only pairs with available MSAs.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.paired_msa_dir is None:
        raise SystemExit(
            "RF2-PPI requires precomputed paired MSAs. Pass "
            "--paired-msa-dir /path/to/paired_msas; files must be named "
            "<Protein pairs>.i90.a3m (or set --msa-suffix)."
        )
    paired_msa_dir = args.paired_msa_dir.expanduser().resolve()
    if not paired_msa_dir.is_dir():
        raise SystemExit(f"Missing paired-MSA directory: {paired_msa_dir}")
    if args.number_seqs < 1:
        raise SystemExit("--number-seqs must be positive.")
    if args.gpu_id < 0:
        raise SystemExit("--gpu-id must be non-negative.")
    if args.gpu_poll_seconds <= 0:
        raise SystemExit("--gpu-poll-seconds must be positive.")
    if args.maximum_combined_length < 2:
        raise SystemExit("--maximum-combined-length must be at least 2.")

    rf2_root = args.rf2_root.expanduser().resolve()
    if not rf2_root.is_dir():
        raise SystemExit(
            f"Missing RoseTTAFold2-PPI checkout: {rf2_root}. "
            "Set --rf2-root to the repository path."
        )
    model_file = (
        args.model_file.expanduser().resolve()
        if args.model_file is not None
        else rf2_root / "src" / "models" / "RF2-PPI.pt"
    )
    if not model_file.is_file():
        raise SystemExit(
            f"Missing RF2-PPI checkpoint: {model_file}. Download RF2-PPI.pt "
            "as described by the RF2-PPI README, or pass --model-file."
        )
    predict_script = (
        args.predict_script.expanduser().resolve()
        if args.predict_script is not None
        else rf2_root / "src" / "predict_list_PPI.py"
    )
    if not predict_script.is_file():
        raise SystemExit(
            f"Missing RF2-PPI predictor: {predict_script}. "
            "Pass --predict-script if the checkout layout differs."
        )
    return paired_msa_dir, model_file, predict_script


def _resolve_python(executable: str) -> str:
    expanded = Path(executable).expanduser()
    if expanded.parent != Path(".") or os.sep in executable:
        if not expanded.is_file():
            raise SystemExit(
                f"Missing RF2 Python executable: {expanded}. "
                "Pass --python from the RF2-PPI environment."
            )
        return str(expanded.resolve())
    resolved = shutil.which(executable)
    if resolved is None:
        raise SystemExit(
            f"Cannot find RF2 Python executable {executable!r}; pass --python."
        )
    return resolved


def _resolve_msa(
    paired_msa_dir: Path,
    pair: str,
    suffix: str,
) -> Path | None:
    candidate = paired_msa_dir / f"{pair}{suffix}"
    if candidate.is_file():
        return candidate.resolve()
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_runtime_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _query_process_gpu_memory_mib(pid: int) -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    total = 0.0
    found = False
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            row_pid = int(fields[0])
            memory_mib = float(fields[1])
        except ValueError:
            continue
        if row_pid == pid:
            total += memory_mib
            found = True
    return total if found else None


def _monitor_gpu_memory(
    pid: int,
    interval_seconds: float,
    stop: threading.Event,
    result: dict[str, Any],
) -> None:
    peak: float | None = None
    while not stop.is_set():
        current = _query_process_gpu_memory_mib(pid)
        if current is not None:
            peak = current if peak is None else max(peak, current)
        stop.wait(interval_seconds)
    current = _query_process_gpu_memory_mib(pid)
    if current is not None:
        peak = current if peak is None else max(peak, current)
    result["peak_gpu_memory_mib"] = peak


def _run_rf2(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    poll_seconds: float,
) -> tuple[int, float, float | None]:
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        raise SystemExit(
            f"Could not start RF2-PPI ({exc}). Check --python and the RF2 environment."
        ) from exc

    stop = threading.Event()
    monitor_result: dict[str, Any] = {}
    monitor = threading.Thread(
        target=_monitor_gpu_memory,
        args=(process.pid, poll_seconds, stop, monitor_result),
        daemon=True,
    )
    monitor.start()
    stdout, _ = process.communicate()
    stop.set()
    monitor.join(timeout=max(5.0, poll_seconds * 2))
    elapsed = time.perf_counter() - started
    stdout_path.write_text(stdout or "", encoding="utf-8")
    return process.returncode, elapsed, monitor_result.get("peak_gpu_memory_mib")


def _parse_rf2_log(
    log_path: Path,
    msa_to_item: dict[str, dict[str, Any]],
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    predictions: dict[str, tuple[float, float]] = {}
    errors: dict[str, str] = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line_number == 1:
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                continue
            msa_key = str(Path(fields[0]).expanduser().resolve())
            item = msa_to_item.get(msa_key)
            if item is None:
                continue
            pair = item["pair"]
            try:
                predictions[pair] = (float(fields[1]), float(fields[2]))
            except ValueError:
                errors[pair] = line
    return predictions, errors


def _npz_shapes(npz_path: Path) -> dict[str, list[int]]:
    if not npz_path.is_file():
        return {}
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as archive:
        return {
            str(Path(key).expanduser().resolve()): list(archive[key].shape)
            for key in archive.files
        }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paired_msa_dir, model_file, predict_script = _validate_args(args)
    rf2_python = _resolve_python(args.python)

    from common import (
        load_test_pairs,
        prepare_logging_dir,
        require_file,
        write_benchmark_results,
    )

    test_set = require_file(args.test_set_tsv, "test set TSV")
    fasta = require_file(args.fasta, "FASTA")

    logging_dir = prepare_logging_dir("RF2-PPI", args.logging_dir)
    native_input_path = (logging_dir / "rf2_input.tsv").resolve()
    runtime_json_path = logging_dir / "runtime.json"
    runtime_csv_path = logging_dir / "runtime.csv"
    stdout_path = logging_dir / "rf2_stdout.log"

    preprocess_started = time.perf_counter()
    all_items = load_test_pairs(test_set, fasta)
    unsupported = [
        item["pair"]
        for item in all_items
        if len(item["sequence_1"]) + len(item["sequence_2"])
        > args.maximum_combined_length
    ]
    items = [
        item
        for item in all_items
        if len(item["sequence_1"]) + len(item["sequence_2"])
        <= args.maximum_combined_length
    ]
    covered: list[dict[str, Any]] = []
    missing: list[str] = []
    per_pair_preprocess: dict[str, float] = {}
    for item in items:
        pair_started = time.perf_counter()
        msa_path = _resolve_msa(paired_msa_dir, item["pair"], args.msa_suffix)
        per_pair_preprocess[item["pair"]] = time.perf_counter() - pair_started
        if msa_path is None:
            missing.append(item["pair"])
            continue
        covered.append({**item, "msa_path": msa_path})

    with native_input_path.open("w", encoding="utf-8") as handle:
        for item in covered:
            handle.write(f"{item['msa_path']}\t{len(item['sequence_1'])}\n")
    preprocessing_seconds = time.perf_counter() - preprocess_started
    coverage = (len(covered) / len(items)) if items else 0.0
    overall_coverage = len(covered) / max(1, len(all_items))

    command = [
        rf2_python,
        str(predict_script),
        "-list_fn",
        str(native_input_path),
        "-model_file",
        str(model_file),
        "-number_seqs",
        str(args.number_seqs),
    ]
    metadata: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "test_set_tsv": str(test_set.resolve()),
        "fasta": str(fasta.resolve()),
        "paired_msa_dir": str(paired_msa_dir),
        "model_file": str(model_file),
        "predict_script": str(predict_script),
        "command": command,
        "n_total": len(all_items),
        "n_length_supported": len(items),
        "n_unsupported_length": len(unsupported),
        "unsupported_length_pairs": unsupported,
        "maximum_combined_length": args.maximum_combined_length,
        "n_covered": len(covered),
        "n_missing": len(missing),
        "supported_subset_coverage": coverage,
        "overall_coverage": overall_coverage,
        "missing_pairs": missing,
        "preprocessing_latency_seconds": preprocessing_seconds,
        "model_latency_source": (
            "RF2-PPI Compute_time; includes its input tensor preparation and forward pass"
        ),
        "process_peak_gpu_memory_source": (
            "nvidia-smi process sampling; null when unavailable"
        ),
    }
    _write_json(runtime_json_path, metadata)

    if not covered:
        expected = paired_msa_dir / f"{items[0]['pair']}{args.msa_suffix}" if items else None
        raise SystemExit(
            "No paired MSAs matched the test set. "
            + (f"For example, expected: {expected}" if expected is not None else "")
        )
    if missing and args.require_complete_coverage:
        sample = ", ".join(missing[:5])
        raise SystemExit(
            f"Paired-MSA coverage is {len(covered)}/{len(items)}; missing: {sample}. "
            "Add the files or omit --require-complete-coverage."
        )
    if missing:
        print(
            f"Warning: paired-MSA coverage is {len(covered)}/{len(items)} "
            f"({coverage:.1%}); benchmarking covered pairs only."
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    returncode, subprocess_seconds, peak_gpu_memory_mib = _run_rf2(
        command,
        cwd=args.rf2_root.expanduser().resolve(),
        env=env,
        stdout_path=stdout_path,
        poll_seconds=args.gpu_poll_seconds,
    )
    metadata.update(
        {
            "status": "rf2_failed" if returncode else "rf2_finished",
            "rf2_returncode": returncode,
            "subprocess_wall_time_seconds": subprocess_seconds,
            "process_peak_gpu_memory_mib": peak_gpu_memory_mib,
        }
    )
    _write_json(runtime_json_path, metadata)
    if returncode != 0:
        output_lines = stdout_path.read_text(encoding="utf-8").splitlines()
        excerpt = "\n".join(output_lines[-12:])
        raise SystemExit(
            f"RF2-PPI exited with code {returncode}. Full output: {stdout_path}\n"
            f"{excerpt}"
        )

    native_log_path = Path(f"{native_input_path}.log")
    native_npz_path = Path(f"{native_input_path}.npz")
    if not native_log_path.is_file():
        raise SystemExit(
            f"RF2-PPI completed but did not write its expected log: {native_log_path}"
        )

    msa_to_item = {str(item["msa_path"].resolve()): item for item in covered}
    predictions, rf2_errors = _parse_rf2_log(native_log_path, msa_to_item)
    shapes = _npz_shapes(native_npz_path)
    prediction_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for item in covered:
        pair = item["pair"]
        prediction = predictions.get(pair)
        error = rf2_errors.get(pair, "")
        probability = prediction[0] if prediction is not None else None
        latency = prediction[1] if prediction is not None else None
        if prediction is None and not error:
            error = "RF2-PPI log contained no result for this paired MSA"
        if probability is not None:
            prediction_rows.append(
                {
                    "Protein pairs": pair,
                    "Category": bool(item["label"]),
                    "p_interaction": probability,
                    "pred_interaction": probability >= float(args.threshold),
                }
            )
        runtime_rows.append(
            {
                "Protein pairs": pair,
                "paired_msa": str(item["msa_path"]),
                "p1_length": len(item["sequence_1"]),
                "preprocessing_latency_seconds": per_pair_preprocess[pair],
                "model_latency_seconds": latency,
                "process_peak_gpu_memory_mib": peak_gpu_memory_mib,
                "output_shape": json.dumps(
                    shapes.get(str(item["msa_path"].resolve())),
                    separators=(",", ":"),
                ),
                "success": prediction is not None,
                "error": error,
            }
        )

    _write_runtime_csv(runtime_csv_path, runtime_rows)
    predictions_path, metrics_path = write_benchmark_results(
        logging_dir,
        prediction_rows,
        positive_weight=float(args.positive_weight),
    )
    model_latencies = [
        float(row["model_latency_seconds"])
        for row in runtime_rows
        if row["model_latency_seconds"] is not None
    ]
    metadata.update(
        {
            "status": "complete",
            "n_scored": len(prediction_rows),
            "n_rf2_errors": len(covered) - len(prediction_rows),
            "model_latency_seconds_total": sum(model_latencies),
            "model_latency_seconds_mean": (
                sum(model_latencies) / len(model_latencies)
                if model_latencies
                else None
            ),
            "native_log": str(native_log_path),
            "native_npz": str(native_npz_path) if native_npz_path.is_file() else None,
        }
    )
    _write_json(runtime_json_path, metadata)

    print(
        f"[test-metrics] n={len(prediction_rows)} "
        f"supported_coverage={coverage:.3f} overall_coverage={overall_coverage:.3f}"
    )
    print(f"Wrote {predictions_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {runtime_csv_path}")
    print(f"Wrote {runtime_json_path}")
    return 0 if len(prediction_rows) == len(covered) else 1


if __name__ == "__main__":
    raise SystemExit(main())
