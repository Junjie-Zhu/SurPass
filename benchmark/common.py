"""Shared test-set loading and metric writing for external PPI benchmarks."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (  # noqa: E402
    load_fasta_sequences,
    parse_ppi_label,
    parse_protein_pair,
    resolve_label_column,
)
from src.runtime.metrics import metrics_from_rows  # noqa: E402

DEFAULT_DATA_ROOT = "/groups/sbinlab/clschf/datasets/PDB_PPI"
BENCHMARK_COLUMNS = [
    "Protein pairs",
    "Category",
    "p_interaction",
    "pred_interaction",
]


def load_test_pairs(test_set_tsv: str | Path, fasta_path: str | Path) -> list[dict[str, Any]]:
    """Pairs from the SurPass test TSV, with sequences taken from the FASTA."""
    metadata = pd.read_csv(test_set_tsv, sep="\t")
    if "Protein pairs" not in metadata.columns:
        raise ValueError("Test set TSV is missing the 'Protein pairs' column.")
    label_column = resolve_label_column(metadata.columns)
    sequences = load_fasta_sequences(fasta_path)
    items: list[dict[str, Any]] = []
    for _, row in metadata.iterrows():
        pair = str(row["Protein pairs"]).strip()
        protein_1, protein_2 = parse_protein_pair(pair)
        if protein_1 not in sequences:
            raise KeyError(f"Missing FASTA sequence for {protein_1}")
        if protein_2 not in sequences:
            raise KeyError(f"Missing FASTA sequence for {protein_2}")
        items.append(
            {
                "pair": pair,
                "protein_1": protein_1,
                "protein_2": protein_2,
                "label": parse_ppi_label(row[label_column]),
                "sequence_1": sequences[protein_1],
                "sequence_2": sequences[protein_2],
            }
        )
    return items


def score_test_pairs(
    items: list[dict[str, Any]],
    score_sequences: Callable[[str, str], float],
    threshold: float,
    desc: str,
) -> list[dict[str, Any]]:
    """Score every pair on its full sequences."""
    rows: list[dict[str, Any]] = []
    for item in tqdm(items, desc=desc):
        probability = float(score_sequences(item["sequence_1"], item["sequence_2"]))
        rows.append(
            {
                "Protein pairs": item["pair"],
                "Category": bool(item["label"]),
                "p_interaction": probability,
                "pred_interaction": probability >= float(threshold),
            }
        )
    return rows


def prepare_logging_dir(prefix: str, logging_dir: str | Path = "logs") -> Path:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(logging_dir) / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_benchmark_results(
    logging_dir: str | Path,
    rows: list[dict[str, Any]],
    positive_weight: float,
) -> tuple[Path, Path]:
    logging_dir = Path(logging_dir)
    predictions_path = logging_dir / "predictions.tsv"
    metrics_path = logging_dir / "metrics.csv"
    pd.DataFrame(rows, columns=BENCHMARK_COLUMNS).to_csv(
        predictions_path,
        sep="\t",
        index=False,
    )
    weights = [1.0]
    if float(positive_weight) != 1.0:
        weights.append(float(positive_weight))
    metrics_by_head = {
        "interaction": {
            weight: metrics_from_rows(
                rows,
                "p_interaction",
                positive_weight=weight,
            )
            for weight in weights
        }
    }
    pd.DataFrame(
        [
            {
                "head": "interaction",
                "positive_weight": weight,
                "AUROC": metrics["auroc"],
                "AUPRC": metrics["auprc"],
                "N": metrics["n"],
                "N_pos": metrics["n_pos"],
                "N_neg": metrics["n_neg"],
            }
            for weight, metrics in metrics_by_head["interaction"].items()
        ]
    ).to_csv(metrics_path, index=False)
    for weight, metrics in metrics_by_head["interaction"].items():
        print(
            f"[test-metrics] head=interaction "
            f"positive_weight={weight:g} "
            f"auroc={metrics['auroc']:.4f} "
            f"auprc={metrics['auprc']:.4f}"
        )
    return predictions_path, metrics_path


def require_file(path: str | Path, description: str) -> Path:
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"Missing {description}: {file_path}")
    return file_path


def resolve_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(gpu_id)}")
    print("CUDA is not available; running on CPU.")
    return torch.device("cpu")
