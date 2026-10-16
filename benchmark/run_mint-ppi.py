"""Score the SurPass test set with MINT.

Pairs longer than the ESM context are tiled. The reported probability is the
maximum over those windows, so a long complex is not truncated to its first
residues or dropped from the metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    load_test_pairs,
    prepare_logging_dir,
    require_file,
    resolve_device,
    score_test_pairs,
    write_benchmark_results,
)

# ESM-2 context is 1024 tokens. 1022 leaves room for the boundary tokens MINT adds.
MINT_CONTEXT = 1022


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MINT on the SurPass PPI test set")
    parser.add_argument(
        "--test-set-tsv",
        default=str(Path(DEFAULT_DATA_ROOT) / "test_set.tsv"),
    )
    parser.add_argument(
        "--fasta",
        default=str(Path(DEFAULT_DATA_ROOT) / "test_set.fasta"),
    )
    parser.add_argument(
        "--config",
        default="data/esm2_t33_650M_UR50D.json",
        help="MINT ESM-2 config JSON.",
    )
    parser.add_argument("--checkpoint", default="mint.ckpt", help="MINT checkpoint.")
    parser.add_argument(
        "--mlp-checkpoint",
        default="bernett_mlp.pth",
        help="Bernett MLP checkpoint.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MINT_CONTEXT,
        help="Concatenated residue budget for one MINT forward.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--positive-weight", type=float, default=0.01)
    parser.add_argument("--logging-dir", default="logs")
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def load_mint(
    config_path: Path,
    checkpoint_path: Path,
    mlp_path: Path,
    device: torch.device,
    max_length: int,
):
    try:
        from mint.helpers.extract import CollateFn, MINTWrapper, load_config
        from mint.helpers.predict import SimpleMLP
    except ImportError as exc:
        raise SystemExit(
            "MINT is not installed. Install the mint package before running this script."
        ) from exc

    wrapper = MINTWrapper(
        load_config(str(config_path)),
        str(checkpoint_path),
        sep_chains=True,
        device=str(device),
    )
    model = SimpleMLP()
    model.load_state_dict(torch.load(mlp_path, map_location=device))
    model.eval()
    model.to(device)
    return wrapper, model, CollateFn(int(max_length))


def main() -> None:
    args = parse_args()
    if int(args.max_length) < 2:
        raise SystemExit(f"--max-length must be at least 2, got {args.max_length}.")
    test_set = require_file(args.test_set_tsv, "test set TSV")
    fasta = require_file(args.fasta, "FASTA")
    config_path = require_file(args.config, "MINT config")
    checkpoint_path = require_file(args.checkpoint, "MINT checkpoint")
    mlp_path = require_file(args.mlp_checkpoint, "MINT MLP checkpoint")

    device = resolve_device(args.gpu_id)
    wrapper, model, collate = load_mint(
        config_path,
        checkpoint_path,
        mlp_path,
        device,
        int(args.max_length),
    )
    items = load_test_pairs(test_set, fasta)

    def score_sequences(sequence_1: str, sequence_2: str) -> float:
        chains, chain_ids = collate([(sequence_1, sequence_2)])
        chains = chains.to(device)
        chain_ids = chain_ids.to(device)
        with torch.no_grad():
            embeddings = wrapper(chains, chain_ids)
            probability = torch.sigmoid(model(embeddings))
        return float(probability.reshape(-1)[0].detach().float().cpu())

    rows, stats = score_test_pairs(
        items,
        budget=int(args.max_length),
        score_sequences=score_sequences,
        threshold=float(args.threshold),
        desc="MINT",
    )
    logging_dir = prepare_logging_dir("MINT", args.logging_dir)
    predictions_path, metrics_path = write_benchmark_results(
        logging_dir,
        rows,
        positive_weight=float(args.positive_weight),
    )
    print(
        f"[test-metrics] n={len(rows)} "
        f"n_dropped={stats['n_dropped']} "
        f"n_windowed={stats['n_windowed']} "
        f"max_length={stats['max_length']}"
    )
    print(f"Wrote {predictions_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
