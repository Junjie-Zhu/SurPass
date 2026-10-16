"""Score the SurPass test set with PPLM-PPI.

Each pair is scored on its full sequences in one forward.
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPLM-PPI on the SurPass PPI test set")
    parser.add_argument(
        "--test-set-tsv",
        default=str(Path(DEFAULT_DATA_ROOT) / "test_set.tsv"),
    )
    parser.add_argument(
        "--fasta",
        default=str(Path(DEFAULT_DATA_ROOT) / "test_set.fasta"),
    )
    parser.add_argument(
        "--weights",
        default=str(BENCHMARK_DIR / "weights" / "ppi_models.pkl"),
        help="Ensemble of PPLM-PPI heads.",
    )
    parser.add_argument(
        "--backbone",
        default=str(BENCHMARK_DIR / "weights" / "pplm_t33_650M.pt"),
        help="PPLM t33 backbone checkpoint.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--positive-weight", type=float, default=0.01)
    parser.add_argument("--logging-dir", default="logs")
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def load_pplm(backbone_path: Path, device: torch.device):
    try:
        from pplm import Alphabet, PPLM
    except ImportError as exc:
        raise SystemExit(
            "PPLM is not installed. Install the pplm package before running this script."
        ) from exc

    alphabet = Alphabet.from_architecture()
    batch_converter = alphabet.get_batch_converter()
    model_data = torch.load(backbone_path, map_location="cpu")
    model_param = model_data["param"]
    model = PPLM(
        num_layers=model_param["encoder_layers"],
        embed_dim=model_param["encoder_embed_dim"],
        attention_heads=model_param["encoder_attention_heads"],
        token_dropout=False,
        alphabet=alphabet,
    )
    model.to(device)
    model.load_state_dict(model_data["model"], strict=False)
    model.eval()
    return model, batch_converter, int(model_param["encoder_layers"]), int(
        model_param["encoder_attention_heads"]
    )


def pplm_features(seq_a: str, seq_b: str, device: torch.device, model, batch_converter, n_layers: int, n_heads: int):
    with torch.no_grad():
        _, _, tokens_a = batch_converter([("seqA", seq_a)])
        _, _, tokens_b = batch_converter([("seqB", seq_b)])
        tokens = torch.cat([tokens_a, tokens_b], dim=-1).to(device)
        length_a = len(seq_a)
        length_b = len(seq_b)
        token_length = length_a + 2 + length_b + 2
        inter_chain_mask = torch.ones((token_length, token_length), device=device)
        inter_chain_mask[: length_a + 2, : length_a + 2] = 0
        inter_chain_mask[length_a + 2 :, length_a + 2 :] = 0

        out = model(
            tokens,
            inter_chain_mask,
            repr_layers=[n_layers],
            need_head_weights=True,
            return_contacts=False,
        )
        embed_a = out["representations"][n_layers][0, 1 : (length_a + 1), :]
        embed_b = out["representations"][n_layers][0, -(length_b + 1) : -1, :]
        attentions = out["attentions"].squeeze()
        attn_aa = attentions[:, :, 1 : (length_a + 1), 1 : (length_a + 1)].reshape(
            n_layers * n_heads, length_a, length_a
        )
        attn_ab = attentions[:, :, 1 : (length_a + 1), -(length_b + 1) : -1].reshape(
            n_layers * n_heads, length_a, length_b
        )
        attn_ba = attentions[:, :, -(length_b + 1) : -1, 1 : (length_a + 1)].reshape(
            n_layers * n_heads, length_b, length_a
        )
        attn_bb = attentions[:, :, -(length_b + 1) : -1, -(length_b + 1) : -1].reshape(
            n_layers * n_heads, length_b, length_b
        )
        inter_attn = (attn_ab + attn_ba.transpose(1, 2)) / 2
        features = (
            inter_attn.mean(dim=(1, 2)),
            attn_aa.mean(dim=(1, 2)),
            attn_bb.mean(dim=(1, 2)),
            embed_a.mean(dim=0),
            embed_b.mean(dim=0),
            torch.amax(inter_attn, dim=(1, 2)),
            torch.amax(attn_aa, dim=(1, 2)),
            torch.amax(attn_bb, dim=(1, 2)),
            torch.amax(embed_a, dim=0),
            torch.amax(embed_b, dim=0),
        )
        del out, attentions
        return features


def ensemble_probability(features, model, model_weights) -> float:
    mean_features = features[:5]
    max_features = features[5:]
    predictions = []
    with torch.no_grad():
        for state in model_weights["mean"]:
            model.load_state_dict(state)
            forward = model(*mean_features)
            swapped = model(
                mean_features[0],
                mean_features[2],
                mean_features[1],
                mean_features[4],
                mean_features[3],
            )
            predictions.append((forward + swapped) / 2)
        for state in model_weights["max"]:
            model.load_state_dict(state)
            forward = model(*max_features)
            swapped = model(
                max_features[0],
                max_features[2],
                max_features[1],
                max_features[4],
                max_features[3],
            )
            predictions.append((forward + swapped) / 2)
        stacked = torch.stack(predictions)
        score = torch.mean(stacked, dim=0).reshape(-1)[0]
    return float(score.detach().float().cpu())


def main() -> None:
    args = parse_args()
    test_set = require_file(args.test_set_tsv, "test set TSV")
    fasta = require_file(args.fasta, "FASTA")
    weights_path = require_file(args.weights, "PPLM-PPI weights")
    backbone_path = require_file(args.backbone, "PPLM backbone")

    try:
        from pplm_ppi import PPLM_PPI
    except ImportError as exc:
        raise SystemExit(
            "pplm_ppi is not installed. Install it before running this script."
        ) from exc

    device = resolve_device(args.gpu_id)
    backbone, batch_converter, n_layers, n_heads = load_pplm(backbone_path, device)
    model_weights = torch.load(weights_path, map_location=device)
    head = PPLM_PPI().to(device)
    head.eval()
    items = load_test_pairs(test_set, fasta)

    def score_sequences(sequence_1: str, sequence_2: str) -> float:
        features = pplm_features(
            sequence_1,
            sequence_2,
            device,
            backbone,
            batch_converter,
            n_layers,
            n_heads,
        )
        return ensemble_probability(features, head, model_weights)

    rows = score_test_pairs(
        items,
        score_sequences=score_sequences,
        threshold=float(args.threshold),
        desc="PPLM",
    )
    logging_dir = prepare_logging_dir("PPLM", args.logging_dir)
    predictions_path, metrics_path = write_benchmark_results(
        logging_dir,
        rows,
        positive_weight=float(args.positive_weight),
    )
    print(f"[test-metrics] n={len(rows)}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
