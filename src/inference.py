"""Dense full-complex SurPass inference."""

from __future__ import annotations

import datetime
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from src.common.residue_constants import restype_order, unk_restype_index
from src.data.collate import collate_concatenated_pairs, get_dataloader
from src.model.loss import pair_contact_probability, max_contact_probability
from src.runtime.checkpoint import load_checkpoint
from src.runtime.config import create_model, cfg_get
from src.runtime.device import (
    autocast_context,
    normalize_precision,
    resolve_cuda_device,
    resolve_dist_backend,
    to_device,
)
from src.runtime.metrics import metrics_from_rows
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything


PREDICTION_COLUMNS = (
    "Protein pairs",
    "Category",
    "combined_length",
    "status",
    "p_pair",
    "pred_pair",
    "p_aux_protein",
    "n_inter_contacts",
)


def parse_protein_pair(value: str) -> tuple[str, str]:
    parts = str(value).strip().split("_")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Expected two protein IDs separated by '_': {value!r}")
    return parts[0], parts[1]


def parse_ppi_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "positive", "pos"}:
        return True
    if text in {"false", "0", "negative", "neg"}:
        return False
    raise ValueError(f"Unsupported PPI label: {value!r}")


def resolve_label_column(columns) -> str:
    for name in ("Category", "Catagory"):
        if name in columns:
            return name
    raise ValueError("Test set TSV must contain Category")


def load_fasta_sequences(path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    identifier: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if identifier is not None:
            sequence = "".join(chunks).replace(" ", "").upper()
            if not sequence:
                raise ValueError(f"Empty FASTA sequence for {identifier}")
            records[identifier] = sequence

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                identifier = line[1:].split()[0]
                chunks = []
            else:
                if identifier is None:
                    raise ValueError("FASTA sequence appears before a header")
                chunks.append(line)
    flush()
    return records


def load_plm_embedding(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        payload = payload.get("plm_emb")
    tensor = torch.as_tensor(payload, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"Expected [L,D] embedding in {path}")
    # torch.save preserves backing storage, including unused rows of a view.
    # Keep only the embedding's own values alive after loading an older file.
    if tensor.untyped_storage().nbytes() > tensor.numel() * tensor.element_size():
        tensor = tensor.clone(memory_format=torch.contiguous_format)
    return tensor


def build_residue_features(
    sequence: str,
    plm_embedding: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if len(sequence) != int(plm_embedding.shape[0]):
        raise ValueError("Sequence and PLM embedding lengths differ")
    length = len(sequence)
    return {
        "plm_emb": plm_embedding,
        "residue_type": torch.tensor(
            [
                restype_order.get(residue, unk_restype_index)
                for residue in sequence.upper()
            ],
            dtype=torch.long,
        ),
        "residue_index": torch.arange(length, dtype=torch.long),
        "chain_index": torch.zeros(length, dtype=torch.long),
        "mask": torch.ones(length, dtype=torch.bool),
    }


class InferencePairDataset(Dataset):
    def __init__(
        self,
        test_set_tsv: str | Path,
        fasta_path: str | Path,
        embedding_dir: str | Path,
        *,
        maximum_length: int = 2048,
        embedding_cache_mb: float = 256,
    ):
        cache_mb = float(embedding_cache_mb)
        if not math.isfinite(cache_mb) or cache_mb < 0:
            raise ValueError("embedding_cache_mb must be finite and non-negative")
        frame = pd.read_csv(test_set_tsv, sep="\t")
        label_column = resolve_label_column(frame.columns)
        sequences = load_fasta_sequences(fasta_path)
        self.embedding_dir = Path(embedding_dir)
        self.maximum_length = int(maximum_length)
        self.items: list[dict[str, Any]] = []
        self.unsupported_rows: list[dict[str, Any]] = []
        # Each distributed rank / DataLoader worker owns its own cache.
        self._feature_cache_limit_bytes = int(cache_mb * 1024**2)
        self._feature_cache_bytes = 0
        self._feature_cache: OrderedDict[
            str, tuple[dict[str, torch.Tensor], int]
        ] = OrderedDict()

        for index, row in frame.iterrows():
            pair = str(row["Protein pairs"]).strip()
            protein_1, protein_2 = parse_protein_pair(pair)
            if protein_1 not in sequences or protein_2 not in sequences:
                raise KeyError(f"Missing FASTA sequence for {pair}")
            length = len(sequences[protein_1]) + len(sequences[protein_2])
            item = {
                "index": int(index),
                "pair": pair,
                "protein_1": protein_1,
                "protein_2": protein_2,
                "sequence_1": sequences[protein_1],
                "sequence_2": sequences[protein_2],
                "label": parse_ppi_label(row[label_column]),
                "combined_length": length,
            }
            if length > self.maximum_length:
                self.unsupported_rows.append(
                    {
                        "Protein pairs": pair,
                        "Category": item["label"],
                        "combined_length": length,
                        "status": f"unsupported_length>{self.maximum_length}",
                        "p_pair": None,
                        "pred_pair": None,
                        "p_aux_protein": None,
                        "n_inter_contacts": None,
                        "_index": int(index),
                    }
                )
            else:
                self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def _features(self, identifier: str, sequence: str):
        cached = self._feature_cache.get(identifier)
        if cached is not None:
            self._feature_cache.move_to_end(identifier)
            return cached[0]

        path = self.embedding_dir / f"{identifier}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing embedding: {path}")
        features = build_residue_features(sequence, load_plm_embedding(path))
        size = sum(value.numel() * value.element_size() for value in features.values())
        # Oversized entries and a zero budget bypass the cache entirely.
        if 0 < size <= self._feature_cache_limit_bytes:
            while self._feature_cache_bytes + size > self._feature_cache_limit_bytes:
                _, (_, evicted_bytes) = self._feature_cache.popitem(last=False)
                self._feature_cache_bytes -= evicted_bytes
            self._feature_cache[identifier] = (features, size)
            self._feature_cache_bytes += size
        return features

    def __getitem__(self, index: int):
        item = self.items[index]
        return (
            self._features(item["protein_1"], item["sequence_1"]),
            self._features(item["protein_2"], item["sequence_2"]),
            {
                "index": item["index"],
                "pair": item["pair"],
                "label": item["label"],
                "combined_length": item["combined_length"],
            },
        )


def inference_collate_fn(batch):
    proteins_1, proteins_2, metadata = zip(*batch)
    return (
        collate_concatenated_pairs(list(proteins_1), list(proteins_2)),
        list(metadata),
    )


def merge_prediction_rows(rank_rows: list[list[dict[str, Any]]]):
    by_index: dict[int, dict[str, Any]] = {}
    for rows in rank_rows:
        for row in rows:
            by_index[int(row["_index"])] = row
    merged = [by_index[index] for index in sorted(by_index)]
    for row in merged:
        row.pop("_index", None)
    return merged


def gather_prediction_rows(local_rows: list[dict[str, Any]]):
    return merge_prediction_rows(DIST_WRAPPER.all_gather_object(local_rows))


def pair_contact_summary(
    pair_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    contact_bins: int,
    score_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    probability = pair_contact_probability(pair_logits, contact_bins)
    pooled = max_contact_probability(
        probability,
        pair_mask,
    )
    contact_count = (
        (probability >= float(score_threshold)) & pair_mask.to(dtype=torch.bool)
    ).sum(dim=(-2, -1))
    return pooled, contact_count


def evaluate_ppi(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    precision: str,
    score_threshold: float,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    progress = tqdm(
        loader,
        desc=f"Inference rank {DIST_WRAPPER.rank}",
        disable=DIST_WRAPPER.world_size > 1 and DIST_WRAPPER.rank != 0,
    )
    with torch.inference_mode():
        for residue_batch, metadata in progress:
            residue_batch = to_device(residue_batch, device)
            with autocast_context(device, precision):
                output = model(residue_batch)
            pair_probability, contact_count = pair_contact_summary(
                output.pair_logits,
                output.pair_mask,
                contact_bins=model.contact_bins,
                score_threshold=score_threshold,
            )
            auxiliary = torch.sigmoid(output.aux_protein_logit.float())
            for index, meta in enumerate(metadata):
                probability = float(pair_probability[index].item())
                rows.append(
                    {
                        "Protein pairs": meta["pair"],
                        "Category": bool(meta["label"]),
                        "combined_length": int(meta["combined_length"]),
                        "status": "scored",
                        "p_pair": probability,
                        "pred_pair": probability >= float(score_threshold),
                        "p_aux_protein": float(auxiliary[index].item()),
                        "n_inter_contacts": int(contact_count[index].item()),
                        "_index": int(meta["index"]),
                    }
                )
    return rows


def _init_distributed(use_cuda: bool) -> None:
    if DIST_WRAPPER.world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=resolve_dist_backend(use_cuda),
            timeout=datetime.timedelta(
                seconds=int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
            ),
        )


def write_metrics(
    path: str | Path,
    rows: list[dict[str, Any]],
    configured_weight: float,
) -> None:
    scored = [row for row in rows if row["status"] == "scored"]
    records = []
    weights = [1.0]
    if float(configured_weight) != 1.0:
        weights.append(float(configured_weight))
    for head, score_key in (("pair", "p_pair"), ("auxiliary", "p_aux_protein")):
        for weight in weights:
            metrics = metrics_from_rows(
                scored,
                score_key,
                positive_weight=weight,
            )
            records.append(
                {
                    "head": head,
                    "positive_weight": weight,
                    "AUROC": metrics["auroc"],
                    "AUPRC": metrics["auprc"],
                    "N": metrics["n"],
                    "N_pos": metrics["n_pos"],
                    "N_neg": metrics["n_neg"],
                    "coverage": len(scored) / max(1, len(rows)),
                }
            )
    pd.DataFrame(records).to_csv(path, index=False)


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def main(args: DictConfig) -> None:
    if not args.ckpt_dir:
        raise ValueError("ckpt_dir is required")
    use_cuda = torch.cuda.is_available()
    device = (
        resolve_cuda_device(DIST_WRAPPER.local_rank, torch.cuda.device_count())
        if use_cuda
        else torch.device("cpu")
    )
    if use_cuda:
        torch.cuda.set_device(device)
    _init_distributed(use_cuda)
    seed_everything(int(args.seed), bool(args.deterministic))

    dataset = InferencePairDataset(
        args.data.test_set_tsv,
        args.data.fasta_path,
        args.data.embedding_dir,
        maximum_length=int(args.data.maximum_length),
        embedding_cache_mb=float(cfg_get(args, "data.embedding_cache_mb", 256)),
    )
    indices = list(
        range(DIST_WRAPPER.rank, len(dataset), DIST_WRAPPER.world_size)
    )
    loader = get_dataloader(
        Subset(dataset, indices),
        collate_fn=inference_collate_fn,
        batch_size=int(args.data.batch_size),
        shuffle=False,
        num_workers=int(args.data.num_workers),
        pin_memory=bool(args.data.pin_memory),
    )
    model = create_model(args).to(device)
    load_checkpoint(args.ckpt_dir, model, map_location=device,
                    weights=str(cfg_get(args, "runtime.checkpoint_weights", "ema")))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    rows = evaluate_ppi(
        model,
        loader,
        device,
        precision=str(cfg_get(args, "runtime.precision", "bfloat16")),
        score_threshold=float(args.data.ppi_score_threshold),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else None
    )
    if DIST_WRAPPER.rank == 0:
        rows.extend(dataset.unsupported_rows)
    rows = gather_prediction_rows(rows)

    if DIST_WRAPPER.rank == 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(cfg_get(args, "output_dir", Path(args.logging_dir) / f"{str(args.task_prefix).upper()}_{timestamp}"))
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(args, output_dir / "config.yaml")
        pd.DataFrame(rows, columns=PREDICTION_COLUMNS).to_csv(
            output_dir / "predictions.tsv",
            sep="\t",
            index=False,
        )
        write_metrics(
            output_dir / "metrics.csv",
            rows,
            float(cfg_get(args, "metrics.positive_weight", 1.0)),
        )
        supported = sum(row["status"] == "scored" for row in rows)
        runtime = {
            "device": str(device),
            "precision": normalize_precision(str(cfg_get(args, "runtime.precision", "bfloat16")), device),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "n_total": len(rows),
            "n_scored": supported,
            "coverage": supported / max(1, len(rows)),
            "model_latency_seconds_total": inference_seconds,
            "model_latency_seconds_mean": inference_seconds / max(1, supported),
            "peak_allocated_cuda_bytes": peak_allocated,
            "peak_reserved_cuda_bytes": peak_reserved,
            "embedding_preparation_latency_seconds": None,
            "embedding_preparation_note": (
                "Embeddings are precomputed by prepare_inference_embeddings.py "
                "and must be timed separately."
            ),
        }
        with (output_dir / "runtime.json").open("w", encoding="utf-8") as handle:
            json.dump(runtime, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            f"Scored {supported}/{len(rows)} pairs; "
            f"{len(rows) - supported} exceeded the dense length contract"
        )
        print(f"Wrote {output_dir}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
