import datetime
import os
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
from src.data.labels import contact_bin_count, inter_chain_pair_mask
from src.model.surpass import ResOnly
from src.utils.ppi_windows import plan_ppi_windows
from src.train import (
    _binary_classification_metrics,
    _cfg_get,
    build_model_kwargs,
    log_info,
    resolve_cuda_device,
    resolve_dist_backend,
    to_device,
)
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything


PREDICTION_COLUMNS = [
    "Protein pairs",
    "Category",
    "p_protein",
    "pred_protein",
    "p_pair",
    "pred_pair",
    "n_inter_contacts",
]

_RESIDUE_FEATURE_KEYS = (
    "plm_emb",
    "residue_type",
    "residue_index",
    "chain_index",
    "mask",
    "residue_position",
)


def parse_protein_pair(pair: str) -> tuple[str, str]:
    parts = str(pair).strip().split("_")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Protein pair must be two IDs linked by '_': {pair!r}")
    return parts[0], parts[1]


def parse_ppi_label(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(int(value))
        raise ValueError(f"Unsupported numeric PPI label: {value!r}")
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
    raise ValueError("Test set TSV is missing a 'Category' column.")


def load_fasta_sequences(fasta_path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    protein_id: str | None = None
    chunks: list[str] = []

    def _flush() -> None:
        nonlocal protein_id
        if protein_id is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence:
            raise ValueError(f"FASTA record {protein_id} has an empty sequence.")
        records[protein_id] = sequence

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                header = line[1:].strip()
                if not header:
                    raise ValueError("FASTA header is missing a sequence ID.")
                protein_id = header.split()[0]
                chunks = []
                continue
            if protein_id is None:
                raise ValueError("FASTA sequence found before a header.")
            chunks.append(line)
    _flush()
    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}.")
    return records


def load_plm_embedding(path: str | Path) -> torch.Tensor:
    payload = torch.load(path, weights_only=False)
    if isinstance(payload, torch.Tensor):
        return payload.to(dtype=torch.float32)
    if isinstance(payload, dict) and "plm_emb" in payload:
        return torch.as_tensor(payload["plm_emb"], dtype=torch.float32)
    raise ValueError(f"Embedding file {path} must be a tensor or a dict with plm_emb.")


def encode_residue_types(sequence: str) -> torch.Tensor:
    return torch.tensor(
        [restype_order.get(residue, unk_restype_index) for residue in sequence],
        dtype=torch.long,
    )


def build_residue_features(sequence: str, plm_emb: torch.Tensor) -> dict[str, torch.Tensor]:
    sequence = sequence.upper()
    embedding = torch.as_tensor(plm_emb, dtype=torch.float32)
    if embedding.ndim != 2:
        raise ValueError(f"plm_emb must have shape [L, D], got {tuple(embedding.shape)}")
    if int(embedding.shape[0]) != len(sequence):
        raise ValueError(
            f"embedding length {int(embedding.shape[0])} does not match "
            f"sequence length {len(sequence)}"
        )
    length = len(sequence)
    return {
        "plm_emb": embedding,
        "residue_type": encode_residue_types(sequence),
        "residue_index": torch.arange(length, dtype=torch.long),
        "chain_index": torch.zeros(length, dtype=torch.long),
        "mask": torch.ones(length, dtype=torch.bool),
        "residue_position": torch.zeros(length, 3, dtype=torch.float32),
    }


def inference_collate_fn(batch):
    proteins, peptides, metas = zip(*batch)
    residue_batch = collate_concatenated_pairs(list(proteins), list(peptides))
    return residue_batch, list(metas)


class InferencePairDataset(Dataset):
    def __init__(
        self,
        test_set_tsv: str | Path,
        fasta_path: str | Path,
        embedding_dir: str | Path,
    ):
        self.test_set_tsv = Path(test_set_tsv)
        self.fasta_path = Path(fasta_path)
        self.embedding_dir = Path(embedding_dir)
        metadata = pd.read_csv(self.test_set_tsv, sep="\t")
        if "Protein pairs" not in metadata.columns:
            raise ValueError("Test set TSV is missing the 'Protein pairs' column.")
        label_column = resolve_label_column(metadata.columns)
        sequences = load_fasta_sequences(self.fasta_path)

        self._feature_cache: dict[str, dict[str, torch.Tensor]] = {}
        self.items: list[dict[str, Any]] = []
        for _, row in metadata.iterrows():
            pair = str(row["Protein pairs"]).strip()
            protein_1, protein_2 = parse_protein_pair(pair)
            if protein_1 not in sequences:
                raise KeyError(f"Missing FASTA sequence for {protein_1}")
            if protein_2 not in sequences:
                raise KeyError(f"Missing FASTA sequence for {protein_2}")
            sequence_1 = sequences[protein_1]
            sequence_2 = sequences[protein_2]
            self.items.append(
                {
                    "pair": pair,
                    "protein_1": protein_1,
                    "protein_2": protein_2,
                    "label": parse_ppi_label(row[label_column]),
                    "sequence_1": sequence_1,
                    "sequence_2": sequence_2,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def _load_features(self, protein_id: str, sequence: str) -> dict[str, torch.Tensor]:
        cached = self._feature_cache.get(protein_id)
        if cached is not None:
            return cached
        embedding_path = self.embedding_dir / f"{protein_id}.pt"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {embedding_path}")
        features = build_residue_features(sequence, load_plm_embedding(embedding_path))
        self._feature_cache[protein_id] = features
        return features

    def __getitem__(self, idx: int):
        item = self.items[idx]
        p1_features = self._load_features(item["protein_1"], item["sequence_1"])
        p2_features = self._load_features(item["protein_2"], item["sequence_2"])
        meta = {
            "pair": item["pair"],
            "protein_1": item["protein_1"],
            "protein_2": item["protein_2"],
            "label": item["label"],
            "index": idx,
        }
        return p1_features, p2_features, meta


def merge_rank_prediction_rows(rank_rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for rows in rank_rows:
        for row in rows:
            index = row.get("_index")
            if index is not None:
                index = int(index)
                if index in seen_indices:
                    continue
                seen_indices.add(index)
            merged.append(dict(row))
    if seen_indices and all(row.get("_index") is not None for row in merged):
        merged.sort(key=lambda row: int(row["_index"]))
    for row in merged:
        row.pop("_index", None)
    return merged


def metrics_from_prediction_rows(
    rows: list[dict[str, Any]],
    score_key: str,
    positive_weight: float = 1.0,
) -> dict[str, float]:
    if not rows:
        all_scores = torch.empty(0, dtype=torch.float32)
        all_targets = torch.empty(0, dtype=torch.bool)
    else:
        all_scores = torch.tensor([float(row[score_key]) for row in rows], dtype=torch.float32)
        all_targets = torch.tensor([bool(row["Category"]) for row in rows], dtype=torch.bool)
    classification = _binary_classification_metrics(
        all_scores, all_targets, positive_weight=positive_weight
    )
    return {
        "auroc": classification["auroc"],
        "auprc": classification["auprc"],
        "n": int(all_targets.numel()),
        "n_pos": int(all_targets.sum().item()) if all_targets.numel() else 0,
        "n_neg": int((~all_targets).sum().item()) if all_targets.numel() else 0,
    }


def protein_metric_weights(configured_weight: float) -> list[float]:
    weights = [1.0]
    configured_weight = float(configured_weight)
    if configured_weight != 1.0:
        weights.append(configured_weight)
    return weights


def gather_rank_prediction_rows(local_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gathered = DIST_WRAPPER.all_gather_object(local_rows)
    return merge_rank_prediction_rows(gathered)


def shard_dataset(dataset: Dataset) -> Dataset:
    if DIST_WRAPPER.world_size <= 1:
        return dataset
    indices = list(range(DIST_WRAPPER.rank, len(dataset), DIST_WRAPPER.world_size))
    return Subset(dataset, indices)


def init_distributed(use_cuda: bool) -> None:
    if DIST_WRAPPER.world_size > 1 and not dist.is_initialized():
        timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
        dist.init_process_group(
            backend=resolve_dist_backend(use_cuda),
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )


def write_predictions(path: str | Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    frame.to_csv(path, sep="\t", index=False)


def write_metrics(
    path: str | Path,
    metrics_by_head: dict[str, dict[float, dict[str, Any]]],
) -> None:
    frame = pd.DataFrame(
        [
            {
                "head": head,
                "positive_weight": weight,
                "AUROC": metrics["auroc"],
                "AUPRC": metrics["auprc"],
                "N": metrics["n"],
                "N_pos": metrics["n_pos"],
                "N_neg": metrics["n_neg"],
            }
            for head, metrics_by_weight in metrics_by_head.items()
            for weight, metrics in metrics_by_weight.items()
        ]
    )
    frame.to_csv(path, index=False)


def _prediction_row(
    meta: dict[str, Any],
    protein_probability: float,
    pair_probability: float,
    n_inter_contacts: int,
    ppi_score_threshold: float,
) -> dict[str, Any]:
    row = {
        "Protein pairs": meta["pair"],
        "Category": bool(meta["label"]),
        "p_protein": float(protein_probability),
        "pred_protein": float(protein_probability) >= float(ppi_score_threshold),
        "p_pair": float(pair_probability),
        "pred_pair": int(n_inter_contacts) >= 1,
        "n_inter_contacts": int(n_inter_contacts),
    }
    if "index" in meta:
        row["_index"] = meta["index"]
    return row


def _protein_probability(output, index: int = 0) -> float:
    logit = output.protein_logit.detach().float().reshape(-1)[index]
    return float(torch.sigmoid(logit).item())


def resolve_inference_dtype(name: str | None) -> torch.dtype | None:
    """Autocast dtype for the inference forward. ``None`` keeps float32."""
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in {"float32", "fp32", "float"}:
        return None
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(
        f"Unsupported inference_dtype {name!r}. Use bfloat16, float16, or float32."
    )


def aggregate_window_scores(
    protein_probabilities: list[float],
    pair_probabilities: list[float],
    contact_counts: list[int],
) -> tuple[float, float, int]:
    """Max probabilities and summed contacts across tiled windows."""
    if not protein_probabilities or not pair_probabilities or not contact_counts:
        raise ValueError("window scores require at least one window.")
    return (
        max(float(value) for value in protein_probabilities),
        max(float(value) for value in pair_probabilities),
        int(sum(int(value) for value in contact_counts)),
    )


def slice_residue_features(
    residue_batch: dict[str, torch.Tensor],
    index: int,
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    features = {
        key: residue_batch[key][index, start:end]
        for key in _RESIDUE_FEATURE_KEYS
    }
    features["chain_index"] = torch.zeros(
        end - start,
        dtype=torch.long,
        device=features["mask"].device,
    )
    return features


def window_residue_batch(
    residue_batch: dict[str, torch.Tensor],
    index: int,
    p1_length: int,
    span_1: tuple[int, int],
    span_2: tuple[int, int],
) -> dict[str, torch.Tensor]:
    start_1, end_1 = span_1
    start_2, end_2 = span_2
    protein = slice_residue_features(residue_batch, index, start_1, end_1)
    peptide = slice_residue_features(
        residue_batch,
        index,
        int(p1_length) + start_2,
        int(p1_length) + end_2,
    )
    return collate_concatenated_pairs([protein], [peptide])


def _model_forward(model, residue_batch, device: torch.device, autocast_dtype):
    enabled = autocast_dtype is not None and device.type == "cuda"
    with torch.autocast(
        device_type="cuda",
        dtype=autocast_dtype or torch.bfloat16,
        enabled=enabled,
    ):
        return model(residue_batch)


def pair_contact_summary(
    pair_logits: torch.Tensor,
    residue_mask: torch.Tensor,
    p1_length: torch.Tensor,
    p2_length: torch.Tensor,
    contact_bins: int,
    score_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max inter-chain contact probability and the number of such contacts.

    A contact is a directed inter-chain pair whose predicted distance falls in
    the bins closer than the contact threshold with probability at least
    ``score_threshold``. ``p_pair`` is the maximum of those probabilities.
    """
    contact_probability = torch.softmax(pair_logits.float(), dim=-1)[..., : int(contact_bins)].sum(dim=-1)
    valid = residue_mask.to(dtype=torch.bool)
    inter_mask = inter_chain_pair_mask(
        p1_length,
        p2_length,
        device=contact_probability.device,
        total_length=int(contact_probability.shape[-1]),
    )
    pair_valid = inter_mask & valid[:, :, None] & valid[:, None, :]
    contact_probability = contact_probability.masked_fill(~pair_valid, 0.0)
    pair_probability = contact_probability.amax(dim=(-2, -1))
    n_inter_contacts = (contact_probability >= float(score_threshold)).sum(dim=(-2, -1))
    return pair_probability, n_inter_contacts


def evaluate_ppi(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    ppi_score_threshold: float,
    contact_bins: int,
    max_full_length: int | None,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    n_windowed = 0
    max_length = 0
    desc = (
        "Inference"
        if DIST_WRAPPER.world_size <= 1
        else f"Inference [{DIST_WRAPPER.rank}/{DIST_WRAPPER.world_size}]"
    )
    progress = tqdm(
        loader,
        desc=desc,
        total=len(loader),
        leave=True,
        position=DIST_WRAPPER.local_rank if DIST_WRAPPER.world_size > 1 else 0,
    )

    with progress, torch.no_grad():
        for residue_batch, metas in progress:
            if metas:
                pair_label = metas[0]["pair"]
                if len(metas) > 1:
                    pair_label = f"{pair_label} +{len(metas) - 1}"
                progress.set_postfix(pair=pair_label, refresh=False)
            for index, meta in enumerate(metas):
                p1_length = int(residue_batch["p1_length"][index])
                p2_length = int(residue_batch["p2_length"][index])
                full_length = p1_length + p2_length
                max_length = max(max_length, full_length)
                windows = plan_ppi_windows(p1_length, p2_length, max_full_length)
                if len(windows) > 1:
                    n_windowed += 1
                if len(windows) == 1 and len(metas) == 1:
                    prepared = [residue_batch]
                else:
                    prepared = [
                        window_residue_batch(
                            residue_batch,
                            index,
                            p1_length,
                            span_1,
                            span_2,
                        )
                        for span_1, span_2 in windows
                    ]
                protein_probabilities: list[float] = []
                pair_probabilities: list[float] = []
                contact_counts: list[int] = []
                for window_batch in prepared:
                    window_batch = to_device(window_batch, device)
                    output = _model_forward(
                        model,
                        window_batch,
                        device,
                        autocast_dtype,
                    )
                    pair_probability, n_inter_contacts = pair_contact_summary(
                        output.pair_logits,
                        window_batch["mask"],
                        window_batch["p1_length"],
                        window_batch["p2_length"],
                        contact_bins,
                        ppi_score_threshold,
                    )
                    protein_probabilities.append(_protein_probability(output, 0))
                    pair_probabilities.append(float(pair_probability[0].item()))
                    contact_counts.append(int(n_inter_contacts[0].item()))
                    del output, window_batch
                protein_probability, pair_probability_value, n_contacts = (
                    aggregate_window_scores(
                        protein_probabilities,
                        pair_probabilities,
                        contact_counts,
                    )
                )
                rows.append(
                    _prediction_row(
                        meta,
                        protein_probability,
                        pair_probability_value,
                        n_contacts,
                        ppi_score_threshold,
                    )
                )

    return rows, {
        "n_windowed": n_windowed,
        "max_length": max_length,
        "n_dropped": 0,
    }


def gather_inference_stats(stats: dict[str, int]) -> dict[str, int]:
    gathered = DIST_WRAPPER.all_gather_object(stats)
    return {
        "n_windowed": sum(int(item["n_windowed"]) for item in gathered),
        "max_length": max((int(item["max_length"]) for item in gathered), default=0),
        "n_dropped": 0,
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def main(args: DictConfig):
    use_cuda = torch.cuda.device_count() > 0
    if use_cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = resolve_cuda_device(DIST_WRAPPER.local_rank, torch.cuda.device_count())
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    init_distributed(use_cuda)
    seed_everything(seed=args.seed, deterministic=args.deterministic)

    logging_dir = os.path.join(
        args.logging_dir,
        f"{str(args.task_prefix).upper()}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    logging_dir = DIST_WRAPPER.all_gather_object(logging_dir)[0]
    if DIST_WRAPPER.rank == 0:
        os.makedirs(args.logging_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        with open(f"{logging_dir}/config.yaml", "w", encoding="utf-8") as handle:
            OmegaConf.save(args, handle)

    if args.ckpt_dir is None:
        raise ValueError("ckpt_dir is required for inference.")

    embedding_dir = _cfg_get(args, "data.embedding_dir", default=None)
    if embedding_dir is None:
        embedding_dir = str(Path(args.data.root_dir) / "embeddings")

    max_full_length = _cfg_get(args, "data.max_full_length", default=2048)
    if max_full_length is not None:
        max_full_length = int(max_full_length)
        if max_full_length <= 0:
            raise ValueError(
                f"max_full_length must be positive, got {max_full_length}."
            )
    autocast_dtype = resolve_inference_dtype(
        _cfg_get(args, "data.inference_dtype", default="float32")
    )
    dataset = InferencePairDataset(
        test_set_tsv=args.data.test_set_tsv,
        fasta_path=args.data.fasta_path,
        embedding_dir=embedding_dir,
    )
    shard = shard_dataset(dataset)
    loader = get_dataloader(
        shard,
        collate_fn=inference_collate_fn,
        batch_size=int(args.data.batch_size),
        shuffle=False,
        num_workers=int(args.data.num_workers),
        pin_memory=bool(args.data.pin_memory),
    )
    budget_label = "unbounded" if max_full_length is None else str(max_full_length)
    dtype_label = "float32" if autocast_dtype is None else str(autocast_dtype).replace("torch.", "")
    log_info(
        f"Loaded {len(dataset)} test pairs; "
        f"single-forward budget {budget_label}; dtype {dtype_label}"
    )
    if DIST_WRAPPER.world_size > 1:
        print(
            f"[rank {DIST_WRAPPER.rank}/{DIST_WRAPPER.world_size}] "
            f"device={device} shard_pairs={len(shard)}/{len(dataset)}"
        )

    model_kwargs = build_model_kwargs(args, checkpoint_default=False)
    model = ResOnly(**model_kwargs).to(device)
    checkpoint = torch.load(args.ckpt_dir, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    del checkpoint

    configured_weight = float(_cfg_get(args, "metrics.positive_weight", default=1.0))
    contact_bins = contact_bin_count(
        float(args.data.contact_threshold),
        float(args.data.distance_bin_start),
        float(args.data.distance_bin_end),
        int(args.data.distance_bin_count),
    )
    rows, stats = evaluate_ppi(
        model=model,
        loader=loader,
        device=device,
        ppi_score_threshold=float(args.data.ppi_score_threshold),
        contact_bins=contact_bins,
        max_full_length=max_full_length,
        autocast_dtype=autocast_dtype,
    )
    rows = gather_rank_prediction_rows(rows)
    stats = gather_inference_stats(stats)

    if DIST_WRAPPER.rank == 0:
        weights = protein_metric_weights(configured_weight)
        metrics_by_head = {
            head: {
                weight: metrics_from_prediction_rows(
                    rows,
                    score_key,
                    positive_weight=weight,
                )
                for weight in weights
            }
            for head, score_key in (("pair", "p_pair"), ("protein", "p_protein"))
        }
        predictions_path = os.path.join(logging_dir, "predictions.tsv")
        metrics_path = os.path.join(logging_dir, "metrics.csv")
        write_predictions(predictions_path, rows)
        write_metrics(metrics_path, metrics_by_head)
        for head, metrics_by_weight in metrics_by_head.items():
            for weight, head_metrics in metrics_by_weight.items():
                print(
                    f"[test-metrics] head={head} "
                    f"positive_weight={weight:g} "
                    f"auroc={head_metrics['auroc']:.4f} "
                    f"auprc={head_metrics['auprc']:.4f}"
                )
        unweighted = metrics_by_head["pair"][1.0]
        print(
            f"[test-metrics] "
            f"n={unweighted['n']} "
            f"n_pos={unweighted['n_pos']} "
            f"n_neg={unweighted['n_neg']} "
            f"n_dropped={stats['n_dropped']} "
            f"n_windowed={stats['n_windowed']} "
            f"max_length={stats['max_length']}"
        )
        print(f"Wrote {predictions_path}")
        print(f"Wrote {metrics_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
