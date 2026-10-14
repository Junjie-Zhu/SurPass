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
from src.model.loss import distance_bin_centers
from src.model.surpass import ResOnly
from src.train import (
    _binary_classification_metrics,
    _cfg_get,
    build_model_kwargs,
    contact_bin_count,
    log_info,
    pair_bind_scores,
    resolve_cuda_device,
    resolve_dist_backend,
    to_device,
)
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything
from src.utils.ppi_windows import plan_ppi_windows


PPI_HEADS = ("protein", "residue", "pair")
PREDICTION_COLUMNS = [
    "Protein pairs",
    "Category",
    "p_protein",
    "pred_protein",
    "p_residue",
    "pred_residue",
    "n_pos_residues",
    "p_pair",
    "pred_pair",
    "n_inter_contacts",
    "n_forwards",
]


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


def metrics_for_heads(
    rows: list[dict[str, Any]],
    positive_weight: float = 1.0,
) -> dict[str, dict[str, float]]:
    return {
        head: metrics_from_prediction_rows(
            rows,
            f"p_{head}",
            positive_weight=positive_weight,
        )
        for head in PPI_HEADS
    }


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


def distance_map_from_logits(
    pair_logits: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    bins = pair_logits.argmax(dim=-1)
    centers = centers.to(device=bins.device)
    return centers[bins.long()]


def write_pair_maps(
    directory: str | Path,
    pair_name: str,
    residue_probability: torch.Tensor,
    distance: torch.Tensor,
    p1_length: int,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = str(pair_name).replace(os.sep, "_").replace("/", "_")
    path = directory / f"{safe_name}.pt"
    torch.save(
        {
            "residue_probability": residue_probability.detach().to(device="cpu", dtype=torch.float32),
            "distance": distance.detach().to(device="cpu", dtype=torch.float32),
            "p1_length": int(p1_length),
        },
        path,
    )
    return path


def write_metrics(path: str | Path, metrics_by_head: dict[str, dict[str, Any]]) -> None:
    frame = pd.DataFrame(
        [
            {
                "head": head,
                "AUROC": metrics_by_head[head]["auroc"],
                "AUPRC": metrics_by_head[head]["auprc"],
                "N": metrics_by_head[head]["n"],
                "N_pos": metrics_by_head[head]["n_pos"],
                "N_neg": metrics_by_head[head]["n_neg"],
            }
            for head in PPI_HEADS
        ]
    )
    frame.to_csv(path, index=False)


def _slice_features(
    residue_batch: dict[str, torch.Tensor],
    index: int,
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    for key, value in residue_batch.items():
        if key in {"p1_length", "p2_length"} or not isinstance(value, torch.Tensor):
            continue
        features[key] = value[index, start:end]
    return features


def _window_residue_batch(
    residue_batch: dict[str, torch.Tensor],
    index: int,
    span_1: tuple[int, int],
    span_2: tuple[int, int],
) -> dict[str, torch.Tensor]:
    n1 = int(residue_batch["p1_length"][index].item())
    start_1, end_1 = span_1
    start_2, end_2 = span_2
    protein = _slice_features(residue_batch, index, start_1, end_1)
    peptide = _slice_features(residue_batch, index, n1 + start_2, n1 + end_2)
    return collate_concatenated_pairs([protein], [peptide])


def _batch_exceeds_budget(
    residue_batch: dict[str, torch.Tensor],
    max_full_length: int | None,
) -> bool:
    if max_full_length is None:
        return False
    lengths = residue_batch["p1_length"] + residue_batch["p2_length"]
    return bool((lengths > int(max_full_length)).any().item())


def _prediction_row(
    meta: dict[str, Any],
    scores: dict[str, float | int],
    ppi_score_threshold: float,
) -> dict[str, Any]:
    row = {
        "Protein pairs": meta["pair"],
        "Category": bool(meta["label"]),
        "p_protein": float(scores["p_protein"]),
        "pred_protein": float(scores["p_protein"]) >= float(ppi_score_threshold),
        "p_residue": float(scores["p_residue"]),
        "pred_residue": int(scores["n_pos_residues"]) >= 1,
        "n_pos_residues": int(scores["n_pos_residues"]),
        "p_pair": float(scores["p_pair"]),
        "pred_pair": int(scores["n_inter_contacts"]) >= 1,
        "n_inter_contacts": int(scores["n_inter_contacts"]),
        "n_forwards": int(scores["n_forwards"]),
    }
    if "index" in meta:
        row["_index"] = meta["index"]
    return row


def _head_scores_from_output(
    output,
    residue_batch: dict[str, torch.Tensor],
    index: int,
    contact_bins: int,
    ppi_score_threshold: float,
) -> dict[str, float | int]:
    p1_length = int(residue_batch["p1_length"][index].item())
    p2_length = int(residue_batch["p2_length"][index].item())
    length = p1_length + p2_length
    residue_prob = torch.sigmoid(output.residue_logits[index, :length])
    residue_mask = residue_batch["mask"][index, :length].to(dtype=torch.bool)
    residue_prob = residue_prob.masked_fill(~residue_mask, 0.0)
    if bool(residue_mask.any().item()):
        p_residue = float(residue_prob[residue_mask].max().item())
    else:
        p_residue = 0.0
    pair_score, n_contacts = pair_bind_scores(
        output.pair_logits[index : index + 1],
        output.pair_mask[index : index + 1],
        p1_length=residue_batch["p1_length"][index : index + 1],
        p2_length=residue_batch["p2_length"][index : index + 1],
        contact_bins=contact_bins,
        ppi_score_threshold=ppi_score_threshold,
    )
    return {
        "p_protein": float(torch.sigmoid(output.protein_logit.reshape(-1))[index].item()),
        "p_residue": p_residue,
        "n_pos_residues": int((residue_prob.ge(float(ppi_score_threshold)) & residue_mask).sum().item()),
        "p_pair": float(pair_score.reshape(-1)[0].item()),
        "n_inter_contacts": int(n_contacts.reshape(-1)[0].item()),
        "n_forwards": 1,
    }


def _write_sample_maps(
    map_dir: str | Path,
    pair_name: str,
    output,
    residue_batch: dict[str, torch.Tensor],
    index: int,
    centers: torch.Tensor,
) -> None:
    p1_length = int(residue_batch["p1_length"][index].item())
    p2_length = int(residue_batch["p2_length"][index].item())
    length = p1_length + p2_length
    residue_mask = residue_batch["mask"][index, :length].to(dtype=torch.bool)
    pair_valid = output.pair_mask[index, :length, :length].to(dtype=torch.bool)
    residue_probability = torch.sigmoid(output.residue_logits[index, :length])
    distance = distance_map_from_logits(output.pair_logits[index, :length, :length], centers)
    write_pair_maps(
        map_dir,
        pair_name,
        residue_probability.masked_fill(~residue_mask, float("nan")),
        distance.masked_fill(~pair_valid, float("nan")),
        p1_length,
    )


def _score_pair_windows(
    model: torch.nn.Module,
    residue_batch: dict[str, torch.Tensor],
    index: int,
    max_full_length: int | None,
    contact_bins: int,
    ppi_score_threshold: float,
    map_dir: str | Path | None,
    pair_name: str,
    centers: torch.Tensor,
) -> dict[str, float | int]:
    p1_length = int(residue_batch["p1_length"][index].item())
    p2_length = int(residue_batch["p2_length"][index].item())
    windows = plan_ppi_windows(p1_length, p2_length, max_full_length)
    device = residue_batch["mask"].device
    p1_prob = torch.zeros(p1_length, device=device)
    p2_prob = torch.zeros(p2_length, device=device)
    p1_mask = residue_batch["mask"][index, :p1_length].to(dtype=torch.bool)
    p2_mask = residue_batch["mask"][index, p1_length : p1_length + p2_length].to(dtype=torch.bool)
    protein_score = 0.0
    pair_score = 0.0
    n_contacts = 0
    threshold = float(ppi_score_threshold)

    for span_1, span_2 in windows:
        window_batch = _window_residue_batch(residue_batch, index, span_1, span_2)
        output = model(window_batch)
        start_1, end_1 = span_1
        start_2, end_2 = span_2
        width_1 = end_1 - start_1
        width_2 = end_2 - start_2
        probability = torch.sigmoid(output.residue_logits[0])
        window_mask = window_batch["mask"][0].to(dtype=torch.bool)
        probability = probability.masked_fill(~window_mask, 0.0)
        p1_prob[start_1:end_1] = torch.maximum(p1_prob[start_1:end_1], probability[:width_1])
        p2_prob[start_2:end_2] = torch.maximum(
            p2_prob[start_2:end_2],
            probability[width_1 : width_1 + width_2],
        )
        protein_score = max(
            protein_score,
            float(torch.sigmoid(output.protein_logit.reshape(-1))[0].item()),
        )
        window_pair, window_contacts = pair_bind_scores(
            output.pair_logits,
            output.pair_mask,
            p1_length=window_batch["p1_length"],
            p2_length=window_batch["p2_length"],
            contact_bins=contact_bins,
            ppi_score_threshold=threshold,
        )
        pair_score = max(pair_score, float(window_pair.reshape(-1)[0].item()))
        n_contacts += int(window_contacts.reshape(-1)[0].item())
        if len(windows) == 1 and map_dir is not None:
            _write_sample_maps(map_dir, pair_name, output, window_batch, 0, centers)

    residue_values = []
    if bool(p1_mask.any().item()):
        residue_values.append(float(p1_prob[p1_mask].max().item()))
    if bool(p2_mask.any().item()):
        residue_values.append(float(p2_prob[p2_mask].max().item()))
    n_pos = int((p1_prob.ge(threshold) & p1_mask).sum().item())
    n_pos += int((p2_prob.ge(threshold) & p2_mask).sum().item())
    return {
        "p_protein": protein_score,
        "p_residue": max(residue_values) if residue_values else 0.0,
        "n_pos_residues": n_pos,
        "p_pair": pair_score,
        "n_inter_contacts": n_contacts,
        "n_forwards": len(windows),
    }


def evaluate_ppi(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    contact_threshold: float,
    distance_bin_start: float,
    distance_bin_end: float,
    distance_bin_count: int,
    ppi_score_threshold: float,
    positive_weight: float = 1.0,
    map_dir: str | Path | None = None,
    max_full_length: int | None = 1024,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    model.eval()
    contact_bins = contact_bin_count(
        contact_threshold,
        distance_bin_start,
        distance_bin_end,
        distance_bin_count,
    )
    centers = distance_bin_centers(
        distance_bin_start,
        distance_bin_end,
        distance_bin_count,
    )
    rows: list[dict[str, Any]] = []
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
            residue_batch = to_device(residue_batch, device)
            if _batch_exceeds_budget(residue_batch, max_full_length):
                for index, meta in enumerate(metas):
                    scores = _score_pair_windows(
                        model,
                        residue_batch,
                        index,
                        max_full_length,
                        contact_bins,
                        ppi_score_threshold,
                        map_dir,
                        meta["pair"],
                        centers,
                    )
                    rows.append(_prediction_row(meta, scores, ppi_score_threshold))
                continue

            output = model(residue_batch)
            for index, meta in enumerate(metas):
                scores = _head_scores_from_output(
                    output,
                    residue_batch,
                    index,
                    contact_bins,
                    ppi_score_threshold,
                )
                rows.append(_prediction_row(meta, scores, ppi_score_threshold))
                if map_dir is not None:
                    _write_sample_maps(
                        map_dir,
                        meta["pair"],
                        output,
                        residue_batch,
                        index,
                        centers,
                    )

    return rows, metrics_for_heads(rows, positive_weight=positive_weight)


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

    max_full_length = _cfg_get(args, "data.max_full_length", default=1024)
    if max_full_length is not None:
        max_full_length = int(max_full_length)
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
    log_info(
        f"Loaded {len(dataset)} test pairs"
        + (
            f" (complexes longer than {max_full_length} are tiled)"
            if max_full_length is not None
            else ""
        )
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

    positive_weight = float(_cfg_get(args, "metrics.positive_weight", default=1.0))
    rows, _ = evaluate_ppi(
        model=model,
        loader=loader,
        device=device,
        contact_threshold=float(args.data.contact_threshold),
        distance_bin_start=float(args.data.distance_bin_start),
        distance_bin_end=float(args.data.distance_bin_end),
        distance_bin_count=int(args.data.distance_bin_count),
        ppi_score_threshold=float(args.data.ppi_score_threshold),
        positive_weight=positive_weight,
        map_dir=os.path.join(logging_dir, "maps"),
        max_full_length=max_full_length,
    )
    rows = gather_rank_prediction_rows(rows)

    if DIST_WRAPPER.rank == 0:
        metrics = metrics_for_heads(rows, positive_weight=positive_weight)
        predictions_path = os.path.join(logging_dir, "predictions.tsv")
        metrics_path = os.path.join(logging_dir, "metrics.csv")
        write_predictions(predictions_path, rows)
        write_metrics(metrics_path, metrics)
        for head in PPI_HEADS:
            head_metrics = metrics[head]
            print(
                f"[test-metrics] head={head} "
                f"auroc={head_metrics['auroc']:.4f} "
                f"auprc={head_metrics['auprc']:.4f}"
            )
        shared = metrics["protein"]
        n_tiled = sum(int(row["n_forwards"]) > 1 for row in rows)
        print(
            f"[test-metrics] "
            f"n={shared['n']} "
            f"n_pos={shared['n_pos']} "
            f"n_neg={shared['n_neg']} "
            f"n_tiled={n_tiled}"
        )
        print(f"Wrote {predictions_path}")
        print(f"Wrote {metrics_path}")
        print(f"Wrote residue probabilities and distance maps to {os.path.join(logging_dir, 'maps')}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
