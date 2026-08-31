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
from src.data.dataset import get_dataloader, inter_chain_pair_mask
from src.model.surpass import ResOnly
from src.train import (
    _binary_classification_metrics,
    _cfg_get,
    contact_bin_count,
    log_info,
    resolve_cuda_device,
    resolve_dist_backend,
    to_device,
)
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything


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


def pair_bind_scores(
    logits: torch.Tensor,
    pair_mask: torch.Tensor,
    p1_length: int,
    contact_bins: int,
    ppi_score_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    p_contact = torch.softmax(logits, dim=-1)[..., : int(contact_bins)].sum(dim=-1)
    total_length = int(p_contact.shape[-1])
    inter_mask = inter_chain_pair_mask(
        int(p1_length),
        total_length - int(p1_length),
        device=p_contact.device,
    )
    if p_contact.ndim == 3:
        inter_mask = inter_mask.unsqueeze(0)
    inter_mask = inter_mask & pair_mask.to(dtype=torch.bool)
    p_bind = p_contact.masked_fill(~inter_mask, 0.0).amax(dim=(-2, -1)).clamp(0.0, 1.0)
    n_contacts = (p_contact.ge(float(ppi_score_threshold)) & inter_mask).sum(dim=(-2, -1))
    return p_bind, n_contacts


def _pad_value(tensor: torch.Tensor):
    if tensor.dtype == torch.bool:
        return False
    if torch.is_floating_point(tensor):
        return 0.0
    return 0


def _pad_first_dim(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_n = max(t.shape[0] for t in tensors)
    out = tensors[0].new_full((len(tensors), max_n, *tensors[0].shape[1:]), _pad_value(tensors[0]))
    for index, tensor in enumerate(tensors):
        out[index, : tensor.shape[0]] = tensor
    return out


def inference_collate_fn(batch):
    proteins, peptides, metas = zip(*batch)
    protein_batch = {}
    peptide_batch = {}
    for key in proteins[0]:
        values = [sample[key] for sample in proteins]
        protein_batch[key] = values if not isinstance(values[0], torch.Tensor) else _pad_first_dim(values)
    for key in peptides[0]:
        values = [sample[key] for sample in peptides]
        peptide_batch[key] = values if not isinstance(values[0], torch.Tensor) else _pad_first_dim(values)
    return protein_batch, peptide_batch, list(metas)


class InferencePairDataset(Dataset):
    def __init__(
        self,
        test_set_tsv: str | Path,
        fasta_path: str | Path,
        embedding_dir: str | Path,
        max_full_length: int | None = 1024,
    ):
        self.test_set_tsv = Path(test_set_tsv)
        self.fasta_path = Path(fasta_path)
        self.embedding_dir = Path(embedding_dir)
        self.max_full_length = None if max_full_length is None else int(max_full_length)
        metadata = pd.read_csv(self.test_set_tsv, sep="\t")
        if "Protein pairs" not in metadata.columns:
            raise ValueError("Test set TSV is missing the 'Protein pairs' column.")
        label_column = resolve_label_column(metadata.columns)
        sequences = load_fasta_sequences(self.fasta_path)

        self.items: list[dict[str, Any]] = []
        self.n_skipped = 0
        for _, row in metadata.iterrows():
            pair = str(row["Protein pairs"]).strip()
            protein_1, protein_2 = parse_protein_pair(pair)
            if protein_1 not in sequences:
                raise KeyError(f"Missing FASTA sequence for {protein_1}")
            if protein_2 not in sequences:
                raise KeyError(f"Missing FASTA sequence for {protein_2}")
            sequence_1 = sequences[protein_1]
            sequence_2 = sequences[protein_2]
            full_length = len(sequence_1) + len(sequence_2)
            if self.max_full_length is not None and full_length > self.max_full_length:
                self.n_skipped += 1
                continue
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
        embedding_path = self.embedding_dir / f"{protein_id}.pt"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {embedding_path}")
        return build_residue_features(sequence, load_plm_embedding(embedding_path))

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
    positive_weight: float = 1.0,
) -> dict[str, float]:
    if not rows:
        all_scores = torch.empty(0, dtype=torch.float32)
        all_targets = torch.empty(0, dtype=torch.bool)
    else:
        all_scores = torch.tensor([float(row["p_bind"]) for row in rows], dtype=torch.float32)
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
    frame = pd.DataFrame(rows, columns=["Protein pairs", "Category", "p_bind", "predicted", "n_inter_contacts"])
    frame.to_csv(path, sep="\t", index=False)


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    frame = pd.DataFrame(
        [
            {
                "AUROC": metrics["auroc"],
                "AUPRC": metrics["auprc"],
                "N": metrics["n"],
                "N_pos": metrics["n_pos"],
                "N_neg": metrics["n_neg"],
            }
        ]
    )
    frame.to_csv(path, index=False)


def evaluate_ppi(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    contact_threshold: float,
    distance_bin_start: float,
    distance_bin_end: float,
    distance_bin_count: int,
    recycle_rounds: int,
    ppi_score_threshold: float,
    positive_weight: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    contact_bins = contact_bin_count(
        contact_threshold,
        distance_bin_start,
        distance_bin_end,
        distance_bin_count,
    )
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for p1_batch, p2_batch, metas in loader:
            p1_batch = to_device(p1_batch, device)
            p2_batch = to_device(p2_batch, device)
            logits, pair_mask = model(
                p1_batch,
                p2_batch,
                recycle_rounds=max(1, int(recycle_rounds)),
            )
            p1_length = int(p1_batch["mask"].shape[1])
            p_bind, n_contacts = pair_bind_scores(
                logits,
                pair_mask,
                p1_length=p1_length,
                contact_bins=contact_bins,
                ppi_score_threshold=ppi_score_threshold,
            )
            p_bind = p_bind.detach().cpu()
            n_contacts = n_contacts.detach().cpu()
            for index, meta in enumerate(metas):
                score = float(p_bind[index].item())
                row = {
                    "Protein pairs": meta["pair"],
                    "Category": bool(meta["label"]),
                    "p_bind": score,
                    "predicted": score >= float(ppi_score_threshold),
                    "n_inter_contacts": int(n_contacts[index].item()),
                }
                if "index" in meta:
                    row["_index"] = meta["index"]
                rows.append(row)
            set_postfix = getattr(loader, "set_postfix", None)
            if callable(set_postfix) and len(p_bind) > 0:
                set_postfix(p_bind=f"{float(p_bind[0].item()):.3f}")

    return rows, metrics_from_prediction_rows(rows, positive_weight=positive_weight)


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
    dataset = InferencePairDataset(
        test_set_tsv=args.data.test_set_tsv,
        fasta_path=args.data.fasta_path,
        embedding_dir=embedding_dir,
        max_full_length=max_full_length,
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
    skipped_msg = (
        f" (skipped {dataset.n_skipped} with full length > {dataset.max_full_length})"
        if dataset.n_skipped
        else ""
    )
    log_info(f"Loaded {len(dataset)} test pairs{skipped_msg}")
    if DIST_WRAPPER.world_size > 1:
        print(
            f"[rank {DIST_WRAPPER.rank}/{DIST_WRAPPER.world_size}] "
            f"device={device} shard_pairs={len(shard)}/{len(dataset)}"
        )

    model_kwargs = _cfg_get(args, "model", default={})
    if isinstance(model_kwargs, DictConfig):
        model_kwargs = OmegaConf.to_container(model_kwargs, resolve=True)
    model_kwargs = dict(model_kwargs or {})
    model_kwargs.setdefault("num_classes", args.data.distance_bin_count)
    model = ResOnly(**model_kwargs).to(device)
    checkpoint = torch.load(args.ckpt_dir, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    del checkpoint

    eval_iter = tqdm(
        loader,
        desc="Inference",
        total=len(loader),
        leave=True,
        disable=DIST_WRAPPER.rank != 0,
    )
    positive_weight = float(_cfg_get(args, "metrics.positive_weight", default=1.0))
    rows, _ = evaluate_ppi(
        model=model,
        loader=eval_iter,
        device=device,
        contact_threshold=float(args.data.contact_threshold),
        distance_bin_start=float(args.data.distance_bin_start),
        distance_bin_end=float(args.data.distance_bin_end),
        distance_bin_count=int(args.data.distance_bin_count),
        recycle_rounds=int(args.recycle_rounds),
        ppi_score_threshold=float(args.data.ppi_score_threshold),
        positive_weight=positive_weight,
    )
    rows = gather_rank_prediction_rows(rows)

    if DIST_WRAPPER.rank == 0:
        metrics = metrics_from_prediction_rows(rows, positive_weight=positive_weight)
        predictions_path = os.path.join(logging_dir, "predictions.tsv")
        metrics_path = os.path.join(logging_dir, "metrics.csv")
        write_predictions(predictions_path, rows)
        write_metrics(metrics_path, metrics)
        print(
            f"[test-metrics] "
            f"auroc={metrics['auroc']:.4f} "
            f"auprc={metrics['auprc']:.4f} "
            f"n={metrics['n']} "
            f"n_pos={metrics['n_pos']} "
            f"n_neg={metrics['n_neg']}"
        )
        print(f"Wrote {predictions_path}")
        print(f"Wrote {metrics_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
