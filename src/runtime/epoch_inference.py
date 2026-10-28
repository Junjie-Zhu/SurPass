"""Checkpoint-linked full inference, shared with the standalone scoring path."""
import hashlib
import json
import time
from pathlib import Path
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import Subset

from src.data.collate import get_dataloader
from src.inference import InferencePairDataset, inference_collate_fn, evaluate_ppi, gather_prediction_rows, write_metrics, PREDICTION_COLUMNS
from src.runtime.config import cfg_get
from src.runtime.device import normalize_precision
from src.runtime.state import preserve_rng
from src.utils.ddp_utils import DIST_WRAPPER


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def inference_due(args, epoch):
    if not bool(cfg_get(args, "epoch_inference.enabled", False)):
        return False
    interval = int(cfg_get(args, "epoch_inference.every_n_epochs", args.checkpoint_interval))
    if interval != int(args.checkpoint_interval) or interval < 1:
        raise ValueError("epoch_inference interval must match checkpoint_interval")
    return epoch % interval == 0 or (epoch == int(args.epochs) and bool(cfg_get(args, "epoch_inference.run_on_final", True)))


def run_epoch_inference(args, epoch, checkpoint_path, output_dir, live_model, ema, device):
    if ema is None:
        raise ValueError("Periodic EMA inference requires EMA to be enabled")
    if str(cfg_get(args, "epoch_inference.weights", "ema")) != "ema":
        raise ValueError("Periodic checkpoint inference uses EMA weights")
    output = Path(output_dir) / "epoch_inference" / f"epoch_{epoch:04d}"
    # Identity binds outputs to checkpoint and resolved inference settings.
    cfg = OmegaConf.create(OmegaConf.to_container(args.epoch_inference.config, resolve=True))
    cfg.ckpt_dir = str(Path(checkpoint_path).resolve())
    cfg.output_dir = str(output.resolve())
    cfg.seed = int(args.seed)
    cfg.deterministic = bool(args.deterministic)
    cfg.task_prefix = f"infer_epoch_{epoch:04d}"
    cfg.logging_dir = str(output.parent.resolve())
    cfg.bin_spec = live_model.bin_spec
    cfg.dataset_identity = {
        "test_set_sha256": file_sha256(cfg.data.test_set_tsv),
        "fasta_sha256": file_sha256(cfg.data.fasta_path)}
    cfg.runtime.checkpoint_weights = "ema"
    resolved = OmegaConf.to_container(cfg, resolve=True)
    for key in ("distance_bin_start", "distance_bin_end", "distance_bin_count", "distance_bin_convention", "contact_threshold"):
        if cfg_get(cfg, f"data.{key}") != cfg_get(args, f"data.{key}"):
            raise ValueError(f"Periodic inference {key} differs from training")
    if resolved["model"] != OmegaConf.to_container(args.model, resolve=True):
        raise ValueError("Periodic inference model configuration differs from training")
    checkpoint_hash = file_sha256(checkpoint_path)
    identity = hashlib.sha256((checkpoint_hash + OmegaConf.to_yaml(cfg, resolve=True)).encode()).hexdigest()
    completed = output / "completion.json"
    skip = None
    if DIST_WRAPPER.rank == 0:
        try:
            skip = completed.is_file() and json.loads(completed.read_text()).get("identity") == identity
        except (json.JSONDecodeError, OSError):
            skip = False
    if dist.is_initialized():
        values = [skip]
        dist.broadcast_object_list(values, src=0)
        skip = values[0]
    if skip:
        return {}
    if DIST_WRAPPER.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / "config.yaml", resolve=True)
        (output / "status.json").write_text(json.dumps({"status": "running", "identity": identity}))
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    local_error = None
    rows = []
    try:
        with preserve_rng(), ema.average_parameters():
            dataset = InferencePairDataset(cfg.data.test_set_tsv, cfg.data.fasta_path, cfg.data.embedding_dir,
                                          maximum_length=int(cfg.data.maximum_length),
                                          embedding_cache_mb=float(cfg.data.embedding_cache_mb))
            indices = range(DIST_WRAPPER.rank, len(dataset), DIST_WRAPPER.world_size)
            loader = get_dataloader(Subset(dataset, list(indices)), inference_collate_fn,
                                    int(cfg.data.batch_size), shuffle=False, num_workers=int(cfg.data.num_workers),
                                    pin_memory=bool(cfg.data.pin_memory))
            rows = evaluate_ppi(live_model, loader, device, precision=str(cfg.runtime.precision),
                                score_threshold=float(cfg.data.ppi_score_threshold))
            if DIST_WRAPPER.rank == 0:
                rows.extend(dataset.unsupported_rows)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    errors = DIST_WRAPPER.all_gather_object(local_error)
    if any(errors):
        if DIST_WRAPPER.rank == 0:
            (output / "status.json").write_text(json.dumps({"status": "failed", "errors": errors}))
        raise RuntimeError(f"Periodic inference failed: {errors}")
    memory = DIST_WRAPPER.all_gather_object({
        "rank": DIST_WRAPPER.rank, "device": str(device),
        "precision": normalize_precision(str(cfg.runtime.precision), device),
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
    rows = gather_prediction_rows(rows)
    summary = {}
    if DIST_WRAPPER.rank == 0:
        pd.DataFrame(rows, columns=PREDICTION_COLUMNS).to_csv(output / "predictions.tsv", sep="\t", index=False)
        write_metrics(output / "metrics.csv", rows, float(cfg.metrics.positive_weight))
        metrics = pd.read_csv(output / "metrics.csv")
        for row in metrics.itertuples(index=False):
            summary[f"{row.head}_auprc_w{row.positive_weight:g}"] = row.AUPRC
            summary[f"{row.head}_auroc"] = row.AUROC
        scored = sum(row["status"] == "scored" for row in rows)
        summary.update(n_total=len(rows), n_scored=scored, coverage=scored / max(1, len(rows)))
        (output / "runtime.json").write_text(json.dumps({**summary, "seconds": time.perf_counter()-started, "memory_by_rank": memory,
                                                    "torch": torch.__version__, "cuda": torch.version.cuda}))
        record = {"status": "complete", "identity": identity, "checkpoint_sha256": checkpoint_hash,
                  "epoch": epoch, "bin_spec": live_model.bin_spec}
        atomic_json(completed, record)
        (output / "status.json").write_text(json.dumps(record, indent=2))
    return summary
