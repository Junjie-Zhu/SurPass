import datetime
import json
import os
import warnings

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra

from src.data.dataset import PepoDataset, collate_fn, get_dataloader
from src.model.vispepo import VisPepo
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except Exception:
    warnings.warn("torch_npu is not available")
warnings.filterwarnings("ignore")


def log_info(message: str):
    if DIST_WRAPPER.rank == 0:
        print(message)


def _binary_classification_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    probs = torch.sigmoid(logits.detach().flatten().to(dtype=torch.float32).cpu())
    target = target.detach().flatten().to(dtype=torch.float32).cpu()
    target_bool = target > 0.5
    pred_bool = probs >= threshold

    tp = (pred_bool & target_bool).sum().item()
    fp = (pred_bool & (~target_bool)).sum().item()
    fn = ((~pred_bool) & target_bool).sum().item()
    tn = ((~pred_bool) & (~target_bool)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)

    pos_count = int(target_bool.sum().item())
    neg_count = int((~target_bool).sum().item())
    if pos_count == 0 or neg_count == 0:
        auc = float("nan")
        auprc = float("nan")
    else:
        order = torch.argsort(probs, descending=True)
        y_sorted = target_bool[order].to(dtype=torch.float32)
        tps = torch.cumsum(y_sorted, dim=0)
        fps = torch.cumsum(1.0 - y_sorted, dim=0)

        tpr = torch.cat(
            [torch.tensor([0.0]), tps / max(pos_count, 1), torch.tensor([1.0])]
        )
        fpr = torch.cat(
            [torch.tensor([0.0]), fps / max(neg_count, 1), torch.tensor([1.0])]
        )
        auc = torch.trapz(tpr, fpr).item()

        precision_curve = tps / torch.arange(1, len(y_sorted) + 1, dtype=torch.float32)
        recall_curve = tps / max(pos_count, 1)
        recall_curve = torch.cat([torch.tensor([0.0]), recall_curve])
        precision_curve = torch.cat([torch.tensor([1.0]), precision_curve])
        auprc = torch.sum(
            (recall_curve[1:] - recall_curve[:-1]) * precision_curve[1:]
        ).item()

    return {
        "auc": float(auc),
        "auprc": float(auprc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": float(tn),
    }


def _masked_binary_rows(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.detach().reshape(-1)
    target = target.detach().reshape(-1)
    if mask is None:
        valid = torch.ones_like(logits, dtype=torch.bool, device=logits.device)
    else:
        valid = mask.detach().reshape(-1).to(dtype=torch.bool, device=logits.device)
    return logits[valid].unsqueeze(-1), target[valid].unsqueeze(-1)


def _to_device(obj, device):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                _to_device(value, device)
            elif isinstance(value, torch.Tensor):
                obj[key] = value.to(device)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    else:
        try:
            obj = obj.to(device)
        except Exception as exc:
            raise TypeError(f"Unsupported type for to_device: {type(obj)}") from exc
    return obj


def _cfg_get(cfg: DictConfig, *paths: str, default=None):
    for path in paths:
        selected = OmegaConf.select(cfg, path, default=None)
        if selected is not None:
            return selected
    return default


def _safe_float(x: float) -> float | None:
    if isinstance(x, float) and (x != x or x == float("inf") or x == float("-inf")):
        return None
    return float(x)


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def main(args: DictConfig):
    if not args.ckpt_dir:
        raise ValueError("`ckpt_dir` is required for inference.")
    if not _cfg_get(args, "data.test_csv_path"):
        raise ValueError("`data.test_csv_path` is required for inference.")

    logging_dir = os.path.join(
        args.logging_dir,
        f"{args.task_prefix}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(args.logging_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    with open(os.path.join(logging_dir, "config.yaml"), "w", encoding="utf-8") as f:
        OmegaConf.save(args, f)

    use_cuda = torch.cuda.device_count() > 0
    if use_cuda:
        device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    seed_everything(seed=args.seed, deterministic=args.deterministic)

    dataset = PepoDataset(
        csv_path=args.data.test_csv_path,
        center_coordinates=args.data.center_coordinates,
        random_rotation=args.data.random_rotation,
        align_surface=args.data.align_surface,
        surface_layout=args.data.surface_layout,
    )
    inference_loader = get_dataloader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.data.batch_size,
        shuffle=False,
        num_workers=args.data.num_workers,
        pin_memory=args.data.pin_memory,
        drop_last=False,
    )
    log_info(f"Loaded {len(dataset)} samples from {args.data.test_csv_path}")

    model_kwargs = _cfg_get(args, "model", default={})
    if isinstance(model_kwargs, DictConfig):
        model_kwargs = OmegaConf.to_container(model_kwargs, resolve=True)
    model = VisPepo(**(model_kwargs or {})).to(device)

    checkpoint = torch.load(args.ckpt_dir, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    log_info(f"Loaded checkpoint from {args.ckpt_dir}")
    log_info(
        f"Model has {sum(p.numel() for p in model.parameters()) / 1000000:.2f}M parameters"
    )

    model.eval()
    threshold = float(_cfg_get(args, "threshold", default=0.5))
    outputs_by_task = {"bind": [], "site1d": [], "pair2d": []}
    prediction_rows: list[dict] = []
    sample_offset = 0

    with torch.no_grad():
        infer_iter = enumerate(inference_loader)
        if DIST_WRAPPER.rank == 0:
            infer_iter = tqdm(
                infer_iter,
                desc="Inference",
                total=len(inference_loader),
                leave=True,
            )

        for _, step_batch in infer_iter:
            protein_batch, peptide_batch, labels = step_batch
            protein_batch = _to_device(protein_batch, device)
            peptide_batch = _to_device(peptide_batch, device)
            labels = _to_device(labels, device)

            binding_logits, binding_site_logits, pairwise_logits = model(
                protein_residue_batch=protein_batch["residue_features"],
                protein_atom_batch=protein_batch["atom_features"],
                protein_surface_batch=protein_batch["surface_features"],
                peptide_residue_batch=peptide_batch["residue_features"],
                peptide_atom_batch=peptide_batch["atom_features"],
                peptide_surface_batch=peptide_batch["surface_features"],
            )

            binding_target = labels["binding"].to(dtype=torch.float32).view(-1, 1)
            site_target = labels["label_1d"].to(dtype=torch.float32)
            pair_target = labels["label_2d"].to(dtype=torch.float32)

            protein_token_mask = protein_batch["residue_features"]["mask"].to(
                dtype=torch.bool
            )
            peptide_token_mask = peptide_batch["residue_features"]["mask"].to(
                dtype=torch.bool
            )
            site_mask = torch.cat([protein_token_mask, peptide_token_mask], dim=-1)
            pair_mask = protein_token_mask[:, :, None] & peptide_token_mask[:, None, :]

            bind_rows = _masked_binary_rows(
                logits=binding_logits, target=binding_target, mask=None
            )
            site_rows = _masked_binary_rows(
                logits=binding_site_logits.squeeze(-1),
                target=site_target,
                mask=site_mask,
            )
            pair_rows = _masked_binary_rows(
                logits=pairwise_logits,
                target=pair_target,
                mask=pair_mask,
            )
            outputs_by_task["bind"].append(bind_rows)
            outputs_by_task["site1d"].append(site_rows)
            outputs_by_task["pair2d"].append(pair_rows)

            bind_probs = torch.sigmoid(binding_logits).squeeze(-1).detach().cpu()
            bind_logits_cpu = binding_logits.squeeze(-1).detach().cpu()
            bind_labels_cpu = binding_target.squeeze(-1).detach().cpu()

            site_logits_cpu = binding_site_logits.squeeze(-1).detach().cpu()
            site_probs_cpu = torch.sigmoid(site_logits_cpu)
            site_labels_cpu = site_target.detach().cpu()
            site_mask_cpu = site_mask.detach().cpu()

            pair_logits_cpu = pairwise_logits.detach().cpu()
            pair_probs_cpu = torch.sigmoid(pair_logits_cpu)
            pair_labels_cpu = pair_target.detach().cpu()
            pair_mask_cpu = pair_mask.detach().cpu()

            batch_size = bind_probs.shape[0]
            for i in range(batch_size):
                sample_index = sample_offset + i
                row_meta = dataset.metadata.iloc[sample_index]

                site_valid = site_mask_cpu[i].to(dtype=torch.bool)
                pair_valid = pair_mask_cpu[i].to(dtype=torch.bool)

                site_prob_mean = (
                    site_probs_cpu[i][site_valid].mean().item()
                    if site_valid.any()
                    else float("nan")
                )
                site_label_mean = (
                    site_labels_cpu[i][site_valid].mean().item()
                    if site_valid.any()
                    else float("nan")
                )
                pair_prob_mean = (
                    pair_probs_cpu[i][pair_valid].mean().item()
                    if pair_valid.any()
                    else float("nan")
                )
                pair_label_mean = (
                    pair_labels_cpu[i][pair_valid].mean().item()
                    if pair_valid.any()
                    else float("nan")
                )

                prediction_rows.append(
                    {
                        "sample_index": sample_index,
                        "protein_path": row_meta.get(dataset.protein_path_column, ""),
                        "peptide_path": row_meta.get(dataset.peptide_path_column, ""),
                        "bind_logit": float(bind_logits_cpu[i].item()),
                        "bind_prob": float(bind_probs[i].item()),
                        "bind_label": float(bind_labels_cpu[i].item()),
                        "site1d_prob_mean": _safe_float(site_prob_mean),
                        "site1d_label_mean": _safe_float(site_label_mean),
                        "pair2d_prob_mean": _safe_float(pair_prob_mean),
                        "pair2d_label_mean": _safe_float(pair_label_mean),
                    }
                )
            sample_offset += batch_size

    metrics_by_task = {}
    for task_name in ("bind", "site1d", "pair2d"):
        task_logits = torch.cat([rows[0] for rows in outputs_by_task[task_name]], dim=0)
        task_targets = torch.cat([rows[1] for rows in outputs_by_task[task_name]], dim=0)
        metrics_by_task[task_name] = _binary_classification_metrics(
            logits=task_logits,
            target=task_targets,
            threshold=threshold,
        )

    summary_payload = {
        "task_prefix": str(args.task_prefix),
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint_path": str(args.ckpt_dir),
        "num_samples": int(len(dataset)),
        "threshold": threshold,
        "metrics": {
            task_name: {
                metric_name: _safe_float(metric_value)
                for metric_name, metric_value in task_metrics.items()
            }
            for task_name, task_metrics in metrics_by_task.items()
        },
    }

    metrics_row = {
        "num_samples": len(dataset),
        "threshold": threshold,
    }
    for task_name, task_metrics in metrics_by_task.items():
        for metric_name, metric_value in task_metrics.items():
            metrics_row[f"{task_name}_{metric_name}"] = _safe_float(metric_value)

    metrics_csv_path = os.path.join(logging_dir, "metrics.csv")
    metrics_json_path = os.path.join(logging_dir, "metrics.json")
    summary_txt_path = os.path.join(logging_dir, "summary.txt")
    predictions_csv_path = os.path.join(logging_dir, "predictions.csv")

    pd.DataFrame([metrics_row]).to_csv(metrics_csv_path, index=False)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    pd.DataFrame(prediction_rows).to_csv(predictions_csv_path, index=False)

    bind_m = metrics_by_task["bind"]
    site_m = metrics_by_task["site1d"]
    pair_m = metrics_by_task["pair2d"]
    summary_lines = [
        "Inference summary",
        f"samples={len(dataset)} threshold={threshold:.3f}",
        f"checkpoint={args.ckpt_dir}",
        (
            f"[bind] auc={bind_m['auc']:.4f} auprc={bind_m['auprc']:.4f} "
            f"precision={bind_m['precision']:.4f} recall={bind_m['recall']:.4f} f1={bind_m['f1']:.4f}"
        ),
        (
            f"[site1d] auc={site_m['auc']:.4f} auprc={site_m['auprc']:.4f} "
            f"precision={site_m['precision']:.4f} recall={site_m['recall']:.4f} f1={site_m['f1']:.4f}"
        ),
        (
            f"[pair2d] auc={pair_m['auc']:.4f} auprc={pair_m['auprc']:.4f} "
            f"precision={pair_m['precision']:.4f} recall={pair_m['recall']:.4f} f1={pair_m['f1']:.4f}"
        ),
        f"metrics_csv={metrics_csv_path}",
        f"metrics_json={metrics_json_path}",
        f"summary_txt={summary_txt_path}",
        f"predictions_csv={predictions_csv_path}",
    ]
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    log_info("\n".join(summary_lines))


if __name__ == "__main__":
    main()