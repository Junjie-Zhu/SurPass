import datetime
import os
import warnings
from typing import Any, Callable, Dict

import hydra
from sklearn.metrics import average_precision_score, roc_curve, auc
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None

from src.data.balancing import DistributedWeightedSampler, build_multi_source_split
from src.data.collate import collate_fn, get_dataloader
from src.data.labels import inter_chain_pair_mask
from src.model.loss import (
    ConsistencyLoss,
    DistogramMAELoss,
    FocalCELoss,
    ProteinBindLoss,
    ResidueBindLoss,
    TverskyLoss,
    distogram_supervision,
    downsample_inter_negatives,
    residue_bind_targets,
)
from src.model.ema import EMAWrapper
from src.model.optimizer import get_lr_scheduler, get_optimizer
from src.model.surpass import ResOnly, SurpassOutput
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything
# from src.utils.nan_monitor import abort_if_nonfinite_loss

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except Exception:
    warnings.warn("torch_npu is not available")
warnings.filterwarnings("ignore")


def log_info(message: str):
    if DIST_WRAPPER.rank == 0:
        print(message)


def _cfg_get(cfg: DictConfig, *paths: str, default=None):
    for path in paths:
        selected = OmegaConf.select(cfg, path, default=None)
        if selected is not None:
            return selected
    return default


def to_device(obj, device):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                to_device(value, device)
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


def contact_bin_count(
    contact_threshold: float,
    distance_bin_start: float,
    distance_bin_end: float,
    distance_bin_count: int,
) -> int:
    bin_limits = torch.linspace(
        float(distance_bin_start),
        float(distance_bin_end),
        int(distance_bin_count) - 1,
    )
    count = int(torch.bucketize(torch.tensor(float(contact_threshold)), bin_limits).item())
    return max(1, min(count, int(distance_bin_count)))


def _binary_classification_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
    positive_weight: float = 1.0,
) -> dict[str, float]:
    scores = scores.detach().flatten().to(dtype=torch.float32).cpu().numpy()
    target_bool = target.detach().flatten().to(dtype=torch.bool).cpu().numpy()
    pos_count = int(target_bool.sum())
    neg_count = int(target_bool.size - pos_count)
    weight = float(positive_weight)
    if scores.size == 0 or pos_count == 0 or neg_count == 0 or weight <= 0.0:
        return {"auroc": float("nan"), "auprc": float("nan")}
    sample_weight = None
    if weight != 1.0:
        sample_weight = 1.0 + (weight - 1.0) * target_bool.astype("float64")
    fpr, tpr, _ = roc_curve(target_bool, scores, sample_weight=sample_weight)
    auroc = auc(fpr, tpr)
    auprc = average_precision_score(
        target_bool, scores, sample_weight=sample_weight
    )
    return {"auroc": float(auroc), "auprc": float(auprc)}


def _region_masks(
    pair_mask: torch.Tensor,
    label_mask: torch.Tensor,
    p1_length: int | torch.Tensor,
    p2_length: int | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = pair_mask.to(dtype=torch.bool) & label_mask.to(dtype=torch.bool)
    total_length = int(valid_mask.shape[-1])
    if p2_length is None:
        if isinstance(p1_length, torch.Tensor) and p1_length.ndim > 0:
            raise ValueError("p2_length is required when p1_length is batched.")
        p2_length = total_length - int(p1_length)
    inter_mask = inter_chain_pair_mask(
        p1_length,
        p2_length,
        device=valid_mask.device,
        total_length=total_length,
    )
    if valid_mask.ndim == 3 and inter_mask.ndim == 2:
        inter_mask = inter_mask.unsqueeze(0)
    intra_mask = valid_mask & ~inter_mask
    inter_mask = valid_mask & inter_mask
    return intra_mask, inter_mask


def pair_bind_scores(
    logits: torch.Tensor,
    pair_mask: torch.Tensor,
    p1_length: int | torch.Tensor,
    contact_bins: int,
    ppi_score_threshold: float = 0.5,
    p2_length: int | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    p_contact = torch.softmax(logits, dim=-1)[..., : int(contact_bins)].sum(dim=-1)
    total_length = int(p_contact.shape[-1])
    if p2_length is None:
        if isinstance(p1_length, torch.Tensor) and p1_length.ndim > 0:
            raise ValueError("p2_length is required when p1_length is batched.")
        p2_length = total_length - int(p1_length)
    inter_mask = inter_chain_pair_mask(
        p1_length,
        p2_length,
        device=p_contact.device,
        total_length=total_length,
    )
    if p_contact.ndim == 3 and inter_mask.ndim == 2:
        inter_mask = inter_mask.unsqueeze(0)
    inter_mask = inter_mask & pair_mask.to(dtype=torch.bool)
    p_bind = p_contact.masked_fill(~inter_mask, 0.0).amax(dim=(-2, -1)).clamp(0.0, 1.0)
    n_contacts = (p_contact.ge(float(ppi_score_threshold)) & inter_mask).sum(dim=(-2, -1))
    return p_bind, n_contacts


def _compute_pair_losses(
    loss_fn: Dict[str, torch.nn.Module],
    loss_weights: Dict[str, float],
    output: SurpassOutput,
    target_bins: torch.Tensor,
    intra_mask: torch.Tensor,
    inter_mask: torch.Tensor,
    is_contact: torch.Tensor,
    pair_probs: torch.Tensor,
    contact_probability: torch.Tensor,
    residue_target: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    residue_inter_mask: torch.Tensor | None = None,
    is_positive: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = output.pair_logits
    raw_loss = logits.new_zeros(())
    terms: dict[str, torch.Tensor] = {}
    full_inter_mask = residue_inter_mask if residue_inter_mask is not None else inter_mask
    for name, loss_term in loss_fn.items():
        if name in {"residue", "protein", "consistency"}:
            continue
        term_weight = float(loss_weights[name])
        if term_weight <= 0.0:
            continue
        if name == "tversky":
            value = loss_term(contact_probability, is_contact, inter_mask)
        elif name == "focal":
            intra_loss, inter_loss = loss_term(
                logits, target_bins, intra_mask, inter_mask, is_contact
            )
            value = (
                float(loss_weights["intra"]) * intra_loss
                + float(loss_weights["inter"]) * inter_loss
            )
        elif name == "mae":
            intra_loss, inter_loss = loss_term(
                pair_probs, target_bins, intra_mask, inter_mask
            )
            value = (
                float(loss_weights["intra"]) * intra_loss
                + float(loss_weights["inter"]) * inter_loss
            )
        else:
            raise KeyError(f"Unknown pair loss {name}.")
        terms[name] = value.detach()
        raw_loss = raw_loss + term_weight * value

    residue_weight = float(loss_weights.get("residue", 0.0))
    if (
        residue_weight > 0.0
        and "residue" in loss_fn
        and residue_mask is not None
        and is_positive is not None
    ):
        residue_loss = loss_fn["residue"](
            output.residue_logits,
            residue_target,
            residue_mask,
            is_positive,
        )
        terms["residue"] = residue_loss.detach()
        raw_loss = raw_loss + residue_weight * residue_loss

    protein_weight = float(loss_weights.get("protein", 0.0))
    if protein_weight > 0.0 and "protein" in loss_fn and is_positive is not None:
        protein_loss = loss_fn["protein"](output.protein_logit, is_positive)
        terms["protein"] = protein_loss.detach()
        raw_loss = raw_loss + protein_weight * protein_loss

    consistency_weight = float(loss_weights.get("consistency", 0.0))
    if (
        consistency_weight > 0.0
        and "consistency" in loss_fn
        and residue_mask is not None
    ):
        consistency_loss = loss_fn["consistency"](
            output.residue_logits,
            output.protein_logit,
            contact_probability,
            full_inter_mask,
            residue_mask,
        )
        terms["consistency"] = consistency_loss.detach()
        raw_loss = raw_loss + consistency_weight * consistency_loss

    terms["total"] = raw_loss.detach()
    return raw_loss, terms


def _as_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _terms_to_floats(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: _as_float(value) for name, value in terms.items()}


def _accumulate_floats(
    totals: dict[str, float],
    values: dict[str, float],
) -> None:
    for name, value in values.items():
        totals[name] = totals.get(name, 0.0) + value


def _mean_floats(totals: dict[str, float], count: int) -> dict[str, float]:
    scale = max(int(count), 1)
    return {name: value / scale for name, value in totals.items()}


_TRAIN_LOG_KEYS = (
    "total",
    "focal",
    "mae",
    "tversky",
    "residue",
    "protein",
    "consistency",
)
_TEST_LOG_KEYS = (
    "total",
    "protein_auroc",
    "protein_auprc",
    "residue_auroc",
    "residue_auprc",
    "inter_auroc",
    "inter_auprc",
)


def _select_log_keys(metrics: dict[str, float], keys: tuple[str, ...]) -> dict[str, float]:
    return {name: metrics[name] for name in keys if name in metrics}


def _progress_postfix(terms: dict[str, float]) -> dict[str, str]:
    aliases = (
        ("loss", "total"),
        ("focal", "focal"),
        ("mae", "mae"),
        ("tv", "tversky"),
        ("res", "residue"),
        ("prot", "protein"),
        ("cons", "consistency"),
    )
    return {
        label: f"{terms[key]:.3f}"
        for label, key in aliases
        if key in terms
    }


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


def _wandb_enabled(cfg: DictConfig) -> bool:
    return bool(_cfg_get(cfg, "wandb.enabled", default=True))


def _init_wandb(cfg: DictConfig, logging_dir: str, run_name: str):
    if DIST_WRAPPER.rank != 0 or not _wandb_enabled(cfg):
        return None
    if wandb is None:
        log_info("wandb is enabled in config but the package is not installed")
        return None

    init_kwargs: dict[str, Any] = {
        "project": str(_cfg_get(cfg, "wandb.project", default="surpass")),
        "name": str(_cfg_get(cfg, "wandb.name", default=run_name)),
        "dir": logging_dir,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "mode": str(_cfg_get(cfg, "wandb.mode", default="online")),
        "resume": str(_cfg_get(cfg, "wandb.resume", default="allow")),
        "tags": list(_cfg_get(cfg, "wandb.tags", default=[]) or []),
    }
    entity = _cfg_get(cfg, "wandb.entity", default=None)
    if entity:
        init_kwargs["entity"] = str(entity)
    run_id = _cfg_get(cfg, "wandb.id", default=None)
    if run_id:
        init_kwargs["id"] = str(run_id)

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        log_info(f"wandb.init failed ({exc}); continuing without wandb")
        return None
    return run


def _wandb_log(payload: dict[str, Any], step: int) -> None:
    if wandb is None or wandb.run is None:
        return
    wandb.log(payload, step=int(step))


def _build_loss_weights(cfg: DictConfig) -> dict[str, float]:
    return {
        "intra": float(_cfg_get(cfg, "loss.intra", default=0.3)),
        "inter": float(_cfg_get(cfg, "loss.inter", default=0.7)),
        "focal": float(_cfg_get(cfg, "loss.focal.weight", default=1.0)),
        "mae": float(_cfg_get(cfg, "loss.mae.weight", default=1.0)),
        "tversky": float(_cfg_get(cfg, "loss.tversky.weight", default=1.0)),
        "residue": float(_cfg_get(cfg, "loss.residue_bind.weight", default=1.0)),
        "protein": float(_cfg_get(cfg, "loss.protein.weight", default=1.0)),
        "consistency": float(_cfg_get(cfg, "loss.consistency.weight", default=0.1)),
    }


def resolve_chunk_size(cfg: DictConfig, default: int | None = 64) -> int | None:
    value = _cfg_get(cfg, "performance.chunk_size", default=default)
    if value is None:
        return None
    chunk_size = int(value)
    return None if chunk_size <= 0 else chunk_size


def _validate_model_bin_counts(model_kwargs: dict, distance_bin_count: int) -> dict:
    model_kwargs = dict(model_kwargs)
    distance_bin_count = int(distance_bin_count)
    model_kwargs.setdefault("num_classes", distance_bin_count)
    model_kwargs.setdefault("xt_pair_dist_dim", distance_bin_count)
    num_classes = int(model_kwargs["num_classes"])
    xt_pair_dist_dim = int(model_kwargs["xt_pair_dist_dim"])
    if len({num_classes, xt_pair_dist_dim, distance_bin_count}) != 1:
        raise ValueError(
            "num_classes, xt_pair_dist_dim, and data.distance_bin_count must be equal, "
            f"got {num_classes}, {xt_pair_dist_dim}, {distance_bin_count}."
        )
    return model_kwargs


def build_model_kwargs(cfg: DictConfig, *, checkpoint_default: bool = True) -> dict:
    model_kwargs = _cfg_get(cfg, "model", default={})
    if isinstance(model_kwargs, DictConfig):
        model_kwargs = OmegaConf.to_container(model_kwargs, resolve=True)
    model_kwargs = dict(model_kwargs or {})
    model_kwargs = _validate_model_bin_counts(model_kwargs, cfg.data.distance_bin_count)
    model_kwargs.pop("n_pair_heads", None)
    model_kwargs["contact_bins"] = contact_bin_count(
        cfg.data.contact_threshold,
        cfg.data.distance_bin_start,
        cfg.data.distance_bin_end,
        cfg.data.distance_bin_count,
    )
    model_kwargs["checkpoint_pair_blocks"] = bool(
        _cfg_get(cfg, "performance.checkpoint_pair_blocks", default=checkpoint_default)
    )
    return model_kwargs


def _live_model(model: torch.nn.Module) -> torch.nn.Module:
    if DIST_WRAPPER.world_size > 1:
        return model.module
    return model


def _ema_keywords(cfg: DictConfig) -> list:
    keywords = _cfg_get(cfg, "ema.mutable_param_keywords", default=[""])
    if keywords is None:
        return [""]
    if OmegaConf.is_config(keywords):
        keywords = OmegaConf.to_container(keywords, resolve=True)
    return list(keywords)


def _unpack_batch(step_batch, device):
    residue_batch, labels = step_batch
    residue_batch = to_device(residue_batch, device)
    labels = to_device(labels, device)
    return residue_batch, labels


def _set_progress_postfix(progress, **kwargs) -> None:
    set_postfix = getattr(progress, "set_postfix", None)
    if callable(set_postfix):
        set_postfix(**kwargs)


def _release_rejected_loss_graph(loss: torch.Tensor) -> None:
    loss.backward(gradient=torch.zeros_like(loss))


def resolve_cuda_device(local_rank: int, device_count: int) -> torch.device:
    if device_count <= 0:
        return torch.device("cpu")
    index = 0 if device_count == 1 else int(local_rank)
    if index >= device_count:
        index = int(local_rank) % device_count
    return torch.device(f"cuda:{index}")


def resolve_dist_backend(use_cuda: bool) -> str:
    return "nccl" if use_cuda else "gloo"


def wrap_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if device.type == "cuda":
        return DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )
    return DDP(model, find_unused_parameters=False)


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Dict[str, torch.nn.Module],
    loss_weights: Dict[str, float],
    device: torch.device,
    max_grad_norm: float = 0.0,
    grad_accum_steps: int = 1,
    scheduler: Any | None = None,
    max_batches: int | None = None,
    step_logger: Callable[[dict[str, float]], None] | None = None,
    contact_bins: int | None = None,
    inter_neg_per_pos: float = 0.0,
    dump_dir: str | None = None,
    epoch: int = 0,
    chunk_size: int | None = None,
    ema_wrapper: EMAWrapper | None = None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    term_totals: dict[str, float] = {}
    num_steps = 0
    grad_accum_steps = max(1, int(grad_accum_steps))

    for step, step_batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        residue_batch, labels = _unpack_batch(step_batch, device)

        output = model(residue_batch, chunk_size=chunk_size)
        intra_mask, inter_mask = _region_masks(
            output.pair_mask,
            labels["label_2d_mask"],
            p1_length=residue_batch["p1_length"],
            p2_length=residue_batch["p2_length"],
        )
        is_contact, pair_probs, contact_probability = distogram_supervision(
            output.pair_logits,
            labels["label_2d_bins"],
            int(contact_bins),
        )
        pair_inter_mask = inter_mask
        if float(inter_neg_per_pos) > 0.0:
            pair_inter_mask = downsample_inter_negatives(
                inter_mask,
                is_contact,
                neg_per_pos=float(inter_neg_per_pos),
            )
        raw_loss, terms = _compute_pair_losses(
            loss_fn,
            loss_weights,
            output,
            labels["label_2d_bins"],
            intra_mask,
            pair_inter_mask,
            is_contact,
            pair_probs,
            contact_probability,
            residue_bind_targets(is_contact, inter_mask),
            residue_mask=residue_batch["mask"],
            residue_inter_mask=inter_mask,
            is_positive=labels["is_positive"],
        )

        loss = raw_loss / grad_accum_steps
        loss.backward()

        should_step = ((step + 1) % grad_accum_steps == 0) or (
            step + 1 == len(loader)
        )
        if should_step:
            clip_limit = float(max_grad_norm) if float(max_grad_norm) > 0.0 else float("inf")
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit)
            optimizer.step()
            if ema_wrapper is not None:
                ema_wrapper.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        term_floats = _terms_to_floats(terms)
        _accumulate_floats(term_totals, term_floats)
        num_steps += 1
        _set_progress_postfix(loader, **_progress_postfix(term_floats))
        if step_logger is not None:
            step_logger(term_floats)

    return _mean_floats(term_totals, num_steps)


def _contact_metrics_from_rows(
    score_rows: list[torch.Tensor],
    target_rows: list[torch.Tensor],
    positive_weight: float = 1.0,
) -> dict[str, float]:
    if not score_rows:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "contact_prev": float("nan"),
        }
    all_scores = torch.cat(score_rows, dim=0)
    all_targets = torch.cat(target_rows, dim=0)
    if all_scores.numel() == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "contact_prev": float("nan"),
        }
    metrics = _binary_classification_metrics(
        all_scores, all_targets, positive_weight=positive_weight
    )
    contact_prev = (
        float(all_targets.to(dtype=torch.float32).mean().item())
        if all_targets.numel() > 0
        else float("nan")
    )
    return {
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "contact_prev": contact_prev,
    }


def evaluate_epoch(
    model: torch.nn.Module,
    loader,
    loss_fn: Dict[str, torch.nn.Module],
    loss_weights: Dict[str, float],
    device: torch.device,
    contact_bins: int,
    max_batches: int | None = None,
    positive_weight: float = 1.0,
    chunk_size: int | None = None,
) -> dict[str, float]:
    model.eval()
    term_totals: dict[str, float] = {}
    num_steps = 0
    inter_score_rows = []
    inter_target_rows = []
    residue_score_rows = []
    residue_target_rows = []
    protein_score_rows = []
    protein_target_rows = []

    with torch.no_grad():
        for step, step_batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            residue_batch, labels = _unpack_batch(step_batch, device)
            output = model(residue_batch, chunk_size=chunk_size)
            intra_mask, inter_mask = _region_masks(
                output.pair_mask,
                labels["label_2d_mask"],
                p1_length=residue_batch["p1_length"],
                p2_length=residue_batch["p2_length"],
            )
            is_contact, pair_probs, contact_probability = distogram_supervision(
                output.pair_logits,
                labels["label_2d_bins"],
                int(contact_bins),
            )
            residue_target = residue_bind_targets(is_contact, inter_mask)
            _, terms = _compute_pair_losses(
                loss_fn,
                loss_weights,
                output,
                labels["label_2d_bins"],
                intra_mask,
                inter_mask,
                is_contact,
                pair_probs,
                contact_probability,
                residue_target,
                residue_mask=residue_batch["mask"],
                residue_inter_mask=inter_mask,
                is_positive=labels["is_positive"],
            )

            inter_score_rows.append(contact_probability[inter_mask].detach().cpu())
            inter_target_rows.append(is_contact[inter_mask].detach().cpu())

            residue_mask = residue_batch["mask"].to(dtype=torch.bool)
            residue_prob = torch.sigmoid(output.residue_logits).reshape(residue_mask.shape)
            residue_score_rows.append(residue_prob[residue_mask].detach().cpu())
            residue_target_rows.append(residue_target[residue_mask].detach().cpu())
            protein_score_rows.append(torch.sigmoid(output.protein_logit.reshape(-1)).detach().cpu())
            protein_target_rows.append(
                labels["is_positive"].reshape(-1).to(dtype=torch.bool).detach().cpu()
            )

            term_floats = _terms_to_floats(terms)
            _accumulate_floats(term_totals, term_floats)
            num_steps += 1
            _set_progress_postfix(loader, **_progress_postfix(term_floats))

    inter_metrics = _contact_metrics_from_rows(
        inter_score_rows, inter_target_rows, positive_weight=positive_weight
    )
    residue_metrics = _contact_metrics_from_rows(
        residue_score_rows, residue_target_rows, positive_weight=positive_weight
    )
    protein_metrics = _contact_metrics_from_rows(
        protein_score_rows, protein_target_rows, positive_weight=positive_weight
    )
    return _select_log_keys(
        {
            "total": _mean_floats(term_totals, num_steps).get("total", float("nan")),
            "protein_auroc": protein_metrics["auroc"],
            "protein_auprc": protein_metrics["auprc"],
            "residue_auroc": residue_metrics["auroc"],
            "residue_auprc": residue_metrics["auprc"],
            "inter_auroc": inter_metrics["auroc"],
            "inter_auprc": inter_metrics["auprc"],
        },
        _TEST_LOG_KEYS,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(args: DictConfig):
    logging_dir = os.path.join(
        args.logging_dir,
        f"{str(args.task_prefix).upper()}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    if DIST_WRAPPER.rank == 0:
        os.makedirs(args.logging_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(os.path.join(logging_dir, "checkpoints"), exist_ok=True)
        with open(f"{logging_dir}/config.yaml", "w", encoding="utf-8") as f:
            OmegaConf.save(args, f)

    use_cuda = torch.cuda.device_count() > 0
    if use_cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = resolve_cuda_device(DIST_WRAPPER.local_rank, torch.cuda.device_count())
        all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        devices = "cpu"

    if DIST_WRAPPER.world_size > 1 and not dist.is_initialized():
        if DIST_WRAPPER.rank == 0:
            log_info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            log_info(
                f"Using DDP with {DIST_WRAPPER.world_size} processes, rank: {DIST_WRAPPER.rank}"
            )
        timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
        dist.init_process_group(
            backend=resolve_dist_backend(use_cuda),
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )

    seed_everything(seed=args.seed, deterministic=args.deterministic)

    run_name = os.path.basename(logging_dir)
    wandb_run = _init_wandb(args, logging_dir, run_name)
    wandb_log_interval = max(1, int(_cfg_get(args, "wandb.log_interval", default=1)))
    global_step = 0
    if wandb_run is not None:
        log_info(
            f"wandb run: {getattr(wandb_run, 'name', None)} "
            f"({getattr(wandb_run, 'id', None)})"
        )

    sources = _cfg_get(args, "data.datasets", default=None)
    if sources is None:
        raise ValueError("data.datasets must list at least one cluster source.")
    sources = OmegaConf.to_container(sources, resolve=True)
    train_dataset, test_dataset, train_weights, source_info = build_multi_source_split(
        sources,
        root_dir=args.data.root_dir,
        test_fraction=args.data.test_fraction,
        seed=args.seed,
        negative_ratio=int(_cfg_get(args, "data.negative_ratio", default=1)),
        dataset_kwargs={
            "distance_bin_start": args.data.distance_bin_start,
            "distance_bin_end": args.data.distance_bin_end,
            "distance_bin_count": args.data.distance_bin_count,
            "crop_size": args.data.crop_size,
            "crop_contiguous_probability": float(
                _cfg_get(args, "data.crop_contiguous_prob", default=0.3)
            ),
            "contact_threshold": args.data.contact_threshold,
        },
    )
    single_source = len(sources) == 1
    if single_source:
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=DIST_WRAPPER.world_size,
                rank=DIST_WRAPPER.rank,
                shuffle=True,
            )
            if DIST_WRAPPER.world_size > 1
            else None
        )
    else:
        epoch_size = int(_cfg_get(args, "data.epoch_size", default=30000))
        train_sampler = DistributedWeightedSampler(
            train_weights,
            num_samples=epoch_size,
            num_replicas=DIST_WRAPPER.world_size,
            rank=DIST_WRAPPER.rank,
            seed=args.seed,
        )
    for info in source_info:
        if info["assemble_negatives"]:
            kind = "on-the-fly negatives"
        elif info["is_positive"]:
            kind = "static positive"
        else:
            kind = "static negative"
        if single_source:
            log_info(
                f"Source {info['name']}: train={info['train_length']} test={info['test_length']} "
                f"({kind}); one shuffled pass per epoch"
            )
        else:
            log_info(
                f"Source {info['name']}: train={info['train_length']} test={info['test_length']} "
                f"probability={info['probability']:.3f} ({kind})"
            )
    if single_source:
        samples_per_rank = len(train_dataset) if train_sampler is None else len(train_sampler)
        log_info(
            f"One shuffled pass: {samples_per_rank} samples per rank "
            f"across {DIST_WRAPPER.world_size} ranks"
        )
    else:
        log_info(
            f"Epoch size {epoch_size}: {len(train_sampler)} samples per rank "
            f"across {DIST_WRAPPER.world_size} ranks"
        )
    test_sampler = (
        DistributedSampler(
            test_dataset,
            num_replicas=DIST_WRAPPER.world_size,
            rank=DIST_WRAPPER.rank,
            shuffle=False,
        )
        if DIST_WRAPPER.world_size > 1
        else None
    )

    train_loader = get_dataloader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.data.batch_size,
        shuffle=train_sampler is None,
        num_workers=args.data.num_workers,
        sampler=train_sampler,
        pin_memory=args.data.pin_memory,
    )
    test_loader = get_dataloader(
        test_dataset,
        collate_fn=collate_fn,
        batch_size=args.data.batch_size,
        shuffle=False,
        num_workers=args.data.num_workers,
        sampler=test_sampler,
        pin_memory=args.data.pin_memory,
    )

    model_kwargs = build_model_kwargs(args, checkpoint_default=True)
    n_contact_bins = int(model_kwargs["contact_bins"])
    chunk_size = resolve_chunk_size(args, default=64)
    model = ResOnly(**model_kwargs).to(device)
    if DIST_WRAPPER.world_size > 1:
        model = wrap_ddp(model, device)
    ema_decay = float(_cfg_get(args, "ema.decay", default=0.0))
    ema_wrapper = None
    if ema_decay > 0:
        ema_wrapper = EMAWrapper(
            model=model,
            decay=ema_decay,
            mutable_param_keywords=_ema_keywords(args),
        )
        ema_wrapper.register()
    log_info(
        f"Model instantiated with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    focal_loss = FocalCELoss(
        contact_bins=n_contact_bins,
        gamma=float(_cfg_get(args, "loss.focal.gamma", default=1.5)),
        alpha=float(_cfg_get(args, "loss.focal.alpha", default=5.0)),
    ).to(device)
    mae_loss = DistogramMAELoss(
        bin_start=args.data.distance_bin_start,
        bin_end=args.data.distance_bin_end,
        bin_count=args.data.distance_bin_count,
    ).to(device)
    tversky_loss = TverskyLoss(
        contact_bins=n_contact_bins,
        alpha=float(_cfg_get(args, "loss.tversky.alpha", default=0.7)),
        beta=float(_cfg_get(args, "loss.tversky.beta", default=0.3)),
    ).to(device)
    residue_bind_loss = ResidueBindLoss(
        neg_per_pos=float(_cfg_get(args, "loss.residue_bind.neg_per_pos", default=4.0)),
        negative_pair_weight=float(
            _cfg_get(args, "loss.residue_bind.negative_pair_weight", default=0.25)
        ),
    ).to(device)
    consistency_loss = ConsistencyLoss(contact_bins=n_contact_bins).to(device)
    loss_fn = {
        "focal": focal_loss,
        "mae": mae_loss,
        "tversky": tversky_loss,
        "residue": residue_bind_loss,
        "protein": ProteinBindLoss().to(device),
        "consistency": consistency_loss,
    }
    loss_weights = _build_loss_weights(args)
    log_info(
        f"Loss setup: contact_bins={n_contact_bins}, weights={loss_weights}, "
        f"inter_neg_per_pos={float(_cfg_get(args, 'loss.inter_neg_per_pos', default=4.0))}, "
        f"chunk_size={chunk_size}"
    )
    optimizer = get_optimizer(
        model,
        lr=args.optimizer.lr,
        weight_decay=args.optimizer.weight_decay,
        betas=(args.optimizer.beta1, args.optimizer.beta2),
        use_adamw=args.optimizer.use_adamw,
    )
    scheduler = get_lr_scheduler(
        optimizer,
        lr_scheduler=args.optimizer.lr_scheduler,
        lr=args.optimizer.lr,
        max_steps=args.epochs * len(train_loader) + 100,
        warmup_steps=args.optimizer.warmup_steps,
        decay_every_n_steps=args.optimizer.decay_every_n_steps,
        decay_factor=args.optimizer.decay_factor,
    )

    start_epoch = 1
    ema_dir = _cfg_get(args, "ema_dir", default=None)
    if ema_dir is not None and ema_wrapper is not None:
        ema_checkpoint = torch.load(ema_dir, map_location=device)
        _live_model(model).load_state_dict(ema_checkpoint["model_state_dict"])
        ema_wrapper.register()
        del ema_checkpoint

    if args.ckpt_dir is not None:
        checkpoint = torch.load(args.ckpt_dir, map_location=device)
        _live_model(model).load_state_dict(checkpoint["model_state_dict"])
        if not args.load_model_only:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
        del checkpoint

    csv_path = os.path.join(logging_dir, "loss.csv")
    csv_fields = (
        ["epoch"]
        + [f"train_{name}" for name in _TRAIN_LOG_KEYS]
        + [f"test_{name}" for name in _TEST_LOG_KEYS]
    )
    if DIST_WRAPPER.rank == 0:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(csv_fields) + "\n")

    epoch_progress = (
        tqdm(total=args.epochs, leave=False, position=0)
        if DIST_WRAPPER.rank == 0
        else None
    )

    for crt_epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(crt_epoch)

        def log_train_step(term_floats: dict[str, float]) -> None:
            nonlocal global_step
            global_step += 1
            if DIST_WRAPPER.rank != 0 or global_step % wandb_log_interval != 0:
                return
            payload = _prefix_metrics("train", _select_log_keys(term_floats, _TRAIN_LOG_KEYS))
            payload["train/lr"] = float(optimizer.param_groups[0]["lr"])
            _wandb_log(payload, step=global_step)

        train_iter = train_loader
        if DIST_WRAPPER.rank == 0:
            train_iter = tqdm(
                train_loader,
                desc="Train",
                total=len(train_loader),
                leave=True,
                position=1,
            )
        train_metrics = train_epoch(
            model=model,
            loader=train_iter,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
            device=device,
            max_grad_norm=float(args.optimizer.max_grad_norm),
            grad_accum_steps=max(1, int(args.optimizer.grad_accum_steps)),
            scheduler=scheduler,
            step_logger=log_train_step,
            contact_bins=n_contact_bins,
            inter_neg_per_pos=float(_cfg_get(args, "loss.inter_neg_per_pos", default=4.0)),
            dump_dir=logging_dir,
            epoch=crt_epoch,
            chunk_size=chunk_size,
            ema_wrapper=ema_wrapper,
        )

        test_iter = test_loader
        if DIST_WRAPPER.rank == 0:
            test_iter = tqdm(
                test_loader,
                desc="Test",
                total=len(test_loader),
                leave=True,
                position=1,
            )
        if ema_wrapper is not None:
            ema_wrapper.apply_shadow()
        test_metrics = evaluate_epoch(
            model=model,
            loader=test_iter,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
            device=device,
            contact_bins=n_contact_bins,
            positive_weight=float(_cfg_get(args, "metrics.positive_weight", default=1.0)),
            chunk_size=chunk_size,
        )
        if ema_wrapper is not None:
            ema_wrapper.restore()

        if DIST_WRAPPER.rank == 0:
            if epoch_progress is not None:
                epoch_progress.set_postfix(
                    loss=f"{train_metrics.get('total', float('nan')):.3f}",
                    test=f"{test_metrics.get('total', float('nan')):.3f}",
                    pAUC=f"{test_metrics.get('protein_auroc', float('nan')):.3f}",
                    pAP=f"{test_metrics.get('protein_auprc', float('nan')):.3f}",
                    rAUC=f"{test_metrics.get('residue_auroc', float('nan')):.3f}",
                    rAP=f"{test_metrics.get('residue_auprc', float('nan')):.3f}",
                    iAUC=f"{test_metrics.get('inter_auroc', float('nan')):.3f}",
                    iAP=f"{test_metrics.get('inter_auprc', float('nan')):.3f}",
                )
                epoch_progress.update()

            logged_train = _select_log_keys(train_metrics, _TRAIN_LOG_KEYS)
            logged_test = _select_log_keys(test_metrics, _TEST_LOG_KEYS)
            row = {"epoch": crt_epoch}
            row.update({f"train_{name}": value for name, value in logged_train.items()})
            row.update({f"test_{name}": value for name, value in logged_test.items()})
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(
                    ",".join(
                        "" if name not in row else f"{row[name]}"
                        for name in csv_fields
                    )
                    + "\n"
                )
            _wandb_log(
                _prefix_metrics("test", logged_test),
                step=max(global_step, 1),
            )
            log_info(
                f"[epoch={crt_epoch}] "
                + " ".join(
                    f"train/{name}={logged_train.get(name, float('nan')):.4f}"
                    for name in _TRAIN_LOG_KEYS
                    if name in logged_train
                )
                + " "
                + " ".join(
                    f"test/{name}={logged_test.get(name, float('nan')):.4f}"
                    for name in _TEST_LOG_KEYS
                )
            )

            if crt_epoch % args.checkpoint_interval == 0 or crt_epoch == args.epochs:
                checkpoint_path = os.path.join(logging_dir, f"checkpoints/epoch_{crt_epoch}.pth")
                torch.save(
                    {
                        "epoch": crt_epoch,
                        "model_state_dict": _live_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    checkpoint_path,
                )
                if ema_wrapper is not None:
                    ema_wrapper.apply_shadow()
                    ema_path = os.path.join(
                        logging_dir,
                        f"checkpoints/_ema_{ema_wrapper.decay}_{crt_epoch}.pth",
                    )
                    torch.save(
                        {"model_state_dict": _live_model(model).state_dict()},
                        ema_path,
                    )
                    ema_wrapper.restore()

    if wandb is not None and wandb.run is not None:
        wandb.finish()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
