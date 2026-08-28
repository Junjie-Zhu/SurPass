import datetime
import os
import warnings
from typing import Any, Callable, Dict

import hydra
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None

from src.data.dataset import (
    BalancedClusterDataset,
    PepoTrainDataset,
    collate_fn,
    get_dataloader,
    inter_chain_pair_mask,
)
from src.model.loss import (
    FocalCELoss,
    DistogramMAELoss,
    TverskyLoss,
)
from src.model.optimizer import get_lr_scheduler, get_optimizer
from src.model.surpass import ResOnly
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


def split_dataset(
    dataset: Dataset,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if len(dataset) == 0:
        raise ValueError("Cannot split an empty dataset.")
    if len(dataset) == 1:
        return dataset, dataset

    test_len = max(1, int(round(len(dataset) * test_fraction)))
    train_len = len(dataset) - test_len
    if train_len == 0:
        train_len, test_len = 1, len(dataset) - 1
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, test_len], generator=generator)


def _as_balanced_cluster_dataset(
    dataset: Dataset,
    negative_ratio: int,
    distance_bin_count: int,
) -> Dataset:
    if negative_ratio <= 0:
        return dataset
    if isinstance(dataset, Subset) and isinstance(dataset.dataset, PepoTrainDataset):
        if len(dataset.indices) < 2:
            return dataset
        return BalancedClusterDataset(
            dataset.dataset,
            indices=list(dataset.indices),
            distance_bin_count=distance_bin_count,
            negative_ratio=negative_ratio,
        )
    if isinstance(dataset, PepoTrainDataset):
        if len(dataset) < 2:
            return dataset
        return BalancedClusterDataset(
            dataset,
            distance_bin_count=distance_bin_count,
            negative_ratio=negative_ratio,
        )
    raise TypeError(f"Unsupported dataset type for negative sampling: {type(dataset)}")


def create_balanced_split_datasets(
    dataset: PepoTrainDataset,
    test_fraction: float,
    seed: int,
    negative_ratio: int,
    distance_bin_count: int,
) -> tuple[Dataset, Dataset]:
    train_dataset, test_dataset = split_dataset(dataset, test_fraction=test_fraction, seed=seed)
    return (
        _as_balanced_cluster_dataset(train_dataset, negative_ratio, distance_bin_count),
        _as_balanced_cluster_dataset(test_dataset, negative_ratio, distance_bin_count),
    )


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


def resolve_train_recycle_rounds(
    recycle_rounds: int,
    self_conditioning_probability: float,
    random_value: float | None = None,
) -> int:
    recycle_rounds = max(1, int(recycle_rounds))
    if recycle_rounds <= 1:
        return 1

    probability = max(0.0, min(1.0, float(self_conditioning_probability)))
    if probability <= 0.0:
        return 1

    if random_value is None:
        random_value = float(torch.rand(()).item())
    return recycle_rounds if float(random_value) < probability else 1


def _binary_classification_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    scores = scores.detach().flatten().to(dtype=torch.float32).cpu().numpy()
    target_bool = target.detach().flatten().to(dtype=torch.bool).cpu().numpy()
    pos_count = int(target_bool.sum())
    neg_count = int(target_bool.size - pos_count)
    if scores.size == 0 or pos_count == 0 or neg_count == 0:
        return {"auroc": float("nan"), "auprc": float("nan")}
    fpr, tpr, _ = roc_curve(target_bool, scores)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(target_bool, scores)
    auprc = auc(recall, precision)
    return {"auroc": float(auroc), "auprc": float(auprc)}


def _region_masks(
    pair_mask: torch.Tensor,
    label_mask: torch.Tensor,
    p1_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = pair_mask.to(dtype=torch.bool) & label_mask.to(dtype=torch.bool)
    total_length = int(valid_mask.shape[-1])
    inter_mask = inter_chain_pair_mask(
        int(p1_length),
        total_length - int(p1_length),
        device=valid_mask.device,
    )
    if valid_mask.ndim == 3:
        inter_mask = inter_mask.unsqueeze(0)
    intra_mask = valid_mask & ~inter_mask
    inter_mask = valid_mask & inter_mask
    return intra_mask, inter_mask


def _compute_pair_losses(
    loss_fn: Dict[str, torch.nn.Module],
    loss_weights: Dict[str, float],
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    intra_mask: torch.Tensor,
    inter_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw_loss = logits.new_zeros(())
    terms: dict[str, torch.Tensor] = {}
    for name, loss_term in loss_fn.items():
        term_weight = float(loss_weights[name])
        if term_weight <= 0.0:
            continue
        if name == "tversky":
            value = loss_term(logits, target_bins, inter_mask)
            terms[name] = value.detach()
            raw_loss = raw_loss + term_weight * value
            continue
        intra_loss, inter_loss = loss_term(logits, target_bins, intra_mask, inter_mask)
        terms[f"{name}_intra"] = intra_loss.detach()
        terms[f"{name}_inter"] = inter_loss.detach()
        weighted = (
            float(loss_weights["intra"]) * intra_loss
            + float(loss_weights["inter"]) * inter_loss
        )
        terms[name] = weighted.detach()
        raw_loss = raw_loss + term_weight * weighted
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


def _progress_postfix(terms: dict[str, float]) -> dict[str, str]:
    aliases = (
        ("loss", "total"),
        ("foc_i", "focal_intra"),
        ("foc_x", "focal_inter"),
        ("mae_x", "mae_inter"),
        ("tv", "tversky"),
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
    }


def _unpack_batch(step_batch, device):
    p1_batch, p2_batch, labels = step_batch
    p1_batch = to_device(p1_batch, device)
    p2_batch = to_device(p2_batch, device)
    labels = to_device(labels, device)
    return p1_batch, p2_batch, labels


def _set_progress_postfix(progress, **kwargs) -> None:
    set_postfix = getattr(progress, "set_postfix", None)
    if callable(set_postfix):
        set_postfix(**kwargs)


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
    recycle_rounds: int = 2,
    self_conditioning_probability: float = 0.5,
    step_logger: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    term_totals: dict[str, float] = {}
    num_steps = 0

    for step, step_batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        p1_batch, p2_batch, labels = _unpack_batch(step_batch, device)

        step_recycle_rounds = resolve_train_recycle_rounds(
            recycle_rounds=recycle_rounds,
            self_conditioning_probability=self_conditioning_probability,
        )
        logits, pair_mask = model(
            p1_batch,
            p2_batch,
            recycle_rounds=step_recycle_rounds,
        )
        intra_mask, inter_mask = _region_masks(
            pair_mask,
            labels["label_2d_mask"],
            p1_length=int(p1_batch["mask"].shape[1]),
        )
        raw_loss, terms = _compute_pair_losses(
            loss_fn,
            loss_weights,
            logits,
            labels["label_2d_bins"],
            intra_mask,
            inter_mask,
        )

        loss = raw_loss / max(1, grad_accum_steps)
        loss.backward()

        should_step = ((step + 1) % max(1, grad_accum_steps) == 0) or (
            step + 1 == len(loader)
        )
        if should_step:
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
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
    metrics = _binary_classification_metrics(all_scores, all_targets)
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
    contact_threshold: float,
    distance_bin_start: float,
    distance_bin_end: float,
    distance_bin_count: int = 64,
    max_batches: int | None = None,
    recycle_rounds: int = 2,
) -> dict[str, float]:
    model.eval()
    term_totals: dict[str, float] = {}
    num_steps = 0
    inter_score_rows = []
    inter_target_rows = []
    intra_score_rows = []
    intra_target_rows = []
    n_contact_bins = contact_bin_count(
        contact_threshold,
        distance_bin_start,
        distance_bin_end,
        distance_bin_count,
    )

    with torch.no_grad():
        for step, step_batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            p1_batch, p2_batch, labels = _unpack_batch(step_batch, device)
            logits, pair_mask = model(
                p1_batch,
                p2_batch,
                recycle_rounds=max(1, int(recycle_rounds)),
            )
            intra_mask, inter_mask = _region_masks(
                pair_mask,
                labels["label_2d_mask"],
                p1_length=int(p1_batch["mask"].shape[1]),
            )
            _, terms = _compute_pair_losses(
                loss_fn,
                loss_weights,
                logits,
                labels["label_2d_bins"],
                intra_mask,
                inter_mask,
            )

            probs = torch.softmax(logits, dim=-1)[..., :n_contact_bins].sum(dim=-1)
            contacts = labels["label_2d_bins"] < n_contact_bins
            inter_score_rows.append(probs[inter_mask].detach().cpu())
            inter_target_rows.append(contacts[inter_mask].detach().cpu())
            intra_score_rows.append(probs[intra_mask].detach().cpu())
            intra_target_rows.append(contacts[intra_mask].detach().cpu())

            term_floats = _terms_to_floats(terms)
            _accumulate_floats(term_totals, term_floats)
            num_steps += 1
            _set_progress_postfix(loader, **_progress_postfix(term_floats))

    inter_metrics = _contact_metrics_from_rows(inter_score_rows, inter_target_rows)
    intra_metrics = _contact_metrics_from_rows(intra_score_rows, intra_target_rows)
    metrics = _mean_floats(term_totals, num_steps)
    metrics.update(
        {
            "inter_auroc": inter_metrics["auroc"],
            "inter_auprc": inter_metrics["auprc"],
            "inter_contact_prev": inter_metrics["contact_prev"],
            "intra_auroc": intra_metrics["auroc"],
            "intra_auprc": intra_metrics["auprc"],
            "intra_contact_prev": intra_metrics["contact_prev"],
        }
    )
    return metrics


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

    full_dataset = PepoTrainDataset(
        root_dir=args.data.root_dir,
        cluster_tsv_path=args.data.cluster_tsv_path,
        center_coordinates=args.data.center_coordinates,
        random_rotation=args.data.random_rotation,
        distance_bin_start=args.data.distance_bin_start,
        distance_bin_end=args.data.distance_bin_end,
        distance_bin_count=args.data.distance_bin_count,
        crop_size=args.data.crop_size,
        contact_threshold=args.data.contact_threshold,
    )
    train_dataset, test_dataset = create_balanced_split_datasets(
        full_dataset,
        test_fraction=args.data.test_fraction,
        seed=args.seed,
        negative_ratio=int(_cfg_get(args, "data.negative_ratio", default=1)),
        distance_bin_count=args.data.distance_bin_count,
    )
    log_info(
        f"Loaded {len(full_dataset)} clusters: {len(train_dataset)} train, {len(test_dataset)} test"
    )

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

    model_kwargs = _cfg_get(args, "model", default={})
    if isinstance(model_kwargs, DictConfig):
        model_kwargs = OmegaConf.to_container(model_kwargs, resolve=True)
    model_kwargs = dict(model_kwargs or {})
    model_kwargs.setdefault("num_classes", args.data.distance_bin_count)
    model = ResOnly(**model_kwargs).to(device)
    if DIST_WRAPPER.world_size > 1:
        model = wrap_ddp(model, device)
    log_info(
        f"Model instantiated with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    n_contact_bins = contact_bin_count(
        args.data.contact_threshold,
        args.data.distance_bin_start,
        args.data.distance_bin_end,
        args.data.distance_bin_count,
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
    loss_fn = {
        "focal": focal_loss,
        "mae": mae_loss,
        "tversky": tversky_loss,
    }
    loss_weights = _build_loss_weights(args)
    log_info(
        f"Loss setup: contact_bins={n_contact_bins}, weights={loss_weights}"
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
    if args.ckpt_dir is not None:
        checkpoint = torch.load(args.ckpt_dir, map_location=device)
        target_model = model.module if DIST_WRAPPER.world_size > 1 else model
        target_model.load_state_dict(checkpoint["model_state_dict"])
        if not args.load_model_only:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
        del checkpoint

    csv_path = os.path.join(logging_dir, "loss.csv")
    csv_fields = [
        "epoch",
        "train_total",
        "train_focal_intra",
        "train_focal_inter",
        "train_mae_intra",
        "train_mae_inter",
        "train_tversky",
        "test_total",
        "test_focal_intra",
        "test_focal_inter",
        "test_mae_intra",
        "test_mae_inter",
        "test_tversky",
        "test_inter_auroc",
        "test_inter_auprc",
        "test_inter_contact_prev",
        "test_intra_auroc",
        "test_intra_auprc",
        "test_intra_contact_prev",
    ]
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
            payload = _prefix_metrics("train", term_floats)
            payload["epoch"] = crt_epoch
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
            recycle_rounds=int(args.recycle_rounds),
            self_conditioning_probability=float(args.self_conditioning_probability),
            step_logger=log_train_step,
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
        test_metrics = evaluate_epoch(
            model=model,
            loader=test_iter,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
            device=device,
            contact_threshold=args.data.contact_threshold,
            distance_bin_start=args.data.distance_bin_start,
            distance_bin_end=args.data.distance_bin_end,
            distance_bin_count=args.data.distance_bin_count,
            recycle_rounds=int(args.recycle_rounds),
        )

        if DIST_WRAPPER.rank == 0:
            if epoch_progress is not None:
                epoch_progress.set_postfix(
                    loss=f"{train_metrics.get('total', float('nan')):.3f}",
                    test=f"{test_metrics.get('total', float('nan')):.3f}",
                    iAUC=f"{test_metrics.get('inter_auroc', float('nan')):.3f}",
                    oAUC=f"{test_metrics.get('intra_auroc', float('nan')):.3f}",
                )
                epoch_progress.update()

            row = {"epoch": crt_epoch}
            row.update({f"train_{name}": value for name, value in train_metrics.items()})
            row.update({f"test_{name}": value for name, value in test_metrics.items()})
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(
                    ",".join(
                        "" if name not in row else f"{row[name]}"
                        for name in csv_fields
                    )
                    + "\n"
                )
            _wandb_log(
                {
                    "epoch": crt_epoch,
                    **_prefix_metrics("test", test_metrics),
                },
                step=max(global_step, 1),
            )
            log_info(
                f"[epoch={crt_epoch}] "
                f"train={train_metrics.get('total', float('nan')):.4f} "
                f"test={test_metrics.get('total', float('nan')):.4f} "
                f"focal_x={train_metrics.get('focal_inter', float('nan')):.4f} "
                f"mae_x={train_metrics.get('mae_inter', float('nan')):.4f} "
                f"tv={train_metrics.get('tversky', float('nan')):.4f} "
                f"inter_auroc={test_metrics.get('inter_auroc', float('nan')):.4f} "
                f"inter_auprc={test_metrics.get('inter_auprc', float('nan')):.4f} "
                f"intra_auroc={test_metrics.get('intra_auroc', float('nan')):.4f}"
            )

            if crt_epoch % args.checkpoint_interval == 0 or crt_epoch == args.epochs:
                checkpoint_path = os.path.join(logging_dir, f"checkpoints/epoch_{crt_epoch}.pth")
                torch.save(
                    {
                        "epoch": crt_epoch,
                        "model_state_dict": (
                            model.module.state_dict()
                            if DIST_WRAPPER.world_size > 1
                            else model.state_dict()
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    checkpoint_path,
                )

    if wandb is not None and wandb.run is not None:
        wandb.finish()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
