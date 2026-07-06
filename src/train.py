import datetime
import os
import warnings
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.data.dataset import BalancedClusterDataset, PepoTrainDataset, collate_fn, get_dataloader
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
    distance_bin_width: float,
    distance_bin_count: int,
) -> int:
    count = int((contact_threshold - distance_bin_start) / distance_bin_width)
    return max(1, min(count, distance_bin_count))


def masked_cross_entropy(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: torch.nn.Module,
) -> torch.Tensor:
    if logits.shape[:-1] != target_bins.shape:
        raise ValueError(
            f"Logit/target shape mismatch: logits={tuple(logits.shape)} "
            f"target={tuple(target_bins.shape)}"
        )
    loss = loss_fn(logits.permute(0, 3, 1, 2), target_bins.long())
    mask_float = mask.to(dtype=loss.dtype)
    return (loss * mask_float).sum() / mask_float.sum().clamp_min(1.0)


def _binary_classification_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    scores = scores.detach().flatten().to(dtype=torch.float32).cpu()
    target_bool = target.detach().flatten().to(dtype=torch.bool).cpu()
    pos_count = int(target_bool.sum().item())
    neg_count = int((~target_bool).sum().item())
    if pos_count == 0 or neg_count == 0:
        return {"auroc": float("nan"), "auprc": float("nan")}

    order = torch.argsort(scores, descending=True)
    y_sorted = target_bool[order].to(dtype=torch.float32)
    tps = torch.cumsum(y_sorted, dim=0)
    fps = torch.cumsum(1.0 - y_sorted, dim=0)

    tpr = torch.cat([torch.tensor([0.0]), tps / pos_count, torch.tensor([1.0])])
    fpr = torch.cat([torch.tensor([0.0]), fps / neg_count, torch.tensor([1.0])])
    auroc = torch.trapz(tpr, fpr).item()

    precision_curve = tps / torch.arange(1, len(y_sorted) + 1, dtype=torch.float32)
    recall_curve = tps / pos_count
    recall_curve = torch.cat([torch.tensor([0.0]), recall_curve])
    precision_curve = torch.cat([torch.tensor([1.0]), precision_curve])
    auprc = torch.sum(
        (recall_curve[1:] - recall_curve[:-1]) * precision_curve[1:]
    ).item()

    return {"auroc": float(auroc), "auprc": float(auprc)}


def _contact_scores_and_targets(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    mask: torch.Tensor,
    contact_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.to(dtype=torch.bool)
    probs = torch.softmax(logits, dim=-1)[..., :contact_bins].sum(dim=-1)
    contacts = target_bins < contact_bins
    return probs[valid], contacts[valid]


def _unpack_batch(step_batch, device):
    p1_batch, p2_batch, labels = step_batch
    p1_batch = to_device(p1_batch, device)
    p2_batch = to_device(p2_batch, device)
    labels = to_device(labels, device)
    return p1_batch, p2_batch, labels


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    max_grad_norm: float = 0.0,
    grad_accum_steps: int = 1,
    scheduler: Any | None = None,
    max_batches: int | None = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    num_steps = 0

    for step, step_batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        p1_batch, p2_batch, labels = _unpack_batch(step_batch, device)
        logits, pair_mask = model(p1_batch, p2_batch)
        valid_mask = pair_mask.to(dtype=torch.bool) & labels["label_2d_mask"].to(dtype=torch.bool)
        raw_loss = masked_cross_entropy(
            logits,
            labels["label_2d_bins"],
            valid_mask,
            loss_fn,
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

        total_loss += raw_loss.item()
        num_steps += 1

    return total_loss / max(num_steps, 1)


def evaluate_epoch(
    model: torch.nn.Module,
    loader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    contact_threshold: float,
    distance_bin_start: float,
    distance_bin_width: float,
    distance_bin_count: int = 36,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_steps = 0
    score_rows = []
    target_rows = []
    n_contact_bins = contact_bin_count(
        contact_threshold,
        distance_bin_start,
        distance_bin_width,
        distance_bin_count,
    )

    with torch.no_grad():
        for step, step_batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            p1_batch, p2_batch, labels = _unpack_batch(step_batch, device)
            logits, pair_mask = model(p1_batch, p2_batch)
            valid_mask = (
                pair_mask.to(dtype=torch.bool) & labels["label_2d_mask"].to(dtype=torch.bool)
            )
            loss = masked_cross_entropy(
                logits,
                labels["label_2d_bins"],
                valid_mask,
                loss_fn,
            )
            scores, targets = _contact_scores_and_targets(
                logits,
                labels["label_2d_bins"],
                valid_mask,
                n_contact_bins,
            )
            total_loss += loss.item()
            num_steps += 1
            score_rows.append(scores.detach().cpu())
            target_rows.append(targets.detach().cpu())

    if score_rows:
        metrics = _binary_classification_metrics(
            torch.cat(score_rows, dim=0),
            torch.cat(target_rows, dim=0),
        )
    else:
        metrics = {"auroc": float("nan"), "auprc": float("nan")}

    return {
        "test_loss": total_loss / max(num_steps, 1),
        "test_auroc": metrics["auroc"],
        "test_auprc": metrics["auprc"],
    }


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
        device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if DIST_WRAPPER.world_size > 1:
        if DIST_WRAPPER.rank == 0:
            log_info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            log_info(
                f"Using DDP with {DIST_WRAPPER.world_size} processes, rank: {DIST_WRAPPER.rank}"
            )
        timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
        )

    seed_everything(seed=args.seed, deterministic=args.deterministic)

    full_dataset = PepoTrainDataset(
        root_dir=args.data.root_dir,
        cluster_tsv_path=args.data.cluster_tsv_path,
        center_coordinates=args.data.center_coordinates,
        random_rotation=args.data.random_rotation,
        distance_bin_start=args.data.distance_bin_start,
        distance_bin_width=args.data.distance_bin_width,
        distance_bin_count=args.data.distance_bin_count,
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
        model = DDP(
            model,
            device_ids=[DIST_WRAPPER.local_rank],
            output_device=DIST_WRAPPER.local_rank,
            find_unused_parameters=False,
        )
    log_info(
        f"Model instantiated with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none").to(device)
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

    if DIST_WRAPPER.rank == 0:
        with open(f"{logging_dir}/loss.csv", "w", encoding="utf-8") as f:
            f.write("Epoch,Train Loss,Test Loss,Test AUROC,Test AUPRC\n")

    epoch_progress = (
        tqdm(total=args.epochs, leave=False, position=0)
        if DIST_WRAPPER.rank == 0
        else None
    )

    for crt_epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(crt_epoch)

        train_iter = train_loader
        if DIST_WRAPPER.rank == 0:
            train_iter = tqdm(
                train_loader,
                desc="Train",
                total=len(train_loader),
                leave=True,
                position=1,
            )
        train_loss = train_epoch(
            model=model,
            loader=train_iter,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            max_grad_norm=float(args.optimizer.max_grad_norm),
            grad_accum_steps=max(1, int(args.optimizer.grad_accum_steps)),
            scheduler=scheduler,
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
        metrics = evaluate_epoch(
            model=model,
            loader=test_iter,
            loss_fn=loss_fn,
            device=device,
            contact_threshold=args.data.contact_threshold,
            distance_bin_start=args.data.distance_bin_start,
            distance_bin_width=args.data.distance_bin_width,
            distance_bin_count=args.data.distance_bin_count,
        )

        if DIST_WRAPPER.rank == 0 and epoch_progress is not None:
            epoch_progress.set_postfix(
                loss=f"{train_loss:.3f}",
                test_loss=f"{metrics['test_loss']:.3f}",
                auroc=f"{metrics['test_auroc']:.4f}",
                auprc=f"{metrics['test_auprc']:.4f}",
            )
            epoch_progress.update()
            with open(f"{logging_dir}/loss.csv", "a", encoding="utf-8") as f:
                f.write(
                    f"{crt_epoch},{train_loss},{metrics['test_loss']},"
                    f"{metrics['test_auroc']},{metrics['test_auprc']}\n"
                )
            log_info(
                f"[test-metrics][epoch={crt_epoch}] "
                f"loss={metrics['test_loss']:.4f} "
                f"auroc={metrics['test_auroc']:.4f} "
                f"auprc={metrics['test_auprc']:.4f}"
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

    if DIST_WRAPPER.world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
