"""SurPass training entrypoint.

The primary complex score is derived from the rectangular inter-chain contact
map. Training and validation therefore optimize and select checkpoints using
the same score used by inference.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any, Callable

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None

from src.data.balancing import DistributedWeightedSampler, build_multi_source_split
from src.data.collate import collate_fn, get_dataloader
from src.model.ema import EMAWrapper
from src.model.loss import SurpassLoss
from src.model.optimizer import get_lr_scheduler, get_optimizer
from src.model.surpass import ResOnly
from src.runtime.checkpoint import load_checkpoint, save_checkpoint
from src.runtime.config import build_model_kwargs, cfg_get
from src.runtime.device import (
    autocast_context,
    resolve_cuda_device,
    resolve_dist_backend,
    to_device,
)
from src.runtime.metrics import binary_classification_metrics
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything


TRAIN_KEYS = ("total", "protein", "contact", "distance", "aux_protein")
VALID_KEYS = ("total", "pair_auroc", "pair_auprc")


def log_info(message: str) -> None:
    if DIST_WRAPPER.rank == 0:
        print(message)


def _live_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _mean_terms(sums: dict[str, float], count: int) -> dict[str, float]:
    denominator = max(1, int(count))
    return {name: value / denominator for name, value in sums.items()}


def _add_terms(sums: dict[str, float], values: dict[str, torch.Tensor]) -> None:
    for name, value in values.items():
        sums[name] = sums.get(name, 0.0) + float(value.detach().item())


def _gather_validation(
    term_sums: dict[str, float],
    step_count: int,
    scores: list[float],
    targets: list[bool],
) -> tuple[dict[str, float], list[float], list[bool]]:
    payload = {
        "term_sums": term_sums,
        "step_count": int(step_count),
        "scores": scores,
        "targets": targets,
    }
    gathered = DIST_WRAPPER.all_gather_object(payload)
    merged_sums: dict[str, float] = {}
    merged_steps = 0
    merged_scores: list[float] = []
    merged_targets: list[bool] = []
    for rank_payload in gathered:
        merged_steps += int(rank_payload["step_count"])
        merged_scores.extend(rank_payload["scores"])
        merged_targets.extend(rank_payload["targets"])
        for name, value in rank_payload["term_sums"].items():
            merged_sums[name] = merged_sums.get(name, 0.0) + float(value)
    return _mean_terms(merged_sums, merged_steps), merged_scores, merged_targets


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_module: SurpassLoss,
    device: torch.device,
    *,
    precision: str,
    grad_accum_steps: int,
    max_grad_norm: float,
    ema: EMAWrapper | None,
    step_logger: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = {}
    steps = 0
    accumulation = max(1, int(grad_accum_steps))
    for step, (residue_batch, labels) in enumerate(loader):
        residue_batch = to_device(residue_batch, device)
        labels = to_device(labels, device)
        with autocast_context(device, precision):
            output = model(residue_batch)
            raw_loss, terms = loss_module(output, labels)
            loss = raw_loss / accumulation
        loss.backward()

        should_step = (step + 1) % accumulation == 0 or step + 1 == len(loader)
        if should_step:
            if float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update()

        _add_terms(totals, terms)
        steps += 1
        floats = {name: float(value.item()) for name, value in terms.items()}
        if hasattr(loader, "set_postfix"):
            loader.set_postfix(
                loss=f"{floats['total']:.3f}",
                contact=f"{floats['contact']:.3f}",
            )
        if step_logger is not None:
            step_logger(floats)
    return _mean_terms(totals, steps)


def evaluate_epoch(
    model: torch.nn.Module,
    loader,
    loss_module: SurpassLoss,
    device: torch.device,
    *,
    precision: str,
    positive_weight: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    steps = 0
    scores: list[float] = []
    targets: list[bool] = []
    with torch.inference_mode():
        for residue_batch, labels in loader:
            residue_batch = to_device(residue_batch, device)
            labels = to_device(labels, device)
            with autocast_context(device, precision):
                output = model(residue_batch)
                _, terms = loss_module(output, labels)
            _add_terms(totals, terms)
            steps += 1
            scores.extend(
                torch.sigmoid(output.protein_logit.float()).cpu().tolist()
            )
            targets.extend(labels["crop_is_positive"].bool().cpu().tolist())

    means, scores, targets = _gather_validation(totals, steps, scores, targets)
    metrics = binary_classification_metrics(
        scores,
        targets,
        positive_weight=float(positive_weight),
    )
    return {
        "total": means.get("total", float("nan")),
        "pair_auroc": metrics["auroc"],
        "pair_auprc": metrics["auprc"],
    }


def _init_distributed(use_cuda: bool) -> None:
    if DIST_WRAPPER.world_size <= 1 or dist.is_initialized():
        return
    timeout = datetime.timedelta(
        seconds=int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
    )
    dist.init_process_group(
        backend=resolve_dist_backend(use_cuda),
        timeout=timeout,
    )


def _build_datasets(args: DictConfig):
    sources = cfg_get(args, "data.datasets")
    if not sources:
        raise ValueError("data.datasets must list at least one source")
    sources = OmegaConf.to_container(sources, resolve=True)
    return (
        sources,
        *build_multi_source_split(
            sources,
            root_dir=str(args.data.root_dir),
            dataset_kwargs={
                "distance_bin_start": float(args.data.distance_bin_start),
                "distance_bin_end": float(args.data.distance_bin_end),
                "distance_bin_count": int(args.data.distance_bin_count),
                "crop_size": int(args.data.crop_size),
                "contact_threshold": float(args.data.contact_threshold),
                "interface_crop_probability": float(
                    cfg_get(args, "data.interface_crop_probability", 0.8)
                ),
            },
        ),
    )


def _ema_keywords(args: DictConfig) -> list[str]:
    value = cfg_get(args, "ema.mutable_param_keywords", [""])
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return list(value or [""])


def _wandb_init(args: DictConfig, output_dir: Path):
    if DIST_WRAPPER.rank != 0 or not bool(cfg_get(args, "wandb.enabled", False)):
        return None
    if wandb is None:
        log_info("wandb is enabled but not installed; continuing without it")
        return None
    return wandb.init(
        project=str(cfg_get(args, "wandb.project", "surpass")),
        entity=cfg_get(args, "wandb.entity"),
        name=cfg_get(args, "wandb.name", output_dir.name),
        dir=str(output_dir),
        mode=str(cfg_get(args, "wandb.mode", "online")),
        config=OmegaConf.to_container(args, resolve=True),
        tags=list(cfg_get(args, "wandb.tags", []) or []),
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(args: DictConfig) -> None:
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
    precision = str(cfg_get(args, "runtime.precision", "bfloat16"))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.logging_dir) / f"{str(args.task_prefix).upper()}_{timestamp}"
    if DIST_WRAPPER.rank == 0:
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(args, output_dir / "config.yaml")
    wandb_run = _wandb_init(args, output_dir)

    sources, train_dataset, validation_dataset, train_weights, source_info = (
        _build_datasets(args)
    )
    if len(sources) == 1:
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
        train_sampler = DistributedWeightedSampler(
            train_weights,
            num_samples=int(cfg_get(args, "data.epoch_size", 30000)),
            num_replicas=DIST_WRAPPER.world_size,
            rank=DIST_WRAPPER.rank,
            seed=int(args.seed),
        )

    validation_indices = range(
        DIST_WRAPPER.rank,
        len(validation_dataset),
        DIST_WRAPPER.world_size,
    )
    validation_shard = Subset(validation_dataset, list(validation_indices))
    train_loader = get_dataloader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=int(args.data.batch_size),
        shuffle=train_sampler is None,
        num_workers=int(args.data.num_workers),
        sampler=train_sampler,
        pin_memory=bool(args.data.pin_memory),
    )
    validation_loader = get_dataloader(
        validation_shard,
        collate_fn=collate_fn,
        batch_size=int(args.data.batch_size),
        shuffle=False,
        num_workers=int(args.data.num_workers),
        pin_memory=bool(args.data.pin_memory),
    )
    for info in source_info:
        log_info(
            f"source={info['name']} train={info['train_length']} "
            f"validation={info['validation_length']}"
        )

    model = ResOnly(**build_model_kwargs(args)).to(device)
    if DIST_WRAPPER.world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    live_model = _live_model(model)
    log_info(
        f"architecture={live_model.architecture} "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    loss_weights = {
        name: float(cfg_get(args, f"loss.{name}.weight", default))
        for name, default in (
            ("protein", 1.0),
            ("contact", 1.0),
            ("distance", 0.1),
            ("aux_protein", 0.1),
        )
    }
    loss_module = SurpassLoss(
        live_model.contact_bins,
        loss_weights,
        focal_gamma=float(cfg_get(args, "loss.contact.gamma", 1.5)),
        contact_positive_weight=float(
            cfg_get(args, "loss.contact.positive_weight", 4.0)
        ),
        negatives_per_positive=float(
            cfg_get(args, "loss.contact.negatives_per_positive", 8.0)
        ),
    ).to(device)
    optimizer = get_optimizer(
        model,
        lr=float(args.optimizer.lr),
        weight_decay=float(args.optimizer.weight_decay),
        betas=(float(args.optimizer.beta1), float(args.optimizer.beta2)),
        use_adamw=bool(args.optimizer.use_adamw),
    )
    scheduler = get_lr_scheduler(
        optimizer,
        lr_scheduler=str(args.optimizer.lr_scheduler),
        lr=float(args.optimizer.lr),
        max_steps=int(args.epochs) * len(train_loader) + 100,
        warmup_steps=int(args.optimizer.warmup_steps),
        decay_every_n_steps=int(args.optimizer.decay_every_n_steps),
        decay_factor=float(args.optimizer.decay_factor),
    )
    ema = None
    ema_decay = float(cfg_get(args, "ema.decay", 0.0))
    if ema_decay > 0:
        ema = EMAWrapper(model, ema_decay, _ema_keywords(args))
        ema.register()

    start_epoch = 1
    if args.ckpt_dir:
        payload = load_checkpoint(
            args.ckpt_dir,
            live_model,
            map_location=device,
            optimizer=optimizer,
            scheduler=scheduler,
            load_training_state=not bool(args.load_model_only),
        )
        if not bool(args.load_model_only):
            start_epoch = int(payload["epoch"]) + 1
    if args.ema_dir and ema is not None:
        load_checkpoint(args.ema_dir, live_model, map_location=device)
        ema.register()

    csv_path = output_dir / "metrics.csv"
    if DIST_WRAPPER.rank == 0:
        csv_path.write_text(
            ",".join(
                ["epoch"]
                + [f"train_{key}" for key in TRAIN_KEYS]
                + [f"validation_{key}" for key in VALID_KEYS]
            )
            + "\n",
            encoding="utf-8",
        )

    best_auprc = float("-inf")
    global_step = 0
    for epoch in range(start_epoch, int(args.epochs) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_iter = (
            tqdm(train_loader, desc=f"Train {epoch}")
            if DIST_WRAPPER.rank == 0
            else train_loader
        )

        def log_step(values: dict[str, float]) -> None:
            nonlocal global_step
            global_step += 1
            if wandb_run is not None:
                wandb.log(
                    {f"train/{key}": value for key, value in values.items()},
                    step=global_step,
                )

        train_metrics = train_epoch(
            model,
            train_iter,
            optimizer,
            scheduler,
            loss_module,
            device,
            precision=precision,
            grad_accum_steps=int(args.optimizer.grad_accum_steps),
            max_grad_norm=float(args.optimizer.max_grad_norm),
            ema=ema,
            step_logger=log_step,
        )
        if ema is not None:
            ema.apply_shadow()
        validation_metrics = evaluate_epoch(
            model,
            validation_loader,
            loss_module,
            device,
            precision=precision,
            positive_weight=float(cfg_get(args, "metrics.positive_weight", 1.0)),
        )
        if ema is not None:
            ema.restore()

        if DIST_WRAPPER.rank == 0:
            row = [str(epoch)]
            row.extend(str(train_metrics.get(key, "")) for key in TRAIN_KEYS)
            row.extend(str(validation_metrics.get(key, "")) for key in VALID_KEYS)
            with csv_path.open("a", encoding="utf-8") as handle:
                handle.write(",".join(row) + "\n")
            log_info(
                f"epoch={epoch} train_loss={train_metrics['total']:.4f} "
                f"pair_auprc={validation_metrics['pair_auprc']:.4f}"
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        f"validation/{key}": value
                        for key, value in validation_metrics.items()
                    },
                    step=max(global_step, 1),
                )

            should_save = epoch % int(args.checkpoint_interval) == 0 or epoch == int(
                args.epochs
            )
            if should_save:
                save_checkpoint(
                    output_dir / "checkpoints" / f"epoch_{epoch}.pth",
                    live_model,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
            if validation_metrics["pair_auprc"] > best_auprc:
                best_auprc = validation_metrics["pair_auprc"]
                if ema is not None:
                    ema.apply_shadow()
                save_checkpoint(
                    output_dir / "checkpoints" / "best_pair_auprc.pth",
                    live_model,
                    epoch=epoch,
                )
                if ema is not None:
                    ema.restore()

    if wandb_run is not None:
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
