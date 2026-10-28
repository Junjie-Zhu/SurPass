"""SurPass training entrypoint.

The primary complex score is derived from the rectangular inter-chain contact
map. The protein loss uses configured pooling; validation ranking and checkpoint
selection use the same maximum-contact score as inference.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import hydra
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
from src.data.validation_sampling import DistributedValidationSampler, validation_pools
from src.data.fixed_validation import FrozenValidationDataset, UnpaddedShardSampler
from src.runtime.state import capture_rng, restore_rng, preserve_rng
from src.runtime.epoch_inference import inference_due, run_epoch_inference
from src.model.ema import EMAWrapper
from src.model.loss import SurpassLoss, max_contact_probability, pair_contact_probability
from src.model.optimizer import get_lr_scheduler, get_optimizer
from src.runtime.checkpoint import load_checkpoint, save_checkpoint
from src.runtime.config import create_model, cfg_get
from src.runtime.device import (
    autocast_context,
    resolve_cuda_device,
    resolve_dist_backend,
    to_device,
)
from src.runtime.metrics import binary_classification_metrics
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything


TRAIN_KEYS = ("total", "protein", "contact", "distance", "aux_protein")
VALID_KEYS = ("total", "protein", "contact", "distance", "aux_protein", "pair_auroc", "pair_auprc", "pair_auprc_unweighted", "samples", "positive_samples", "negative_samples")


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
            group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            loss = raw_loss / group_size
        if not bool(torch.isfinite(raw_loss)):
            raise FloatingPointError("Nonfinite training loss")
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
            floats["lr"] = float(optimizer.param_groups[0]["lr"])
            floats["optimizer_updates"] = int(scheduler.last_epoch)
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
            batch_samples = int(labels["crop_is_positive"].numel())
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().item()) * batch_samples
            steps += batch_samples
            scores.extend(
                max_contact_probability(
                    pair_contact_probability(output.pair_logits, loss_module.contact.contact_bins),
                    output.pair_mask,
                ).cpu().tolist()
            )
            targets.extend(labels["crop_is_positive"].bool().cpu().tolist())

    means, scores, targets = _gather_validation(totals, steps, scores, targets)
    metrics = binary_classification_metrics(
        scores,
        targets,
        positive_weight=float(positive_weight),
    )
    return {
        **{key: means.get(key, float("nan")) for key in TRAIN_KEYS},
        "pair_auprc_unweighted": binary_classification_metrics(scores, targets, positive_weight=1.0)["auprc"],
        "pair_auroc": metrics["auroc"],
        "pair_auprc": metrics["auprc"],
        "samples": len(targets),
        "positive_samples": sum(targets),
        "negative_samples": len(targets) - sum(targets),
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
                "distance_bin_convention": str(cfg_get(args, "data.distance_bin_convention", "legacy")),
                "contact_threshold": float(args.data.contact_threshold),
            },
            split_dir=cfg_get(args, "data.split_dir"),
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
    if int(args.checkpoint_interval) < 1:
        raise ValueError("checkpoint_interval must be positive")
    if bool(cfg_get(args, "epoch_inference.enabled", False)) and float(cfg_get(args, "ema.decay", 0)) <= 0:
        raise ValueError("Periodic EMA inference requires EMA to be enabled")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.logging_dir) / f"{str(args.task_prefix).upper()}_{timestamp}"
    if dist.is_initialized():
        shared = [str(output_dir) if DIST_WRAPPER.rank == 0 else None]
        dist.broadcast_object_list(shared, src=0)
        output_dir = Path(shared[0])
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

    train_loader = get_dataloader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=int(args.data.batch_size),
        shuffle=train_sampler is None,
        num_workers=int(args.data.num_workers),
        sampler=train_sampler,
        pin_memory=bool(args.data.pin_memory),
    )
    for info in source_info:
        log_info(
            f"source={info['name']} train={info['train_length']} "
            f"validation={info['validation_length']}"
        )

    model = create_model(args).to(device)
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
        max_steps=int(args.epochs) * math.ceil(len(train_loader) / max(1, int(args.optimizer.grad_accum_steps))),
        min_lr=cfg_get(args, "optimizer.min_lr"),
        warmup_steps=int(args.optimizer.warmup_steps),
        decay_every_n_steps=int(args.optimizer.decay_every_n_steps),
        decay_factor=float(args.optimizer.decay_factor),
    )
    ema = None
    ema_decay = float(cfg_get(args, "ema.decay", 0.0))
    if ema_decay > 0:
        ema = EMAWrapper(live_model, ema_decay, _ema_keywords(args))
    start_epoch = 1
    resume_state = {}
    if args.ema_dir:
        raise ValueError("ema_dir is superseded by complete EMA checkpoints; use ckpt_dir")
    if args.ckpt_dir:
        payload = load_checkpoint(args.ckpt_dir, live_model, map_location="cpu",
                                  optimizer=optimizer, scheduler=scheduler, ema=ema,
                                  load_training_state=not bool(args.load_model_only))
        if not bool(args.load_model_only):
            start_epoch = int(payload["epoch"]) + 1
            resume_state = payload.get("training_state", {})
            if not resume_state:
                raise ValueError("Complete resume requires checkpoint training_state")
            if int(resume_state.get("world_size", DIST_WRAPPER.world_size)) != DIST_WRAPPER.world_size:
                raise ValueError("Training resume requires the saved world size")
        del payload
    if ema is not None and not ema.shadow:
        ema.register()

    fixed = bool(cfg_get(args, "data.fixed_validation", False))
    if fixed:
        panel = resume_state.get("validation_panel")
        panel_path = cfg_get(args, "data.validation_manifest")
        if panel is None and panel_path:
            panel = json.loads(Path(panel_path).read_text())
        if panel is None and DIST_WRAPPER.rank == 0:
            with preserve_rng():
                panel = FrozenValidationDataset.build(
                    validation_dataset, sources,
                    OmegaConf.select(args, "data.validation_samples", default=3000),
                    float(cfg_get(args, "data.validation_negative_ratio", 10.0)),
                    int(args.seed) + 100000).manifest
        if dist.is_initialized():
            objects = [panel]
            dist.broadcast_object_list(objects, src=0)
            panel = objects[0]
        validation_dataset = FrozenValidationDataset(validation_dataset, panel)
        validation_info = validation_dataset
        validation_sampler = UnpaddedShardSampler(validation_dataset, DIST_WRAPPER.rank, DIST_WRAPPER.world_size)
        if DIST_WRAPPER.rank == 0:
            (output_dir / "validation_manifest.json").write_text(json.dumps(panel, indent=2))
    else:
        validation_sampler = DistributedValidationSampler(
            validation_pools(validation_dataset, sources),
            num_samples=OmegaConf.select(args, "data.validation_samples", default=3000),
            negative_ratio=float(cfg_get(args, "data.validation_negative_ratio", 1.0)),
            num_replicas=DIST_WRAPPER.world_size, rank=DIST_WRAPPER.rank, seed=int(args.seed)+100000)
        validation_info = validation_sampler
    validation_loader = get_dataloader(validation_dataset, collate_fn=collate_fn,
        batch_size=int(args.data.batch_size), shuffle=False, sampler=validation_sampler,
        num_workers=int(args.data.num_workers), pin_memory=bool(args.data.pin_memory))
    if resume_state.get("rng_states"):
        restore_rng(resume_state["rng_states"][DIST_WRAPPER.rank])
    log_info(f"optimizer updates={scheduler.last_epoch} horizon="
             f"{int(args.epochs) * math.ceil(len(train_loader) / max(1, int(args.optimizer.grad_accum_steps)))} "
             f"warmup={args.optimizer.warmup_steps} lr={optimizer.param_groups[0]['lr']}")

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

    # A job may stop after saving an epoch but before its full inference ends.
    # Replay that event against the original output directory; completed events
    # are skipped by checkpoint/config identity before any model evaluation.
    if resume_state and inference_due(args, start_epoch - 1):
        source_checkpoint = Path(args.ckpt_dir).resolve()
        source_run = source_checkpoint.parent.parent if source_checkpoint.parent.name == "checkpoints" else output_dir
        run_epoch_inference(args, start_epoch - 1, source_checkpoint, source_run, live_model, ema, device)

    best_auprc = float(resume_state.get("best_auprc", float("-inf")))
    global_step = int(resume_state.get("global_step", 0))
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
        if not fixed:
            validation_sampler.set_epoch(epoch)
        log_info(f"validation epoch={epoch} samples={validation_info.total_size} "
                 f"positive={validation_info.positive_samples} negative={validation_info.negative_samples}")
        with preserve_rng(), (ema.average_parameters() if ema is not None else nullcontext()):
            validation_metrics = evaluate_epoch(live_model, validation_loader, loss_module, device,
                precision=precision, positive_weight=float(cfg_get(args, "metrics.positive_weight", 1.0)))
        improved = validation_metrics["pair_auprc"] > best_auprc
        if improved:
            best_auprc = validation_metrics["pair_auprc"]
        should_save = epoch % int(args.checkpoint_interval) == 0 or epoch == int(args.epochs)
        checkpoint_path = output_dir / "checkpoints" / f"epoch_{epoch}.pth"
        if should_save or improved:
            rng_states = DIST_WRAPPER.all_gather_object(capture_rng())
            training_state = {"best_auprc": best_auprc, "global_step": global_step,
                "optimizer_updates": int(scheduler.last_epoch), "world_size": DIST_WRAPPER.world_size,
                "rng_states": rng_states,
                "validation_panel": validation_dataset.manifest if fixed else None}
            if DIST_WRAPPER.rank == 0:
                options = dict(epoch=epoch, optimizer=optimizer, scheduler=scheduler, ema=ema,
                               training_state=training_state, config=OmegaConf.to_container(args, resolve=True))
                if should_save:
                    save_checkpoint(checkpoint_path, live_model, **options)
                if improved:
                    save_checkpoint(output_dir / "checkpoints" / "best_pair_auprc.pth", live_model, **options)
        if dist.is_initialized():
            dist.barrier()
        if inference_due(args, epoch):
            inference_metrics = run_epoch_inference(args, epoch, checkpoint_path, output_dir, live_model, ema, device)
            if wandb_run is not None:
                wandb.log({f"inference/{key}": value for key, value in inference_metrics.items()}, step=max(global_step, 1))

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


    if wandb_run is not None:
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
