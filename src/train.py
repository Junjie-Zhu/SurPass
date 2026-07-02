import os
import warnings
import datetime

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import PepoDataset, collate_fn, get_dataloader
from src.model.loss import MaskedBinnedBCELoss, CrossEntropyLoss
from src.model.surpass import ResOnly
from src.model.optimizer import get_optimizer, get_lr_scheduler
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
        cur = cfg
        found = True
        for key in path.split("."):
            if hasattr(cur, key):
                cur = getattr(cur, key)
            elif isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                found = False
                break
        if found:
            return cur
    return default


def _split_train_val_metadata(
    full_metadata: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(full_metadata) < 2:
        raise ValueError("Need at least 2 samples to create a train/validation split.")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"`val_ratio` must be in (0, 1), got {val_ratio}.")

    shuffled = full_metadata.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    val_metadata = shuffled.iloc[:n_val].reset_index(drop=True)
    train_metadata = shuffled.iloc[n_val:].reset_index(drop=True)

    if "total_length" in train_metadata.columns:
        train_metadata = train_metadata.sort_values(by="total_length", ascending=False)
        val_metadata = val_metadata.sort_values(by="total_length", ascending=False)
    return train_metadata, val_metadata


def _masked_bin_rows(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_bin = logits.detach().argmax(dim=-1).reshape(-1)
    true_bin = target.detach().argmax(dim=-1).reshape(-1)
    valid = mask.detach().reshape(-1).to(dtype=torch.bool, device=pred_bin.device)
    return pred_bin[valid].unsqueeze(-1), true_bin[valid].unsqueeze(-1)


def _gather_tensor_rows(tensor: torch.Tensor) -> torch.Tensor:
    if DIST_WRAPPER.world_size <= 1:
        return tensor

    local_rows = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    row_counts = [torch.zeros_like(local_rows) for _ in range(DIST_WRAPPER.world_size)]
    dist.all_gather(row_counts, local_rows)
    max_rows = int(max(x.item() for x in row_counts))

    if tensor.shape[0] < max_rows:
        pad_shape = (max_rows - tensor.shape[0], *tensor.shape[1:])
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        padded = torch.cat([tensor, pad], dim=0)
    else:
        padded = tensor

    gathered = [torch.zeros_like(padded) for _ in range(DIST_WRAPPER.world_size)]
    dist.all_gather(gathered, padded)
    trimmed = [chunk[: int(count.item())] for chunk, count in zip(gathered, row_counts)]
    return torch.cat(trimmed, dim=0)


def _binned_metrics(pred_bin: torch.Tensor, true_bin: torch.Tensor) -> dict[str, float]:
    pred = pred_bin.detach().flatten().to(dtype=torch.float32).cpu()
    true = true_bin.detach().flatten().to(dtype=torch.float32).cpu()
    if pred.numel() == 0:
        return {"acc": float("nan"), "mae": float("nan")}
    acc = (pred == true).to(dtype=torch.float32).mean().item()
    mae = torch.abs(pred - true).mean().item()
    return {"acc": float(acc), "mae": float(mae)}


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(args: DictConfig):
    logging_dir = os.path.join(
        args.logging_dir,
        f"{args.task_prefix}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    if DIST_WRAPPER.rank == 0:
        os.makedirs(args.logging_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(os.path.join(logging_dir, "checkpoints"), exist_ok=True)
        with open(f"{logging_dir}/config.yaml", "w") as f:
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
            backend="hccl", timeout=datetime.timedelta(seconds=timeout_seconds)
        )

    seed_everything(seed=args.seed, deterministic=args.deterministic)

    train_csv_path = _cfg_get(
        args, "data.train_csv_path", "data.train_csv", "train_csv_path", "train_csv"
    )
    if train_csv_path is None:
        raise ValueError("A training csv path must be provided in config.")

    batch_size = int(_cfg_get(args, "data.batch_size", "batch_size", default=1))
    num_workers = int(_cfg_get(args, "data.num_workers", "num_workers", default=4))
    val_ratio = float(_cfg_get(args, "data.val_ratio", "val_ratio", default=0.05))
    split_seed = int(_cfg_get(args, "data.split_seed", "split_seed", default=args.seed))
    center_coordinates = bool(
        _cfg_get(args, "data.center_coordinates", "center_coordinates", default=True)
    )
    random_rotation = bool(
        _cfg_get(args, "data.random_rotation", "random_rotation", default=True)
    )

    full_metadata = pd.read_csv(train_csv_path)
    train_metadata, val_metadata = _split_train_val_metadata(
        full_metadata=full_metadata,
        val_ratio=val_ratio,
        seed=split_seed,
    )
    log_info(
        f"Split dataset from {train_csv_path}: train={len(train_metadata)} "
        f"val={len(val_metadata)} ratio={val_ratio:.3f}"
    )

    train_dataset = PepoDataset(
        metadata=train_metadata,
        center_coordinates=center_coordinates,
        random_rotation=random_rotation,
    )
    val_dataset = PepoDataset(
        metadata=val_metadata,
        center_coordinates=center_coordinates,
        random_rotation=False,
    )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=DIST_WRAPPER.world_size,
            rank=DIST_WRAPPER.rank,
            shuffle=False,
        )
        if DIST_WRAPPER.world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_dataset,
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=train_sampler,
    )
    val_loader = get_dataloader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler,
    )

    model_kwargs = _cfg_get(args, "model", default={})
    if isinstance(model_kwargs, DictConfig):
        model_kwargs = OmegaConf.to_container(model_kwargs, resolve=True)
    model = ResOnly(**(model_kwargs or {})).to(device)
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

    pair_loss_fn = MaskedBinnedBCELoss().to(device)
    pair_loss_weight = float(_cfg_get(args, "pair_loss_weight", default=1.0))
    empty_cache_each_step = bool(
        _cfg_get(args, "performance.empty_cache_each_step", "empty_cache_each_step", default=False)
    )
    grad_accum_steps = max(
        1, int(_cfg_get(args, "optimizer.grad_accum_steps", "grad_accum_steps", default=1))
    )
    max_grad_norm = float(
        _cfg_get(args, "optimizer.max_grad_norm", "max_grad_norm", default=0.0)
    )
    if use_cuda and (not args.deterministic):
        torch.backends.cudnn.benchmark = True

    def run_forward(
        step_batch,
        loss_weight=1.0,
        collect_eval=False,
    ):
        protein_batch, peptide_batch, labels = step_batch
        protein_batch = to_device(protein_batch, device)
        peptide_batch = to_device(peptide_batch, device)
        labels = to_device(labels, device)

        pairwise_logits, model_pair_mask = model(
            p1_batch=protein_batch["residue_features"],
            p2_batch=peptide_batch["residue_features"],
        )
        pair_target = labels["label_2d_bins"].to(dtype=torch.float32)
        pair_label_mask = labels["label_2d_mask"].to(dtype=torch.bool)

        if pairwise_logits.shape != pair_target.shape:
            raise ValueError(
                f"Logit/target shape mismatch: logits={tuple(pairwise_logits.shape)} "
                f"target={tuple(pair_target.shape)}"
            )
        if tuple(pair_label_mask.shape) != tuple(pairwise_logits.shape[:3]):
            raise ValueError(
                f"Mask/logit shape mismatch: mask={tuple(pair_label_mask.shape)} "
                f"logits={tuple(pairwise_logits.shape)}"
            )

        pair_mask = pair_label_mask & model_pair_mask.to(dtype=torch.bool)
        pair_loss = pair_loss_fn(pairwise_logits, pair_target, pair_mask)
        loss = loss_weight * pair_loss
        loss_dict = {"pair_bin_loss": f"{loss.item():.3f}"}
        if not collect_eval:
            return loss, loss_dict, None

        pred_rows, true_rows = _masked_bin_rows(
            logits=pairwise_logits,
            target=pair_target,
            mask=pair_mask,
        )
        return loss, loss_dict, (pred_rows, true_rows)

    if empty_cache_each_step and use_cuda:
        torch.cuda.empty_cache()
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
        if DIST_WRAPPER.world_size > 1:
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        if not args.load_model_only:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
        del checkpoint

    if empty_cache_each_step and use_cuda:
        torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for check_iter, check_dict in enumerate(val_loader):
            if empty_cache_each_step and use_cuda:
                torch.cuda.empty_cache()
            _, _, _ = run_forward(
                check_dict,
                loss_weight=pair_loss_weight,
                collect_eval=False,
            )
            if check_iter >= 2:
                break
    log_info("Sanity check done")

    if DIST_WRAPPER.rank == 0:
        with open(f"{logging_dir}/loss.csv", "w") as f:
            f.write("Epoch,Loss,Val Loss,pair_bin_acc,pair_bin_mae\n")

    epoch_progress = (
        tqdm(total=args.epochs, leave=False, position=0)
        if DIST_WRAPPER.rank == 0
        else None
    )

    for crt_epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(crt_epoch)

        epoch_loss, epoch_val_loss = 0.0, 0.0
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_iter = enumerate(train_loader)
        if DIST_WRAPPER.rank == 0:
            train_iter = tqdm(
                train_iter,
                desc="Step",
                total=len(train_loader),
                leave=True,
                position=1,
            )

        crt_step, crt_val_step = 0, 0
        for crt_step, train_dict in train_iter:
            if empty_cache_each_step and use_cuda:
                torch.cuda.empty_cache()

            loss, loss_dict, _ = run_forward(
                train_dict,
                loss_weight=pair_loss_weight,
                collect_eval=False,
            )
            raw_loss = loss
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
            loss.backward()

            should_step = ((crt_step + 1) % grad_accum_steps == 0) or (
                (crt_step + 1) == len(train_loader)
            )
            if should_step:
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step_loss = raw_loss.item()
            epoch_loss += step_loss
            if DIST_WRAPPER.rank == 0:
                train_iter.set_postfix(step_loss=f"{step_loss:.3f}", **loss_dict)

        epoch_loss /= (crt_step + 1)

        model.eval()
        pred_rows_all = []
        true_rows_all = []
        with torch.no_grad():
            val_iter = enumerate(val_loader)
            if DIST_WRAPPER.rank == 0:
                val_iter = tqdm(
                    val_iter,
                    desc="Validation",
                    total=len(val_loader),
                    leave=True,
                    position=1,
                )

            for crt_val_step, val_dict in val_iter:
                if empty_cache_each_step and use_cuda:
                    torch.cuda.empty_cache()

                val_loss, val_loss_dict, val_rows = run_forward(
                    val_dict,
                    loss_weight=pair_loss_weight,
                    collect_eval=True,
                )
                pred_rows_all.append(val_rows[0])
                true_rows_all.append(val_rows[1])

                step_val_loss = val_loss.item()
                epoch_val_loss += step_val_loss
                if DIST_WRAPPER.rank == 0:
                    val_iter.set_postfix(val_loss=f"{step_val_loss:.3f}", **val_loss_dict)

        epoch_val_loss /= (crt_val_step + 1)

        pred_rows = torch.cat(pred_rows_all, dim=0)
        true_rows = torch.cat(true_rows_all, dim=0)
        pred_rows = _gather_tensor_rows(pred_rows)
        true_rows = _gather_tensor_rows(true_rows)
        metrics = _binned_metrics(pred_rows, true_rows)

        if DIST_WRAPPER.rank == 0 and epoch_progress is not None:
            epoch_progress.set_postfix(loss=f"{epoch_loss:.3f}", val_loss=f"{epoch_val_loss:.3f}")
            epoch_progress.update()

            with open(f"{logging_dir}/loss.csv", "a") as f:
                f.write(
                    f"{crt_epoch},{epoch_loss},{epoch_val_loss},"
                    f"{metrics['acc']},{metrics['mae']}\n"
                )

            log_info(
                f"[val-metrics][epoch={crt_epoch}] "
                f"[pair-bin] ACC={metrics['acc']:.4f} MAE={metrics['mae']:.4f}"
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

        if empty_cache_each_step and use_cuda:
            torch.cuda.empty_cache()

    if DIST_WRAPPER.world_size > 1:
        dist.destroy_process_group()


def to_device(obj, device):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                to_device(v, device)
            elif isinstance(v, torch.Tensor):
                obj[k] = obj[k].to(device)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    else:
        try:
            obj = obj.to(device)
        except Exception as exc:
            raise Exception(f"type {type(obj)} not supported") from exc
    return obj


if __name__ == "__main__":
    main()
