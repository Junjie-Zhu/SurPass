import os
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

from src.data.dataset import PepoTrainDataset, collate_fn, get_dataloader
from src.model.surpass import ResOnly
from src.train import (
    _cfg_get,
    _unpack_batch,
    _validate_model_bin_counts,
    create_balanced_split_datasets,
    resolve_cuda_device,
)
from src.utils.ddp_utils import seed_everything

PREVIEW_SAMPLE_COUNT = 5
PREVIEW_OUTPUT_DIR = Path("preview")


def distance_bin_centers(
    bin_start: float,
    bin_end: float,
    bin_count: int,
) -> torch.Tensor:
    bin_count = int(bin_count)
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2.")
    limits = torch.linspace(float(bin_start), float(bin_end), bin_count - 1)
    centers = torch.empty(bin_count, dtype=torch.float32)
    if bin_count == 2:
        width = float(bin_end) - float(bin_start)
        centers[0] = limits[0] - 0.5 * width
        centers[1] = limits[-1] + 0.5 * width
        return centers
    first_width = limits[1] - limits[0]
    last_width = limits[-1] - limits[-2]
    centers[0] = limits[0] - 0.5 * first_width
    centers[1:-1] = 0.5 * (limits[:-1] + limits[1:])
    centers[-1] = limits[-1] + 0.5 * last_width
    return centers


def bins_to_distance(bins: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return centers.to(device=bins.device)[bins.long()]


def masked_distance_map(
    bins: torch.Tensor,
    centers: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    distances = bins_to_distance(bins, centers).to(dtype=torch.float32)
    return distances.masked_fill(~valid_mask.to(dtype=torch.bool), float("nan"))


def save_distance_pair_figure(
    label_map: torch.Tensor,
    pred_map: torch.Tensor,
    output_path: Path,
    p1_length: int | None = None,
) -> None:
    label = label_map.detach().cpu().numpy()
    pred = pred_map.detach().cpu().numpy()
    finite = torch.cat([label_map.reshape(-1), pred_map.reshape(-1)])
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(finite.min().item())
        vmax = float(finite.max().item())
        if vmin == vmax:
            vmax = vmin + 1.0

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    image = None
    for axis, array, title in (
        (axes[0], label, "label"),
        (axes[1], pred, "prediction"),
    ):
        image = axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        axis.set_title(title)
        axis.set_xlabel("residue j")
        axis.set_ylabel("residue i")
        _draw_chain_margin(axis, p1_length, map_size=array.shape[0])
    fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _draw_chain_margin(axis, p1_length: int | None, map_size: int) -> None:
    if p1_length is None:
        return
    split = float(p1_length) - 0.5
    if split <= -0.5 or split >= map_size - 0.5:
        return
    for draw in (axis.axhline, axis.axvline):
        draw(split, color="white", linewidth=1.4, linestyle="-", zorder=3)


def summarize_prediction(
    pred_bins: torch.Tensor,
    valid_mask: torch.Tensor,
    last_bin: int,
) -> str:
    valid_bins = pred_bins[valid_mask.to(dtype=torch.bool)]
    if valid_bins.numel() == 0:
        return "valid_pairs=0"
    unique_count = int(torch.unique(valid_bins).numel())
    far_fraction = float((valid_bins == last_bin).to(dtype=torch.float32).mean().item())
    return (
        f"valid_pairs={int(valid_bins.numel())} "
        f"unique_pred_bins={unique_count} "
        f"far_bin_frac={far_fraction:.3f}"
    )


def resolve_device() -> torch.device:
    if torch.cuda.device_count() > 0:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        device = resolve_cuda_device(0, torch.cuda.device_count())
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def build_preview_loader(args: DictConfig):
    full_dataset = PepoTrainDataset(
        root_dir=args.data.root_dir,
        cluster_tsv_path=args.data.cluster_tsv_path,
        center_coordinates=args.data.center_coordinates,
        random_rotation=args.data.random_rotation,
        distance_bin_start=args.data.distance_bin_start,
        distance_bin_end=args.data.distance_bin_end,
        distance_bin_count=args.data.distance_bin_count,
        crop_size=None,
        contact_threshold=args.data.contact_threshold,
    )
    train_dataset, _ = create_balanced_split_datasets(
        full_dataset,
        test_fraction=args.data.test_fraction,
        seed=args.seed,
        negative_ratio=int(_cfg_get(args, "data.negative_ratio", default=1)),
        distance_bin_count=args.data.distance_bin_count,
    )
    sample_count = min(PREVIEW_SAMPLE_COUNT, len(train_dataset))
    if sample_count == 0:
        raise ValueError("Training set is empty; cannot preview samples.")
    preview_dataset = Subset(train_dataset, list(range(sample_count)))
    loader = get_dataloader(
        preview_dataset,
        collate_fn=collate_fn,
        batch_size=1,
        shuffle=False,
        num_workers=args.data.num_workers,
        pin_memory=args.data.pin_memory,
    )
    return loader, sample_count, len(train_dataset)


def build_model(args: DictConfig, device: torch.device) -> torch.nn.Module:
    if args.ckpt_dir is None:
        raise ValueError(
            "ckpt_dir is required. Example: "
            "python -m src.preview ckpt_dir=logs/<run>/checkpoints/epoch_N.pth"
        )
    model_kwargs = _cfg_get(args, "model", default={})
    if isinstance(model_kwargs, DictConfig):
        model_kwargs = OmegaConf.to_container(model_kwargs, resolve=True)
    model_kwargs = dict(model_kwargs or {})
    model_kwargs = _validate_model_bin_counts(model_kwargs, args.data.distance_bin_count)
    model = ResOnly(**model_kwargs).to(device)
    checkpoint = torch.load(args.ckpt_dir, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(args: DictConfig):
    seed_everything(seed=args.seed, deterministic=args.deterministic)
    device = resolve_device()
    model = build_model(args, device)
    loader, sample_count, train_size = build_preview_loader(args)
    centers = distance_bin_centers(
        args.data.distance_bin_start,
        args.data.distance_bin_end,
        args.data.distance_bin_count,
    )
    last_bin = int(args.data.distance_bin_count) - 1
    recycle_rounds = max(1, int(args.recycle_rounds))
    PREVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Previewing {sample_count} of {train_size} train samples "
        f"(full-length, batch_size=1) from ckpt={args.ckpt_dir} on {device}"
    )

    with torch.no_grad():
        for index, step_batch in enumerate(loader):
            residue_batch, labels = _unpack_batch(step_batch, device)
            logits, pair_mask = model(
                residue_batch,
                recycle_rounds=recycle_rounds,
            )
            pred_bins = logits.argmax(dim=-1)
            valid_mask = pair_mask.to(dtype=torch.bool) & labels["label_2d_mask"].to(
                dtype=torch.bool
            )
            label_map = masked_distance_map(labels["label_2d_bins"], centers, valid_mask)
            pred_map = masked_distance_map(pred_bins, centers, valid_mask)

            # batch_size is 1
            output_path = PREVIEW_OUTPUT_DIR / f"sample_{index:02d}.png"
            save_distance_pair_figure(
                label_map[0],
                pred_map[0],
                output_path,
                p1_length=int(residue_batch["p1_length"][0].item()),
            )
            print(
                f"[sample {index:02d}] {summarize_prediction(pred_bins[0], valid_mask[0], last_bin)} "
                f"-> {output_path}"
            )


if __name__ == "__main__":
    main()
