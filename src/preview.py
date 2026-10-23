import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

from src.data.balancing import build_multi_source_split
from src.data.collate import collate_fn, get_dataloader
from src.model.surpass import ResOnly
from src.model.loss import max_contact_probability, pair_contact_probability
from src.runtime.checkpoint import load_checkpoint
from src.runtime.config import build_model_kwargs, cfg_get
from src.runtime.device import autocast_context, resolve_cuda_device, to_device
from src.utils.ddp_utils import seed_everything

PREVIEW_SAMPLE_COUNT = 5


def resolve_device() -> torch.device:
    if torch.cuda.device_count() > 0:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        device = resolve_cuda_device(0, torch.cuda.device_count())
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def build_preview_loader(args: DictConfig):
    sources = cfg_get(args, "data.datasets", default=None)
    if sources is None:
        raise ValueError("data.datasets must list at least one cluster source.")
    sources = OmegaConf.to_container(sources, resolve=True)
    train_dataset, _, _, _ = build_multi_source_split(
        sources,
        root_dir=args.data.root_dir,
        dataset_kwargs={
            "distance_bin_start": args.data.distance_bin_start,
            "distance_bin_end": args.data.distance_bin_end,
            "distance_bin_count": args.data.distance_bin_count,
            "crop_size": args.data.crop_size,
            "contact_threshold": args.data.contact_threshold,
        },
        split_dir=cfg_get(args, "data.split_dir"),
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
    model = ResOnly(**build_model_kwargs(args)).to(device)
    load_checkpoint(args.ckpt_dir, model, map_location=device)
    model.eval()
    return model


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(args: DictConfig):
    seed_everything(seed=args.seed, deterministic=args.deterministic)
    device = resolve_device()
    model = build_model(args, device)
    loader, sample_count, train_size = build_preview_loader(args)

    print(
        f"Previewing {sample_count} of {train_size} train samples "
        f"(training crops, batch_size=1) from ckpt={args.ckpt_dir} on {device}"
    )

    with torch.inference_mode():
        for index, step_batch in enumerate(loader):
            residue_batch, labels = step_batch
            residue_batch = to_device(residue_batch, device)
            labels = to_device(labels, device)
            with autocast_context(
                device,
                str(cfg_get(args, "runtime.precision", "bfloat16")),
            ):
                output = model(residue_batch)
            probability = float(max_contact_probability(
                pair_contact_probability(output.pair_logits, model.contact_bins),
                output.pair_mask,
            )[0].item())
            label = bool(labels["crop_is_positive"].reshape(-1)[0].item())
            print(
                f"[sample {index:02d}] p_pair={probability:.4f} label={label}"
            )


if __name__ == "__main__":
    main()
