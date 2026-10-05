import torch
from torch.utils.data import DataLoader


def _pad_value(tensor: torch.Tensor):
    if tensor.dtype == torch.bool:
        return False
    if torch.is_floating_point(tensor):
        return 0.0
    return 0


def pad_first_dim(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_n = max(int(tensor.shape[0]) for tensor in tensors)
    out = tensors[0].new_full(
        (len(tensors), max_n, *tensors[0].shape[1:]),
        fill_value=_pad_value(tensors[0]),
    )
    for index, tensor in enumerate(tensors):
        out[index, : tensor.shape[0]] = tensor
    return out


def concat_pair_residue_features(p1: dict, p2: dict) -> dict:
    concatenated = {}
    for key, p1_value in p1.items():
        p2_value = p2[key]
        if not isinstance(p1_value, torch.Tensor):
            concatenated[key] = p1_value
            continue
        if key == "chain_index":
            p1_chain = p1_value.long()
            p2_chain = p2_value.long() + p1_chain.amax() + 1
            concatenated[key] = torch.cat([p1_chain, p2_chain], dim=0)
        else:
            concatenated[key] = torch.cat([p1_value, p2_value], dim=0)
    return concatenated


def pad_residue_dicts(samples: list[dict]) -> dict:
    feature_batch = {}
    for key in samples[0].keys():
        values = [sample[key] for sample in samples]
        if not isinstance(values[0], torch.Tensor):
            feature_batch[key] = values
            continue
        feature_batch[key] = pad_first_dim(values)
    return feature_batch


def pad_pair_maps(tensors: list[torch.Tensor], max_total: int) -> torch.Tensor:
    out = tensors[0].new_full(
        (len(tensors), max_total, max_total, *tensors[0].shape[2:]),
        fill_value=_pad_value(tensors[0]),
    )
    for index, tensor in enumerate(tensors):
        length = int(tensor.shape[0])
        out[index, :length, :length] = tensor
    return out


def collate_concatenated_pairs(proteins: list[dict], peptides: list[dict]) -> dict:
    p1_lengths = [int(sample["mask"].shape[0]) for sample in proteins]
    p2_lengths = [int(sample["mask"].shape[0]) for sample in peptides]
    concatenated = [
        concat_pair_residue_features(protein, peptide)
        for protein, peptide in zip(proteins, peptides)
    ]
    residue_batch = pad_residue_dicts(concatenated)
    residue_batch["p1_length"] = torch.tensor(p1_lengths, dtype=torch.long)
    residue_batch["p2_length"] = torch.tensor(p2_lengths, dtype=torch.long)
    return residue_batch


def collate_fn(batch):
    proteins, peptides, labels_2d_bins, labels_2d_mask, is_positive = zip(*batch)
    residue_batch = collate_concatenated_pairs(list(proteins), list(peptides))
    max_total = int(residue_batch["mask"].shape[1])
    label_batch = {
        "label_2d_bins": pad_pair_maps(
            [tensor.to(dtype=torch.long) for tensor in labels_2d_bins],
            max_total,
        ),
        "label_2d_mask": pad_pair_maps(
            [tensor.to(dtype=torch.bool) for tensor in labels_2d_mask],
            max_total,
        ),
        "is_positive": torch.tensor(list(is_positive), dtype=torch.bool),
    }
    return residue_batch, label_batch


def get_dataloader(
    dataset,
    collate_fn,
    batch_size,
    shuffle=True,
    num_workers=4,
    sampler=None,
    pin_memory=True,
    drop_last=False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
