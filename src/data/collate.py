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


def pad_rectangular_pair_maps(
    tensors: list[torch.Tensor],
    max_p1_length: int,
    max_p2_length: int,
) -> torch.Tensor:
    out = tensors[0].new_full(
        (
            len(tensors),
            max_p1_length,
            max_p2_length,
            *tensors[0].shape[2:],
        ),
        fill_value=_pad_value(tensors[0]),
    )
    for index, tensor in enumerate(tensors):
        p1_length, p2_length = int(tensor.shape[0]), int(tensor.shape[1])
        out[index, :p1_length, :p2_length] = tensor
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
    index = torch.arange(residue_batch["plm_emb"].shape[1]).view(1, -1)
    total = (
        residue_batch["p1_length"] + residue_batch["p2_length"]
    ).view(-1, 1)
    residue_batch["padding_mask"] = index < total
    return residue_batch


def _unpack_sample(sample):
    if len(sample) == 7:
        return tuple(sample)
    raise ValueError(f"Expected a seven-field PairSample, got {len(sample)}.")


def _as_inter_chain_map(
    tensor: torch.Tensor,
    p1_length: int,
    p2_length: int,
) -> torch.Tensor:
    """Accept a rectangular map, or extract the forward block from a legacy square map."""
    expected_rectangular = (int(p1_length), int(p2_length))
    if tuple(tensor.shape[:2]) == expected_rectangular:
        return tensor

    total = int(p1_length) + int(p2_length)
    if tuple(tensor.shape[:2]) == (total, total):
        return tensor[:p1_length, p1_length:total]
    raise ValueError(
        f"Pair map shape {tuple(tensor.shape[:2])} is neither "
        f"{expected_rectangular} nor legacy square {(total, total)}."
    )


def collate_fn(batch):
    unpacked = [_unpack_sample(sample) for sample in batch]
    (
        proteins,
        peptides,
        label_inter_bins,
        label_inter_mask,
        crop_is_positive,
        source_is_positive,
        metadata,
    ) = zip(*unpacked)
    proteins = list(proteins)
    peptides = list(peptides)
    residue_batch = collate_concatenated_pairs(proteins, peptides)
    p1_lengths = [int(features["mask"].shape[0]) for features in proteins]
    p2_lengths = [int(features["mask"].shape[0]) for features in peptides]
    max_p1_length = max(p1_lengths)
    max_p2_length = max(p2_lengths)
    rectangular_bins = [
        _as_inter_chain_map(tensor, p1_length, p2_length).to(dtype=torch.long)
        for tensor, p1_length, p2_length in zip(
            label_inter_bins, p1_lengths, p2_lengths
        )
    ]
    rectangular_masks = [
        _as_inter_chain_map(tensor, p1_length, p2_length).to(dtype=torch.bool)
        for tensor, p1_length, p2_length in zip(
            label_inter_mask, p1_lengths, p2_lengths
        )
    ]
    label_batch = {
        "label_inter_bins": pad_rectangular_pair_maps(
            rectangular_bins,
            max_p1_length,
            max_p2_length,
        ),
        "label_inter_mask": pad_rectangular_pair_maps(
            rectangular_masks,
            max_p1_length,
            max_p2_length,
        ),
        "crop_is_positive": torch.tensor(
            list(crop_is_positive), dtype=torch.bool
        ),
        "source_is_positive": torch.tensor(
            list(source_is_positive), dtype=torch.bool
        ),
        "crop_metadata": list(metadata),
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
