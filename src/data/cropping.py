import random

import torch

from src.data.labels import has_inter_chain_contact

DEFAULT_CROP_SIZE = 256
DEFAULT_CROP_NEIGHBORHOOD_SIZE = 10
DEFAULT_CONTACT_THRESHOLD_A = 8.0
DEFAULT_CROP_CONTIGUOUS_PROBABILITY = 0.3


def select_contiguous_crop_indices(
    length: int,
    crop_size: int,
    start: int | None = None,
) -> torch.Tensor:
    if length <= crop_size:
        return torch.arange(length, dtype=torch.long)

    max_start = length - crop_size
    if start is None:
        start = random.randint(0, max_start)
    start = max(0, min(int(start), max_start))
    return torch.arange(start, start + crop_size, dtype=torch.long)


def select_centered_contiguous_crop_indices(
    length: int,
    crop_size: int,
    center: int,
) -> torch.Tensor:
    if length <= crop_size:
        return torch.arange(length, dtype=torch.long)
    start = int(center) - int(crop_size) // 2
    return select_contiguous_crop_indices(length, crop_size, start=start)


def _fill_crop_indices(selected: set[int], length: int, crop_size: int) -> set[int]:
    filled = set(selected)
    left = min(filled)
    right = max(filled)
    while len(filled) < crop_size and (left > 0 or right + 1 < length):
        if left > 0:
            left -= 1
            filled.add(left)
            if len(filled) >= crop_size:
                break
        if right + 1 < length and len(filled) < crop_size:
            right += 1
            filled.add(right)
    if len(filled) < crop_size:
        for idx in range(length):
            if idx not in filled:
                filled.add(idx)
                if len(filled) >= crop_size:
                    break
    return filled


def _gather_neighbor_indices(
    seeds: list[int],
    length: int,
    crop_size: int,
    neighborhood_size: int,
) -> torch.Tensor:
    if length <= crop_size:
        return torch.arange(length, dtype=torch.long)

    selected: set[int] = set()
    window_size = max(1, int(neighborhood_size))
    for seed in seeds:
        seed = int(seed)
        if seed not in selected:
            if len(selected) >= crop_size:
                break
            selected.add(seed)
        start = max(0, min(seed - window_size // 2, length - window_size))
        end = min(length, start + window_size)
        window = list(range(start, end))
        new_indices = [idx for idx in window if idx not in selected]
        if len(selected) + len(new_indices) > crop_size:
            break
        selected.update(new_indices)
        if len(selected) >= crop_size:
            break

    if not selected:
        return select_contiguous_crop_indices(length, crop_size)

    target = min(int(length), int(crop_size))
    if len(selected) < target:
        selected = _fill_crop_indices(selected, length, target)
    return torch.tensor(sorted(selected), dtype=torch.long)


def select_interface_crop_indices(
    pairwise_dist: torch.Tensor,
    pairwise_mask: torch.Tensor,
    crop_size: int,
    contact_threshold: float = DEFAULT_CONTACT_THRESHOLD_A,
    neighborhood_size: int = DEFAULT_CROP_NEIGHBORHOOD_SIZE,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    n1, n2 = int(pairwise_dist.shape[0]), int(pairwise_dist.shape[1])
    contact_mask = (pairwise_dist < float(contact_threshold)) & pairwise_mask.to(dtype=torch.bool)
    if not bool(contact_mask.any().item()):
        return None
    if n1 <= crop_size and n2 <= crop_size:
        return torch.arange(n1, dtype=torch.long), torch.arange(n2, dtype=torch.long)

    contacts = torch.nonzero(contact_mask, as_tuple=False)
    order = list(range(int(contacts.shape[0])))
    random.shuffle(order)

    p1_selected = _gather_neighbor_indices(
        [int(contacts[i, 0]) for i in order],
        length=n1,
        crop_size=crop_size,
        neighborhood_size=neighborhood_size,
    )
    p2_selected = _gather_neighbor_indices(
        [int(contacts[i, 1]) for i in order],
        length=n2,
        crop_size=crop_size,
        neighborhood_size=neighborhood_size,
    )
    return p1_selected, p2_selected


def select_contact_centered_crop_indices(
    pairwise_dist: torch.Tensor,
    pairwise_mask: torch.Tensor,
    crop_size: int,
    contact_threshold: float = DEFAULT_CONTACT_THRESHOLD_A,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    contact_mask = (pairwise_dist < float(contact_threshold)) & pairwise_mask.to(dtype=torch.bool)
    if not bool(contact_mask.any().item()):
        return None
    contacts = torch.nonzero(contact_mask, as_tuple=False)
    choice = contacts[random.randrange(int(contacts.shape[0]))]
    n1, n2 = int(pairwise_dist.shape[0]), int(pairwise_dist.shape[1])
    return (
        select_centered_contiguous_crop_indices(n1, crop_size, int(choice[0])),
        select_centered_contiguous_crop_indices(n2, crop_size, int(choice[1])),
    )


def slice_complex_features(features: dict, indices: torch.Tensor) -> dict:
    residue_count = int(features["mask"].shape[0])
    return {
        key: value.index_select(0, indices.to(device=value.device))
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == residue_count
        else value
        for key, value in features.items()
    }


def _slice_pair_maps(
    p1_features: dict,
    p2_features: dict,
    pairwise_dist: torch.Tensor,
    pairwise_mask: torch.Tensor,
    p1_idx: torch.Tensor,
    p2_idx: torch.Tensor,
    p1_length: int,
) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    full_idx = torch.cat([p1_idx, p2_idx + p1_length], dim=0)
    return (
        slice_complex_features(p1_features, p1_idx),
        slice_complex_features(p2_features, p2_idx),
        pairwise_dist.index_select(0, full_idx).index_select(1, full_idx),
        pairwise_mask.index_select(0, full_idx).index_select(1, full_idx),
    )


def crop_positive_pair(
    p1_features: dict,
    p2_features: dict,
    pairwise_dist: torch.Tensor,
    pairwise_mask: torch.Tensor,
    *,
    crop_size: int | None,
    crop_contiguous_probability: float,
    crop_neighborhood_size: int,
    contact_threshold: float,
) -> tuple[dict, dict, torch.Tensor, torch.Tensor] | None:
    p1_length = int(p1_features["mask"].shape[0])
    if not has_inter_chain_contact(pairwise_dist, pairwise_mask, p1_length, contact_threshold):
        return None
    if crop_size is None:
        return p1_features, p2_features, pairwise_dist, pairwise_mask

    cross_dist = pairwise_dist[:p1_length, p1_length:]
    cross_mask = pairwise_mask[:p1_length, p1_length:]
    probability = float(crop_contiguous_probability)
    use_contiguous = probability >= 1.0 or (probability > 0.0 and random.random() < probability)
    if use_contiguous:
        selected = select_contact_centered_crop_indices(
            cross_dist,
            cross_mask,
            crop_size=crop_size,
            contact_threshold=contact_threshold,
        )
    else:
        selected = select_interface_crop_indices(
            cross_dist,
            cross_mask,
            crop_size=crop_size,
            contact_threshold=contact_threshold,
            neighborhood_size=crop_neighborhood_size,
        )
    if selected is None:
        return None
    p1_idx, p2_idx = selected
    cropped = _slice_pair_maps(
        p1_features,
        p2_features,
        pairwise_dist,
        pairwise_mask,
        p1_idx,
        p2_idx,
        p1_length,
    )
    cropped_p1_length = int(cropped[0]["mask"].shape[0])
    if not has_inter_chain_contact(cropped[2], cropped[3], cropped_p1_length, contact_threshold):
        return None
    return cropped


def crop_negative_pair(
    p1_features: dict,
    p2_features: dict,
    *,
    crop_size: int | None,
) -> tuple[dict, dict]:
    if crop_size is None:
        return p1_features, p2_features
    p1_idx = select_contiguous_crop_indices(int(p1_features["mask"].shape[0]), crop_size)
    p2_idx = select_contiguous_crop_indices(int(p2_features["mask"].shape[0]), crop_size)
    return (
        slice_complex_features(p1_features, p1_idx),
        slice_complex_features(p2_features, p2_idx),
    )
