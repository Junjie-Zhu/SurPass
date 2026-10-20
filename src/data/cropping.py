"""Contiguous chain-pair cropping under one residue budget."""

from __future__ import annotations

import random
from typing import Any

import torch

from src.data.labels import has_inter_chain_contact, inter_chain_contact_mask

DEFAULT_CROP_SIZE = 256
DEFAULT_CONTACT_THRESHOLD_A = 8.0
DEFAULT_INTERFACE_CROP_PROBABILITY = 0.8

CropMetadata = dict[str, Any]


def split_total_crop_budget(
    p1_length: int,
    p2_length: int,
    total_budget: int | None,
) -> tuple[int, int]:
    """Split one budget between chains, reallocating unused capacity.

    Both chains retain at least one residue. When both are long, the second
    chain receives the odd residue; unused budget from a short chain is given
    to the other chain.
    """
    p1_length = int(p1_length)
    p2_length = int(p2_length)
    if p1_length <= 0 or p2_length <= 0:
        raise ValueError("Both chains must contain at least one residue.")
    if total_budget is None or p1_length + p2_length <= int(total_budget):
        return p1_length, p2_length

    total_budget = int(total_budget)
    if total_budget < 2:
        raise ValueError("total_budget must be at least 2 for a chain pair.")

    p1_budget = min(p1_length, total_budget // 2)
    p2_budget = min(p2_length, total_budget - p1_budget)
    remaining = total_budget - p1_budget - p2_budget

    add_to_p1 = min(remaining, p1_length - p1_budget)
    p1_budget += add_to_p1
    remaining -= add_to_p1
    p2_budget += min(remaining, p2_length - p2_budget)
    return p1_budget, p2_budget


def select_contiguous_crop_indices(
    length: int,
    crop_size: int,
    start: int | None = None,
    *,
    rng: random.Random | Any = random,
) -> torch.Tensor:
    """Return a contiguous crop, using ``rng`` only when no start is supplied."""
    length = int(length)
    crop_size = int(crop_size)
    if length < 0:
        raise ValueError("length must be non-negative.")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive.")
    if length <= crop_size:
        return torch.arange(length, dtype=torch.long)

    max_start = length - crop_size
    if start is None:
        start = rng.randint(0, max_start)
    start = max(0, min(int(start), max_start))
    return torch.arange(start, start + crop_size, dtype=torch.long)


def select_centered_crop_indices(
    length: int,
    crop_size: int,
    center: int,
) -> torch.Tensor:
    """Return a contiguous crop centered as closely as possible on a residue."""
    start = int(center) - int(crop_size) // 2
    return select_contiguous_crop_indices(length, crop_size, start=start)


def slice_complex_features(features: dict, indices: torch.Tensor) -> dict:
    residue_count = int(features["mask"].shape[0])
    return {
        key: value.index_select(0, indices.to(device=value.device))
        if isinstance(value, torch.Tensor)
        and value.ndim > 0
        and int(value.shape[0]) == residue_count
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
    distance_idx = full_idx.to(device=pairwise_dist.device)
    mask_idx = full_idx.to(device=pairwise_mask.device)
    return (
        slice_complex_features(p1_features, p1_idx),
        slice_complex_features(p2_features, p2_idx),
        pairwise_dist.index_select(0, distance_idx).index_select(1, distance_idx),
        pairwise_mask.index_select(0, mask_idx).index_select(1, mask_idx),
    )


def _crop_span(indices: torch.Tensor) -> tuple[int, int]:
    if indices.numel() == 0:
        return 0, 0
    start = int(indices[0].item())
    return start, start + int(indices.numel())


def _crop_metadata(
    *,
    mode: str,
    total_budget: int | None,
    p1_idx: torch.Tensor,
    p2_idx: torch.Tensor,
    selected_contact: tuple[int, int] | None = None,
) -> CropMetadata:
    p1_start, p1_end = _crop_span(p1_idx)
    p2_start, p2_end = _crop_span(p2_idx)
    return {
        "mode": mode,
        "total_budget": None if total_budget is None else int(total_budget),
        "p1_start": p1_start,
        "p1_end": p1_end,
        "p2_start": p2_start,
        "p2_end": p2_end,
        "p1_length": p1_end - p1_start,
        "p2_length": p2_end - p2_start,
        "selected_contact": selected_contact,
    }


def crop_positive_pair(
    p1_features: dict,
    p2_features: dict,
    pairwise_dist: torch.Tensor,
    pairwise_mask: torch.Tensor,
    *,
    crop_size: int | None = DEFAULT_CROP_SIZE,
    contact_threshold: float = DEFAULT_CONTACT_THRESHOLD_A,
    interface_crop_probability: float = DEFAULT_INTERFACE_CROP_PROBABILITY,
    rng: random.Random | Any = random,
) -> tuple[
    dict,
    dict,
    torch.Tensor,
    torch.Tensor,
    bool,
    CropMetadata,
]:
    """Crop a known-positive source and report the observed crop label.

    ``crop_size`` is a *total* residue budget. Interface mode centers both
    contiguous chain windows on one observed contact. Random mode samples the
    two windows independently; it may legitimately crop every contact away.
    """
    if not 0.0 <= float(interface_crop_probability) <= 1.0:
        raise ValueError("interface_crop_probability must be between 0 and 1.")

    p1_length = int(p1_features["mask"].shape[0])
    p2_length = int(p2_features["mask"].shape[0])
    p1_budget, p2_budget = split_total_crop_budget(
        p1_length, p2_length, crop_size
    )
    no_crop = p1_budget == p1_length and p2_budget == p2_length

    contacts = inter_chain_contact_mask(
        pairwise_dist,
        pairwise_mask,
        p1_length,
        contact_threshold,
    ).nonzero(as_tuple=False)
    selected_contact: tuple[int, int] | None = None

    if no_crop:
        mode = "full"
        p1_idx = torch.arange(p1_length, dtype=torch.long)
        p2_idx = torch.arange(p2_length, dtype=torch.long)
    elif contacts.numel() > 0 and rng.random() < float(interface_crop_probability):
        mode = "interface"
        contact = contacts[rng.randrange(int(contacts.shape[0]))]
        selected_contact = (int(contact[0].item()), int(contact[1].item()))
        p1_idx = select_centered_crop_indices(
            p1_length, p1_budget, selected_contact[0]
        )
        p2_idx = select_centered_crop_indices(
            p2_length, p2_budget, selected_contact[1]
        )
    else:
        mode = "random"
        p1_idx = select_contiguous_crop_indices(
            p1_length, p1_budget, rng=rng
        )
        p2_idx = select_contiguous_crop_indices(
            p2_length, p2_budget, rng=rng
        )

    p1_features, p2_features, pairwise_dist, pairwise_mask = _slice_pair_maps(
        p1_features,
        p2_features,
        pairwise_dist,
        pairwise_mask,
        p1_idx,
        p2_idx,
        p1_length,
    )
    cropped_p1_length = int(p1_features["mask"].shape[0])
    contains_contact = has_inter_chain_contact(
        pairwise_dist,
        pairwise_mask,
        cropped_p1_length,
        contact_threshold,
    )
    metadata = _crop_metadata(
        mode=mode,
        total_budget=crop_size,
        p1_idx=p1_idx,
        p2_idx=p2_idx,
        selected_contact=selected_contact,
    )
    metadata["contains_contact"] = contains_contact
    return (
        p1_features,
        p2_features,
        pairwise_dist,
        pairwise_mask,
        contains_contact,
        metadata,
    )


def crop_negative_pair(
    p1_features: dict,
    p2_features: dict,
    *,
    crop_size: int | None = DEFAULT_CROP_SIZE,
    rng: random.Random | Any = random,
) -> tuple[dict, dict, CropMetadata]:
    """Randomly crop a negative pair under the same total residue budget."""
    p1_length = int(p1_features["mask"].shape[0])
    p2_length = int(p2_features["mask"].shape[0])
    p1_budget, p2_budget = split_total_crop_budget(
        p1_length, p2_length, crop_size
    )
    no_crop = p1_budget == p1_length and p2_budget == p2_length
    p1_idx = select_contiguous_crop_indices(p1_length, p1_budget, rng=rng)
    p2_idx = select_contiguous_crop_indices(p2_length, p2_budget, rng=rng)
    metadata = _crop_metadata(
        mode="full" if no_crop else "random",
        total_budget=crop_size,
        p1_idx=p1_idx,
        p2_idx=p2_idx,
    )
    metadata["contains_contact"] = False
    return (
        slice_complex_features(p1_features, p1_idx),
        slice_complex_features(p2_features, p2_idx),
        metadata,
    )
