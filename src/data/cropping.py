import random

import torch

from src.data.labels import has_inter_chain_contact

DEFAULT_CROP_SIZE = 256
DEFAULT_CONTACT_THRESHOLD_A = 8.0


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
    contact_threshold: float,
) -> tuple[dict, dict, torch.Tensor, torch.Tensor, bool]:
    """Contiguous crop of both chains. A window with no contact is negative."""
    p1_length = int(p1_features["mask"].shape[0])
    if crop_size is not None:
        p1_idx = select_contiguous_crop_indices(p1_length, crop_size)
        p2_idx = select_contiguous_crop_indices(int(p2_features["mask"].shape[0]), crop_size)
        p1_features, p2_features, pairwise_dist, pairwise_mask = _slice_pair_maps(
            p1_features,
            p2_features,
            pairwise_dist,
            pairwise_mask,
            p1_idx,
            p2_idx,
            p1_length,
        )
        p1_length = int(p1_features["mask"].shape[0])
    is_positive = has_inter_chain_contact(
        pairwise_dist,
        pairwise_mask,
        p1_length,
        contact_threshold,
    )
    return p1_features, p2_features, pairwise_dist, pairwise_mask, is_positive


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
