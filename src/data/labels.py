"""Distance/contact label helpers."""

from __future__ import annotations

import torch

DISTANCE_BIN_START_A = 2.0
DISTANCE_BIN_END_A = 22.0
DISTANCE_BIN_NUM = 16


def distance_bin_boundaries(start, end, count, convention="legacy", *, device=None):
    if convention not in {"legacy", "left_closed"}:
        raise ValueError("Unknown distance bin convention")
    if int(count) < 2 or float(end) <= float(start):
        raise ValueError("Distance bins require count >= 2 and end > start")
    return torch.linspace(float(start), float(end), int(count)-1, device=device)


def distance_bin_spec(start, end, count, convention="legacy"):
    return {"boundaries": distance_bin_boundaries(start, end, count, convention).tolist(),
            "convention": convention, "count": int(count)}


def distance_to_bins(
    distance: torch.Tensor,
    bin_start: float = DISTANCE_BIN_START_A,
    bin_end: float = DISTANCE_BIN_END_A,
    bin_count: int = DISTANCE_BIN_NUM,
    convention: str = "legacy",
) -> torch.Tensor:
    limits = distance_bin_boundaries(bin_start, bin_end, bin_count, convention, device=distance.device)
    return torch.bucketize(distance.float().contiguous(), limits, right=convention == "left_closed").long()


def contact_bin_count(
    contact_threshold: float,
    bin_start: float = DISTANCE_BIN_START_A,
    bin_end: float = DISTANCE_BIN_END_A,
    bin_count: int = DISTANCE_BIN_NUM,
    convention: str = "legacy",
) -> int:
    limits = distance_bin_boundaries(bin_start, bin_end, bin_count, convention)
    if convention == "left_closed":
        if not bool((limits == float(contact_threshold)).any()):
            raise ValueError("Contact threshold must be an exact distance-bin boundary")
        return int((limits <= float(contact_threshold)).sum().item())
    count = int(
        torch.bucketize(torch.tensor(float(contact_threshold)), limits).item()
    )
    return max(1, min(count, int(bin_count)))


def inter_chain_contact_mask(
    pairwise_distance: torch.Tensor,
    pairwise_mask: torch.Tensor,
    p1_length: int,
    contact_threshold: float,
) -> torch.Tensor:
    p1_length = int(p1_length)
    distance = pairwise_distance[:p1_length, p1_length:]
    valid = pairwise_mask[:p1_length, p1_length:].bool()
    return (distance < float(contact_threshold)) & valid


def has_inter_chain_contact(
    pairwise_distance: torch.Tensor,
    pairwise_mask: torch.Tensor,
    p1_length: int,
    contact_threshold: float,
) -> bool:
    return bool(
        inter_chain_contact_mask(
            pairwise_distance,
            pairwise_mask,
            p1_length,
            contact_threshold,
        ).any()
    )
