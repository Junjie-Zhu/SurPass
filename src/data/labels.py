"""Distance/contact label helpers."""

from __future__ import annotations

import torch

DISTANCE_BIN_START_A = 2.0
DISTANCE_BIN_END_A = 22.0
DISTANCE_BIN_NUM = 16


def distance_to_bins(
    distance: torch.Tensor,
    bin_start: float = DISTANCE_BIN_START_A,
    bin_end: float = DISTANCE_BIN_END_A,
    bin_count: int = DISTANCE_BIN_NUM,
) -> torch.Tensor:
    limits = torch.linspace(
        float(bin_start),
        float(bin_end),
        int(bin_count) - 1,
        device=distance.device,
    )
    return torch.bucketize(distance.float(), limits).long()


def contact_bin_count(
    contact_threshold: float,
    bin_start: float = DISTANCE_BIN_START_A,
    bin_end: float = DISTANCE_BIN_END_A,
    bin_count: int = DISTANCE_BIN_NUM,
) -> int:
    limits = torch.linspace(
        float(bin_start),
        float(bin_end),
        int(bin_count) - 1,
    )
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
