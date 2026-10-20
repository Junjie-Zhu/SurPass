"""Contact-focused losses and score pooling for SurPass."""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn


def distance_bin_centers(
    bin_start: float,
    bin_end: float,
    bin_count: int,
) -> torch.Tensor:
    bin_count = int(bin_count)
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2")
    limits = torch.linspace(float(bin_start), float(bin_end), bin_count - 1)
    centers = torch.empty(bin_count, dtype=torch.float32)
    if bin_count == 2:
        width = float(bin_end) - float(bin_start)
        centers[0] = limits[0] - 0.5 * width
        centers[1] = limits[-1] + 0.5 * width
        return centers
    centers[0] = limits[0] - 0.5 * (limits[1] - limits[0])
    centers[1:-1] = 0.5 * (limits[:-1] + limits[1:])
    centers[-1] = limits[-1] + 0.5 * (limits[-1] - limits[-2])
    return centers


def pair_contact_probability(
    pair_logits: torch.Tensor,
    contact_bins: int,
) -> torch.Tensor:
    if pair_logits.ndim != 4:
        raise ValueError(
            f"pair_logits must be [B,L1,L2,C], got {tuple(pair_logits.shape)}"
        )
    contact_bins = int(contact_bins)
    if not 0 < contact_bins < pair_logits.shape[-1] + 1:
        raise ValueError("contact_bins is incompatible with pair logits")
    return torch.softmax(pair_logits.float(), dim=-1)[..., :contact_bins].sum(dim=-1)


def pooled_contact_probability(
    contact_probability: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    topk_fraction: float = 0.01,
    maximum_topk: int = 64,
) -> torch.Tensor:
    """Length-normalized mean of the strongest inter-chain contacts."""
    probabilities: list[torch.Tensor] = []
    fraction = float(topk_fraction)
    if fraction <= 0.0:
        raise ValueError("topk_fraction must be positive")
    for sample_probability, sample_mask in zip(contact_probability, pair_mask):
        valid = sample_probability[sample_mask.to(dtype=torch.bool)]
        if valid.numel() == 0:
            probabilities.append(sample_probability.new_zeros(()))
            continue
        count = max(1, int(math.ceil(valid.numel() * fraction)))
        count = min(int(maximum_topk), count, int(valid.numel()))
        probabilities.append(valid.topk(count, sorted=False).values.mean())
    return torch.stack(probabilities)


def probability_to_logit(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(probability) - torch.log1p(-probability)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class ProteinBindLoss(nn.Module):
    def forward(
        self,
        protein_logit: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            protein_logit.reshape(-1),
            target.reshape(-1).to(dtype=protein_logit.dtype),
            reduction="none",
        )
        if sample_weight is None:
            return loss.mean()
        weight = sample_weight.reshape(-1).to(device=loss.device, dtype=loss.dtype)
        return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


class InterContactFocalLoss(nn.Module):
    """Balanced focal loss with deterministic hard far-pair selection."""

    def __init__(
        self,
        contact_bins: int,
        *,
        gamma: float = 1.5,
        positive_weight: float = 4.0,
        negatives_per_positive: float = 8.0,
        negatives_when_empty: int = 256,
    ):
        super().__init__()
        self.contact_bins = int(contact_bins)
        self.gamma = float(gamma)
        self.positive_weight = float(positive_weight)
        self.negatives_per_positive = float(negatives_per_positive)
        self.negatives_when_empty = int(negatives_when_empty)

    def forward(
        self,
        pair_logits: torch.Tensor,
        target_bins: torch.Tensor,
        pair_mask: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probability = pair_contact_probability(pair_logits, self.contact_bins)
        target = target_bins.long() < self.contact_bins
        losses: list[torch.Tensor] = []
        used_weights: list[torch.Tensor] = []
        for sample_index, (sample_probability, sample_target, sample_mask) in enumerate(zip(
            probability,
            target,
            pair_mask,
        )):
            valid = sample_mask.to(dtype=torch.bool)
            p = sample_probability[valid].clamp(1e-6, 1.0 - 1e-6)
            y = sample_target[valid]
            if p.numel() == 0:
                continue
            positive_loss = -((1.0 - p) ** self.gamma) * torch.log(p)
            negative_loss = -(p**self.gamma) * torch.log1p(-p)
            positive_values = positive_loss[y]
            negative_values = negative_loss[~y]
            if positive_values.numel():
                positive_term = positive_values.mean() * self.positive_weight
                keep_negative = max(
                    1,
                    int(round(positive_values.numel() * self.negatives_per_positive)),
                )
            else:
                positive_term = p.new_zeros(())
                keep_negative = self.negatives_when_empty
            if negative_values.numel():
                keep_negative = min(keep_negative, int(negative_values.numel()))
                negative_term = negative_values.topk(
                    keep_negative,
                    sorted=False,
                ).values.mean()
            else:
                negative_term = p.new_zeros(())
            sample_loss = positive_term + negative_term
            if sample_weight is not None:
                weight = sample_weight[sample_index].to(
                    device=sample_loss.device,
                    dtype=sample_loss.dtype,
                )
                sample_loss = sample_loss * weight
                used_weights.append(weight)
            losses.append(sample_loss)
        if not losses:
            return pair_logits.sum() * 0.0
        if used_weights:
            return torch.stack(losses).sum() / torch.stack(used_weights).sum().clamp_min(
                1e-6
            )
        return torch.stack(losses).mean()


class InterDistanceLoss(nn.Module):
    def forward(
        self,
        pair_logits: torch.Tensor,
        target_bins: torch.Tensor,
        pair_mask: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = F.cross_entropy(
            pair_logits.permute(0, 3, 1, 2),
            target_bins.long(),
            reduction="none",
        )
        if sample_weight is None:
            return _masked_mean(loss, pair_mask)
        weight = pair_mask.to(dtype=loss.dtype) * sample_weight.to(
            device=loss.device,
            dtype=loss.dtype,
        )[:, None, None]
        return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


class SurpassLoss(nn.Module):
    """One explicit objective matching the primary pair-derived score."""

    def __init__(
        self,
        contact_bins: int,
        weights: Mapping[str, float],
        *,
        focal_gamma: float = 1.5,
        contact_positive_weight: float = 4.0,
        negatives_per_positive: float = 8.0,
    ):
        super().__init__()
        self.weights = {name: float(value) for name, value in weights.items()}
        self.protein = ProteinBindLoss()
        self.contact = InterContactFocalLoss(
            contact_bins,
            gamma=focal_gamma,
            positive_weight=contact_positive_weight,
            negatives_per_positive=negatives_per_positive,
        )
        self.distance = InterDistanceLoss()

    def forward(self, output, labels: Mapping[str, torch.Tensor]):
        sample_weight = labels.get("sample_confidence")
        terms = {
            "protein": self.protein(
                output.protein_logit,
                labels["crop_is_positive"],
                sample_weight,
            ),
            "contact": self.contact(
                output.pair_logits,
                labels["label_inter_bins"],
                labels["label_inter_mask"],
                sample_weight,
            ),
            "distance": self.distance(
                output.pair_logits,
                labels["label_inter_bins"],
                labels["label_inter_mask"],
                sample_weight,
            ),
            "aux_protein": self.protein(
                output.aux_protein_logit,
                labels["source_is_positive"],
                sample_weight,
            ),
        }
        total = output.protein_logit.new_zeros(())
        for name, value in terms.items():
            total = total + self.weights.get(name, 0.0) * value
        detached = {name: value.detach() for name, value in terms.items()}
        detached["total"] = total.detach()
        return total, detached
