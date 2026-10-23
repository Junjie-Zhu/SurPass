"""Contact-focused losses and score pooling for SurPass."""

from __future__ import annotations

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


def max_contact_probability(
    contact_probability: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Inference score: one above-threshold valid contact is sufficient."""
    return contact_probability.float().masked_fill(
        ~pair_mask.bool(), 0.0
    ).flatten(1).amax(dim=1)


def pair_contact_logit(pair_logits: torch.Tensor, contact_bins: int) -> torch.Tensor:
    """Binary contact logits without an intermediate rounded probability."""
    contact_bins = int(contact_bins)
    if pair_logits.ndim != 4 or not 0 < contact_bins < pair_logits.shape[-1]:
        raise ValueError("contact logits require both contact and non-contact bins")
    logits = pair_logits.float()
    return torch.logsumexp(logits[..., :contact_bins], dim=-1) - torch.logsumexp(
        logits[..., contact_bins:], dim=-1
    )


def noisy_or_contact_logit(
    pair_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    contact_bins: int,
) -> torch.Tensor:
    """Logit of 1 - product(1 - p_ij), evaluated in float32 log space.

    Compute binary contact logits directly from the categorical logits. This
    avoids rounding p to one, and retains negative-bag gradients even when
    the noisy-OR probability would round to one. No length normalization is
    applied: this is an OR over valid contacts, not a mean contact score.
    Empty bags have a finite near-zero probability and no gradient.
    """
    contact_logit = pair_contact_logit(pair_logits, contact_bins)
    hazard = F.softplus(contact_logit).masked_fill(~pair_mask.bool(), 0.0)
    hazard = hazard.flatten(1).sum(dim=1)
    hazard = hazard.clamp_min(torch.finfo(hazard.dtype).tiny)
    return hazard + torch.log(-torch.expm1(-hazard))


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class ProteinBindLoss(nn.Module):
    def forward(
        self,
        protein_logit: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            protein_logit.reshape(-1),
            target.reshape(-1).to(dtype=protein_logit.dtype),
        )


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
    ) -> torch.Tensor:
        contact_logits = pair_contact_logit(pair_logits, self.contact_bins)
        target = target_bins.long() < self.contact_bins
        losses: list[torch.Tensor] = []
        for sample_logit, sample_target, sample_mask in zip(
            contact_logits,
            target,
            pair_mask,
        ):
            valid = sample_mask.to(dtype=torch.bool)
            logit = sample_logit[valid]
            p = logit.sigmoid()
            y = sample_target[valid]
            if p.numel() == 0:
                continue
            positive_loss = ((-logit).sigmoid() ** self.gamma) * F.softplus(-logit)
            negative_loss = (p**self.gamma) * F.softplus(logit)
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
            losses.append(sample_loss)
        if not losses:
            return pair_logits.sum() * 0.0
        return torch.stack(losses).mean()


class InterDistanceLoss(nn.Module):
    def forward(
        self,
        pair_logits: torch.Tensor,
        target_bins: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.cross_entropy(
            pair_logits.permute(0, 3, 1, 2),
            target_bins.long(),
            reduction="none",
        )
        return _masked_mean(loss, pair_mask)


class SurpassLoss(nn.Module):
    """Noisy-OR protein supervision plus direct contact/distance supervision."""

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
        terms = {
            "protein": self.protein(
                output.protein_logit,
                labels["crop_is_positive"],
            ),
            "contact": self.contact(
                output.pair_logits,
                labels["label_inter_bins"],
                labels["label_inter_mask"],
            ),
            "distance": self.distance(
                output.pair_logits,
                labels["label_inter_bins"],
                labels["label_inter_mask"],
            ),
            "aux_protein": self.protein(
                output.aux_protein_logit,
                labels["source_is_positive"],
            ),
        }
        total = output.protein_logit.new_zeros(())
        for name, value in terms.items():
            total = total + self.weights.get(name, 0.0) * value
        detached = {name: value.detach() for name, value in terms.items()}
        detached["total"] = total.detach()
        return total, detached
