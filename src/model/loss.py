import torch
import torch.nn as nn
import torch.nn.functional as F


def distance_bin_centers(
    bin_start: float,
    bin_end: float,
    bin_count: int,
) -> torch.Tensor:
    bin_count = int(bin_count)
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2.")
    limits = torch.linspace(float(bin_start), float(bin_end), bin_count - 1)
    centers = torch.empty(bin_count, dtype=torch.float32)
    if bin_count == 2:
        width = float(bin_end) - float(bin_start)
        centers[0] = limits[0] - 0.5 * width
        centers[1] = limits[-1] + 0.5 * width
        return centers
    first_width = limits[1] - limits[0]
    last_width = limits[-1] - limits[-2]
    centers[0] = limits[0] - 0.5 * first_width
    centers[1:-1] = 0.5 * (limits[:-1] + limits[1:])
    centers[-1] = limits[-1] + 0.5 * last_width
    return centers


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(dtype=loss.dtype)
    return (loss * mask_float).sum() / mask_float.sum().clamp_min(1.0)


# def downsample_inter_negatives(
#     inter_mask: torch.Tensor,
#     target_bins: torch.Tensor,
#     contact_bins: int,
#     neg_per_pos: float = 4.0,
# ) -> torch.Tensor:
#     """Keep all inter contacts and ~neg_per_pos random inter non-contacts per contact."""
#     if float(neg_per_pos) <= 0.0:
#         return inter_mask.to(dtype=torch.bool)

#     inter_mask = inter_mask.to(dtype=torch.bool)
#     is_contact = target_bins.long() < int(contact_bins)
#     keep = inter_mask & is_contact
#     neg = inter_mask & ~is_contact
#     squeezed = False
#     if keep.ndim == 2:
#         keep = keep.unsqueeze(0)
#         neg = neg.unsqueeze(0)
#         squeezed = True

#     out = keep.clone()
#     ratio = float(neg_per_pos)
#     for batch_index in range(keep.shape[0]):
#         n_pos = int(keep[batch_index].sum().item())
#         neg_flat = torch.nonzero(neg[batch_index].reshape(-1), as_tuple=False).flatten()
#         n_neg = int(neg_flat.numel())
#         if n_neg == 0:
#             continue
#         n_keep_neg = (
#             min(n_neg, max(1, int(round(n_pos * ratio))))
#             if n_pos > 0
#             else min(n_neg, max(1, int(round(ratio))))
#         )
#         order = torch.randperm(n_neg, device=neg_flat.device)[:n_keep_neg]
#         selected = neg_flat[order]
#         flat = out[batch_index].reshape(-1)
#         flat[selected] = True
#         out[batch_index] = flat.view_as(out[batch_index])

#     if squeezed:
#         return out.squeeze(0)
#     return out


def distogram_supervision(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    contact_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One contact label and one softmax, shared by every pair loss and metric."""
    contact_bins = int(contact_bins)
    if logits.shape[-1] < contact_bins:
        raise ValueError(
            f"contact_bins {contact_bins} exceeds logit classes {logits.shape[-1]}."
        )
    is_contact = target_bins.long() < contact_bins
    pair_probs = torch.softmax(logits, dim=-1)
    contact_probability = inter_chain_contact_probability(pair_probs, contact_bins)
    return is_contact, pair_probs, contact_probability


def _validate_pair_shapes(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    *masks: torch.Tensor,
) -> None:
    if logits.shape[:-1] != target_bins.shape:
        raise ValueError(
            f"Logit/target shape mismatch: logits={tuple(logits.shape)} "
            f"target={tuple(target_bins.shape)}"
        )
    for mask in masks:
        if tuple(mask.shape) != tuple(target_bins.shape):
            raise ValueError(
                f"Mask/target shape mismatch: mask={tuple(mask.shape)} "
                f"target={tuple(target_bins.shape)}"
            )


class DistogramCELoss(nn.Module):
    """Masked cross-entropy over pairwise distance bins."""

    def forward(
        self,
        logits: torch.Tensor,
        target_bins: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # _validate_pair_shapes(logits, target_bins, mask)
        loss = F.cross_entropy(
            logits.permute(0, 3, 1, 2),
            target_bins.long(),
            reduction="none",
        )
        return _masked_mean(loss, mask)


class FocalCELoss(nn.Module):
    """Focal CE with a flat contact-bin alpha on both intra and inter pairs."""

    def __init__(
        self,
        contact_bins: int,
        gamma: float = 1.5,
        alpha: float = 5.0,
    ):
        super().__init__()
        if int(contact_bins) <= 0:
            raise ValueError("contact_bins must be positive.")
        self.contact_bins = int(contact_bins)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(
        self,
        logits: torch.Tensor,
        target_bins: torch.Tensor,
        intra_mask: torch.Tensor,
        inter_mask: torch.Tensor,
        is_contact: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # _validate_pair_shapes(logits, target_bins, intra_mask, inter_mask)
        target = target_bins.long()
        ce_loss = F.cross_entropy(
            logits.permute(0, 3, 1, 2),
            target,
            reduction="none",
        )

        p_t = torch.exp(-ce_loss)
        focal = ((1.0 - p_t).clamp(min=0.0) ** self.gamma) * ce_loss
        pos_weight = torch.where(
            is_contact.to(dtype=torch.bool),
            focal.new_tensor(self.alpha),
            focal.new_tensor(1.0),
        )
        focal = focal * pos_weight
        return _masked_mean(focal, intra_mask), _masked_mean(focal, inter_mask)


class DistogramMAELoss(nn.Module):
    """L1 loss between expected bin distance and the target bin center."""

    def __init__(
        self,
        bin_centers: torch.Tensor | None = None,
        bin_start: float = 2.0,
        bin_end: float = 22.0,
        bin_count: int = 64,
    ):
        super().__init__()
        if bin_centers is None:
            bin_centers = distance_bin_centers(bin_start, bin_end, bin_count)
        self.register_buffer("bin_centers", bin_centers.to(dtype=torch.float32))

    def forward(
        self,
        pair_probs: torch.Tensor,
        target_bins: torch.Tensor,
        intra_mask: torch.Tensor,
        inter_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # _validate_pair_shapes(pair_probs, target_bins, intra_mask, inter_mask)
        centers = self.bin_centers.to(device=pair_probs.device, dtype=pair_probs.dtype)
        if pair_probs.shape[-1] != int(centers.shape[0]):
            raise ValueError(
                f"Logit class count {pair_probs.shape[-1]} does not match "
                f"bin_centers {int(centers.shape[0])}."
            )
        pred_dist = (pair_probs * centers).sum(dim=-1)
        target_dist = centers[target_bins.long().clamp(0, centers.shape[0] - 1)]
        loss = F.l1_loss(pred_dist, target_dist, reduction="none")
        return (
            _masked_mean(loss, intra_mask),
            _masked_mean(loss, inter_mask),
        )


class TverskyLoss(nn.Module):
    """Soft Tversky loss on P(d < threshold) versus contact-bin labels."""

    def __init__(
        self,
        contact_bins: int,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1.0,
    ):
        super().__init__()
        if int(contact_bins) <= 0:
            raise ValueError("contact_bins must be positive.")
        self.contact_bins = int(contact_bins)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(
        self,
        contact_probability: torch.Tensor,
        is_contact: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_float = mask.to(dtype=contact_probability.dtype)
        p_contact = contact_probability
        y_contact = is_contact.to(dtype=contact_probability.dtype)
        tp = (p_contact * y_contact * mask_float).sum()
        fp = (p_contact * (1.0 - y_contact) * mask_float).sum()
        fn = ((1.0 - p_contact) * y_contact * mask_float).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1.0 - tversky


def residue_bind_targets(
    is_contact: torch.Tensor,
    inter_mask: torch.Tensor,
) -> torch.Tensor:
    """Residue is positive if it has any inter-chain contact."""
    inter_contact = is_contact.to(dtype=torch.bool) & inter_mask.to(dtype=torch.bool)
    return inter_contact.any(dim=-1)


def downsample_residue_negatives(
    residue_mask: torch.Tensor,
    target: torch.Tensor,
    neg_per_pos: float = 4.0,
) -> torch.Tensor:
    """Keep all interface residues and ~neg_per_pos non-interface residues per positive."""
    residue_mask = residue_mask.to(dtype=torch.bool)
    target = target.to(dtype=torch.bool)
    if float(neg_per_pos) <= 0.0:
        return residue_mask

    keep = residue_mask & target
    neg = residue_mask & ~target
    squeezed = False
    if keep.ndim == 1:
        keep = keep.unsqueeze(0)
        neg = neg.unsqueeze(0)
        squeezed = True

    out = keep.clone()
    ratio = float(neg_per_pos)
    for batch_index in range(keep.shape[0]):
        n_pos = int(keep[batch_index].sum().item())
        neg_flat = torch.nonzero(neg[batch_index], as_tuple=False).flatten()
        n_neg = int(neg_flat.numel())
        if n_neg == 0:
            continue
        n_keep_neg = (
            min(n_neg, max(1, int(round(n_pos * ratio))))
            if n_pos > 0
            else min(n_neg, max(1, int(round(ratio))))
        )
        order = torch.randperm(n_neg, device=neg_flat.device)[:n_keep_neg]
        out[batch_index, neg_flat[order]] = True

    if squeezed:
        return out.squeeze(0)
    return out


class ResidueBindLoss(nn.Module):
    """Masked BCE on per-residue interface labels. Down-weights pair-negative examples."""

    def __init__(
        self,
        pos_weight: float = 4.0,
        neg_per_pos: float = 4.0,
        negative_pair_weight: float = 0.25,
    ):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.neg_per_pos = float(neg_per_pos)
        self.negative_pair_weight = float(negative_pair_weight)

    def forward(
        self,
        residue_logits: torch.Tensor,
        residue_target: torch.Tensor,
        residue_mask: torch.Tensor,
        is_positive: torch.Tensor,
    ) -> torch.Tensor:
        logits = residue_logits.reshape(residue_target.shape)
        target = residue_target

        loss = F.binary_cross_entropy_with_logits(
            logits,
            target.to(dtype=logits.dtype),
            reduction="none",
        )
        pair_weight = torch.where(
            is_positive.to(device=logits.device, dtype=torch.bool).reshape(-1),
            loss.new_tensor(1.0),
            loss.new_tensor(self.negative_pair_weight),
        )
        loss = loss * pair_weight[:, None]
        return _masked_mean(loss, residue_mask)



def inter_chain_contact_probability(
    pair_probs: torch.Tensor,
    contact_bins: int,
) -> torch.Tensor:
    return pair_probs[..., : int(contact_bins)].sum(dim=-1)


def noisy_or_over_partners(
    contact_probability: torch.Tensor,
    inter_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-residue probability that any inter-chain partner is in contact."""
    probability = contact_probability.clamp(0.0, 1.0 - 1e-4)
    log_keep = torch.where(
        inter_mask.to(dtype=torch.bool),
        torch.log1p(-probability),
        torch.zeros_like(probability),
    )
    return (1.0 - torch.exp(log_keep.sum(dim=-1))).clamp(0.0, 1.0)


class ProteinBindLoss(nn.Module):
    """BCE on the complex-level interaction logit."""

    def forward(
        self,
        protein_logit: torch.Tensor,
        is_positive: torch.Tensor,
    ) -> torch.Tensor:
        target = is_positive.reshape(-1).to(dtype=protein_logit.dtype)
        return F.binary_cross_entropy_with_logits(protein_logit.reshape(-1), target)


class ConsistencyLoss(nn.Module):
    """Residue head tracks the distogram, and a visible contact lifts the protein score."""

    def __init__(self, contact_bins: int):
        super().__init__()
        if int(contact_bins) <= 0:
            raise ValueError("contact_bins must be positive.")
        self.contact_bins = int(contact_bins)

    def forward(
        self,
        residue_logits: torch.Tensor,
        protein_logit: torch.Tensor,
        contact_probability: torch.Tensor,
        inter_mask: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        residue_target = noisy_or_over_partners(contact_probability, inter_mask).detach()
        residue_loss = F.binary_cross_entropy_with_logits(
            residue_logits.reshape(residue_target.shape),
            residue_target,
            reduction="none",
        )
        residue_loss = _masked_mean(residue_loss, residue_mask)

        inter_contact = contact_probability.masked_fill(~inter_mask.to(dtype=torch.bool), 0.0)
        max_contact = inter_contact.amax(dim=(-2, -1)).detach()
        protein_probability = torch.sigmoid(protein_logit.reshape(-1))
        hinge = torch.relu(max_contact - protein_probability).mean()
        return residue_loss + hinge
