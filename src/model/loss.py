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
    """Multi-class focal CE with extra weight on contact bins (d < threshold)."""

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
        pair_weight = torch.where(
            target < self.contact_bins,
            focal.new_tensor(self.alpha),
            focal.new_tensor(1.0),
        )
        return (
            _masked_mean(focal * pair_weight, intra_mask),
            _masked_mean(focal * pair_weight, inter_mask),
        )


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
        logits: torch.Tensor,
        target_bins: torch.Tensor,
        intra_mask: torch.Tensor,
        inter_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # _validate_pair_shapes(logits, target_bins, intra_mask, inter_mask)
        centers = self.bin_centers.to(device=logits.device, dtype=logits.dtype)
        if logits.shape[-1] != int(centers.shape[0]):
            raise ValueError(
                f"Logit class count {logits.shape[-1]} does not match "
                f"bin_centers {int(centers.shape[0])}."
            )
        probs = torch.softmax(logits, dim=-1)
        pred_dist = (probs * centers).sum(dim=-1)
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
        logits: torch.Tensor,
        target_bins: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # _validate_pair_shapes(logits, target_bins, mask)
        if logits.shape[-1] < self.contact_bins:
            raise ValueError(
                f"contact_bins {self.contact_bins} exceeds logit classes {logits.shape[-1]}."
            )
        mask_float = mask.to(dtype=logits.dtype)
        p_contact = torch.softmax(logits, dim=-1)[..., : self.contact_bins].sum(dim=-1)
        y_contact = (target_bins.long() < self.contact_bins).to(dtype=logits.dtype)
        tp = (p_contact * y_contact * mask_float).sum()
        fp = (p_contact * (1.0 - y_contact) * mask_float).sum()
        fn = ((1.0 - p_contact) * y_contact * mask_float).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1.0 - tversky
