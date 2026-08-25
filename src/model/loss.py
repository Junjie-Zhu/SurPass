import torch
import torch.nn as nn


# BCE loss on only positive samples for pairwise binding map prediction
class PosOnlyBCELoss(nn.Module):
    def __init__(self):
        super(PosOnlyBCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pos_mask = target == 1
        mask = mask * pos_mask.to(dtype=mask.dtype)
        return (self.loss(logits, target) * mask).sum() / mask.sum().clamp(min=1.0)


class MaskedBinnedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = pair_mask.to(dtype=logits.dtype)[..., None]
        loss = self.loss(logits, target)
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bce = self.loss(logits, target)
        prob = torch.sigmoid(logits)
        p_t = target * prob + (1.0 - target) * (1.0 - prob)
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        focal_weight = alpha_t * ((1.0 - p_t).clamp(min=0.0) ** self.gamma)
        loss = focal_weight * bce
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = self.loss(logits, target)
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)


class FocalCrossEntropyLoss(nn.Module):
    """Multi-class focal CE with a larger weight on inter-chain pairs."""

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 1.5,
        inter_weight: float = 5.0,
        intra_weight: float = 1.0,
    ):
        super().__init__()
        if inter_weight < 0.0 or intra_weight < 0.0:
            raise ValueError("inter_weight and intra_weight must be non-negative.")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.inter_weight = float(inter_weight)
        self.intra_weight = float(intra_weight)
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def _pair_type_weight(
        self,
        p1_length: int,
        total_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weight = torch.full(
            (total_length, total_length),
            self.intra_weight,
            device=device,
            dtype=dtype,
        )
        p1_length = int(p1_length)
        weight[:p1_length, p1_length:] = self.inter_weight
        weight[p1_length:, :p1_length] = self.inter_weight
        return weight

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        p1_length: int,
    ) -> torch.Tensor:
        if logits.shape[:-1] != target.shape:
            raise ValueError(
                f"Logit/target shape mismatch: logits={tuple(logits.shape)} "
                f"target={tuple(target.shape)}"
            )
        ce = self.loss(logits.permute(0, 3, 1, 2), target.long())
        p_true = torch.exp(-ce)
        focal = self.alpha * ((1.0 - p_true).clamp(min=0.0) ** self.gamma) * ce
        pair_weight = self._pair_type_weight(
            p1_length,
            int(target.shape[-1]),
            device=focal.device,
            dtype=focal.dtype,
        )
        if focal.ndim == 3:
            pair_weight = pair_weight.unsqueeze(0)
        weight = pair_weight * mask.to(dtype=focal.dtype)
        return (focal * weight).sum() / weight.sum().clamp(min=1.0)