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
        