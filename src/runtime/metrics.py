from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, roc_curve


def binary_classification_metrics(
    scores: torch.Tensor | Iterable[float],
    target: torch.Tensor | Iterable[bool],
    *,
    positive_weight: float = 1.0,
) -> dict[str, float]:
    scores_array = torch.as_tensor(scores).detach().flatten().float().cpu().numpy()
    target_array = (
        torch.as_tensor(target).detach().flatten().bool().cpu().numpy()
    )
    positives = int(target_array.sum())
    negatives = int(target_array.size - positives)
    weight = float(positive_weight)
    if (
        scores_array.size == 0
        or positives == 0
        or negatives == 0
        or weight <= 0.0
    ):
        return {"auroc": float("nan"), "auprc": float("nan")}
    sample_weight = None
    if weight != 1.0:
        sample_weight = 1.0 + (weight - 1.0) * target_array.astype(np.float64)
    false_positive, true_positive, _ = roc_curve(
        target_array,
        scores_array,
        sample_weight=sample_weight,
    )
    return {
        "auroc": float(auc(false_positive, true_positive)),
        "auprc": float(
            average_precision_score(
                target_array,
                scores_array,
                sample_weight=sample_weight,
            )
        ),
    }


def metrics_from_rows(
    rows: list[dict[str, Any]],
    score_key: str,
    *,
    positive_weight: float = 1.0,
    target_key: str = "Category",
) -> dict[str, float | int]:
    scores = [float(row[score_key]) for row in rows]
    targets = [bool(row[target_key]) for row in rows]
    classification = binary_classification_metrics(
        scores,
        targets,
        positive_weight=positive_weight,
    )
    positives = sum(targets)
    return {
        **classification,
        "n": len(targets),
        "n_pos": positives,
        "n_neg": len(targets) - positives,
    }
