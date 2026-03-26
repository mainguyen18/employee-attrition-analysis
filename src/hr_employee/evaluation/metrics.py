from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Core classification metrics (chỉ số đánh giá phân loại)."""

    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> ClassificationMetrics:
    y_pred = (y_prob >= threshold).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion_matrix=confusion_matrix(y_true, y_pred),
    )


