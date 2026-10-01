"""Original-unit metrics and validation-only threshold selection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    """Report class-1 churn metrics at the frozen threshold."""
    pred = probability >= threshold
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def regression(y: np.ndarray, pred: np.ndarray) -> dict:
    """Report MAE, MSE, RMSE and R-squared on untrimmed outcomes."""
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "mse": float(mean_squared_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def choose_threshold(y: np.ndarray, probabilities: np.ndarray, minimum_recall: float) -> float:
    """Maximize validation F1 among thresholds meeting the recall requirement."""
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = probabilities >= threshold
        if recall_score(y, pred, zero_division=0) >= minimum_recall:
            candidates.append((f1_score(y, pred, zero_division=0), float(threshold)))
    return max(candidates)[1] if candidates else 0.0
