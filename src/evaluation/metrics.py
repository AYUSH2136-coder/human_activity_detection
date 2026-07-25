"""
Metrics calculation module — computing accuracy, precision, recall, and F1-score.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_metrics(
    y_true: list | tuple | np.ndarray,
    y_pred: list | tuple | np.ndarray,
    activity_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Calculate and return a dictionary of standard evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    activity_names : list[str] | None
        List of target activity name strings.

    Returns
    -------
    metrics : dict
        Dictionary of calculated metric values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }
    
    logger.info(f"Calculated metrics: Accuracy={accuracy:.4f}, F1-Macro={f1_macro:.4f}, F1-Weighted={f1_weighted:.4f}")
    
    # Generate and print classification report
    report = classification_report(
        y_true, y_pred, target_names=activity_names, zero_division=0
    )
    logger.info(f"\nClassification Report:\n{report}")
    
    return metrics
