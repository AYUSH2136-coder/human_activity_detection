"""
Evaluation module initialization — exposing metrics and visualization tools.
"""

from src.evaluation.metrics import calculate_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.roc_curve import plot_roc_curve
from src.evaluation.feature_importance import plot_feature_importance
from src.evaluation.visualization import (
    plot_training_curves,
    plot_model_comparison,
)

__all__ = [
    "calculate_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_feature_importance",
    "plot_training_curves",
    "plot_model_comparison",
]
