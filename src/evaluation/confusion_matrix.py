"""
Confusion matrix visualization module.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    activity_names: list[str],
    save_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """
    Generate, plot, and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    activity_names : list[str]
        List of target activity names.
    save_path : str | Path
        Path to save the generated figure.
    title : str
        Title of the confusion matrix plot.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by row (true labels count)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # handle division by zero
    
    plt.figure(figsize=(12, 10))
    
    # Format annotations to show both absolute count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            if count > 0:
                annot[i, j] = f"{count}\n({pct:.1f}%)"
            else:
                annot[i, j] = "0"
                
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="crest",
        xticklabels=activity_names,
        yticklabels=activity_names,
        cbar=True,
        square=True
    )
    
    plt.title(title, fontsize=16, pad=20, weight="bold")
    plt.xlabel("Predicted Activity", fontsize=12, labelpad=15)
    plt.ylabel("True Activity", fontsize=12, labelpad=15)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix plot to {save_path}")
