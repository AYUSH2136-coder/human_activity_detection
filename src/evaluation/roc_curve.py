"""
ROC curve calculation and visualization module for multi-class classification.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    activity_names: list[str],
    save_path: str | Path,
    title: str = "Receiver Operating Characteristic (ROC)",
) -> None:
    """
    Compute and plot One-vs-Rest multi-class ROC curves and AUC.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels (0-indexed).
    y_score : np.ndarray
        Predicted probabilities of shape (num_samples, num_classes).
    activity_names : list[str]
        List of target activity names.
    save_path : str | Path
        Path to save the generated figure.
    title : str
        Title of the ROC plot.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_classes = len(activity_names)
    
    # Binarize labels in One-vs-Rest fashion
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # Check if binarization went okay (e.g. class counts)
    if y_true_bin.shape[1] != n_classes:
        logger.warning(
            f"Binarization class count mismatch: y_true_bin shape {y_true_bin.shape} "
            f"vs class list size {n_classes}. Skipping ROC plotting."
        )
        return
        
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Plot micro-average ROC
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"Micro-average ROC (AUC = {roc_auc['micro']:.3f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    
    # Plot class-specific ROCs with a colormap
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=1.5,
            label=f"{activity_names[i]} (AUC = {roc_auc[i]:.3f})",
        )
        
    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(title, fontsize=14, pad=15, weight="bold")
    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curve plot to {save_path}")
