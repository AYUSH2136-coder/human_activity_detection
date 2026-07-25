"""
Visualization helper module — plotting training history and model comparisons.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | Path,
    title: str = "Model Training History",
) -> None:
    """
    Plot loss and accuracy curves over epochs.

    Parameters
    ----------
    history : dict
        Dict containing lists for 'train_loss', 'val_loss', and optionally
        'train_acc', 'val_acc'.
    save_path : str | Path
        Path to save the generated plot.
    title : str
        Title of the plot.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot loss on primary y-axis
    color = "tab:red"
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", color=color, fontsize=11)
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color=color, linestyle="--", marker="o")
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], label="Val Loss", color="darkred", linestyle="-", marker="x")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Instantiate a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()  
    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color, fontsize=11)
    
    if "train_acc" in history and history["train_acc"]:
        ax2.plot(epochs, history["train_acc"], label="Train Acc", color=color, linestyle="--", marker="o")
    if "val_acc" in history and history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], label="Val Acc", color="darkblue", linestyle="-", marker="x")
    ax2.tick_params(axis="y", labelcolor=color)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True)
    
    plt.title(title, fontsize=14, pad=15, weight="bold")
    fig.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training history curves to {save_path}")


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: str | Path,
    metric: str = "Accuracy",
) -> None:
    """
    Plot a bar chart comparing performance of different models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns 'Model' and the metric specified.
    save_path : str | Path
        Path to save the generated plot.
    metric : str
        Metric column to plot (default: 'Accuracy').
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort models by metric for better visual structure
    df_sorted = comparison_df.sort_values(by=metric, ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(10, 6))
    
    # Using a modern palette
    sns.barplot(
        data=df_sorted,
        x="Model",
        y=metric,
        palette="crest",
        hue="Model",
        legend=False
    )
    
    plt.title(f"Model Performance Comparison ({metric})", fontsize=14, pad=15, weight="bold")
    plt.xlabel("Classifier Model", fontsize=11, labelpad=10)
    plt.ylabel(metric, fontsize=11, labelpad=10)
    plt.ylim(0, 1.05)
    
    # Annotate metrics above the bars
    for i, val in enumerate(df_sorted[metric]):
        plt.text(i, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontsize=10, weight="bold")
        
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved model comparison plot to {save_path}")
