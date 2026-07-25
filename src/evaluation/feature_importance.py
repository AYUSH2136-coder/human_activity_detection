"""
Feature importance extraction and visualization module.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_feature_importance(
    model: any,
    feature_names: list[str],
    save_path: str | Path,
    top_n: int = 15,
    title: str = "Feature Importance",
) -> None:
    """
    Extract and plot feature importance/coefficients from an ML model.

    Parameters
    ----------
    model : any
        Trained Scikit-learn, XGBoost, LightGBM, or CatBoost model.
    feature_names : list[str]
        List of all feature names.
    save_path : str | Path
        Path to save the generated figure.
    top_n : int
        Number of top features to display.
    title : str
        Title of the plot.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    importance = None
    
    # 1. Try feature_importances_ (Trees/Ensembles)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        logger.info("Extracted feature importances via feature_importances_")
        
    # 2. Try coef_ (Linear models like Logistic Regression)
    elif hasattr(model, "coef_"):
        # For multi-class, coef_ is (n_classes, n_features). Take mean absolute value.
        importance = np.mean(np.abs(model.coef_), axis=0)
        logger.info("Extracted feature importance weights via mean absolute coef_")
        
    # 3. For XGBoost JSON/binary models loaded back without sklearn wrapper
    elif hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            scores = booster.get_score(importance_type="gain")
            # Map score dict keys (e.g. 'f0', 'f1') to indices
            importance = np.zeros(len(feature_names))
            for k, v in scores.items():
                idx = int(k[1:]) if k.startswith("f") else int(k)
                if idx < len(importance):
                    importance[idx] = v
            logger.info("Extracted feature importances from raw XGBoost booster gain scores")
        except Exception as e:
            logger.warning(f"Could not extract XGBoost booster scores: {e}")
            
    if importance is None:
        logger.warning(f"Model type {type(model)} does not support feature importance/coefficient extraction.")
        return

    # Create a DataFrame for sorting and plotting
    if len(importance) != len(feature_names):
        logger.warning(
            f"Dimension mismatch: model importances size ({len(importance)}) "
            f"does not match feature names count ({len(feature_names)}). "
            f"Truncating/padding feature names."
        )
        if len(importance) < len(feature_names):
            feature_names = feature_names[:len(importance)]
        else:
            feature_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), len(importance))]

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })
    
    # Normalize importance to sum to 1 (for easy comparison)
    total_imp = df_imp["Importance"].sum()
    if total_imp > 0:
        df_imp["Importance"] = df_imp["Importance"] / total_imp
        
    # Sort and take top N
    df_imp = df_imp.sort_values(by="Importance", ascending=False).head(top_n).reset_index(drop=True)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=df_imp,
        x="Importance",
        y="Feature",
        palette="viridis",
        hue="Feature",
        legend=False
    )
    
    plt.title(title, fontsize=15, pad=15, weight="bold")
    plt.xlabel("Normalized Importance (relative gain/weight)", fontsize=11)
    plt.ylabel("Feature Name", fontsize=11)
    
    # Annotate bar values
    for i, v in enumerate(df_imp["Importance"]):
        plt.text(v + 0.002, i, f"{v:.4f}", va="center", ha="left", fontsize=9, weight="semibold")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance plot to {save_path}")
