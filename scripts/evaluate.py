"""
Script to aggregate ML and DL model results, compare their performance,
identify the best model, and save comparison artifacts.
"""

import argparse
from pathlib import Path
import pandas as pd

from src.evaluation import plot_model_comparison
from src.utils.helpers import load_yaml, get_project_root, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and Compare Trained HAD Models")
    parser.add_argument("--paths-config", type=str, default="configs/paths.yaml", help="Path to paths config file")
    args = parser.parse_args()

    root = get_project_root()
    paths_cfg = load_yaml(resolve_path(args.paths_config, root))
    
    ml_results_path = resolve_path(paths_cfg["outputs"]["ml_results"], root)
    dl_results_path = resolve_path(paths_cfg["outputs"]["dl_results"], root)
    
    # Check if results files exist
    ml_exists = ml_results_path.exists()
    dl_exists = dl_results_path.exists()
    
    if not ml_exists and not dl_exists:
        logger.error("No training results found. Run train_ml.py or train_dl.py first.")
        return
        
    dfs = []
    if ml_exists:
        dfs.append(pd.read_csv(ml_results_path))
        logger.info(f"Loaded ML results from {ml_results_path}")
    if dl_exists:
        dfs.append(pd.read_csv(dl_results_path))
        logger.info(f"Loaded DL results from {dl_results_path}")
        
    # Concatenate and sort
    df_compare = pd.concat(dfs, ignore_index=True)
    df_compare = df_compare.sort_values(by="accuracy", ascending=False).reset_index(drop=True)
    
    # Save comparison table
    compare_csv = resolve_path(paths_cfg["outputs"]["comparison_table"], root)
    df_compare.to_csv(compare_csv, index=False)
    logger.info(f"Saved aggregated model comparison to {compare_csv}")
    
    # Plot model comparison
    fig_dir = resolve_path(paths_cfg["outputs"]["figures_dir"], root)
    fig_path = fig_dir / "model_comparison.png"
    plot_model_comparison(
        comparison_df=df_compare,
        save_path=fig_path,
        metric="accuracy"
    )
    
    # Identify and save the best model name
    best_model_name = df_compare.loc[0, "Model"]
    best_model_acc = df_compare.loc[0, "accuracy"]
    best_model_txt = resolve_path(paths_cfg["outputs"]["best_model_name"], root)
    
    with open(best_model_txt, "w") as f:
        f.write(best_model_name)
        
    logger.info(f"\n==================== BEST MODEL IDENTIFIED ====================")
    logger.info(f"Model: {best_model_name.upper()} with Test Accuracy: {best_model_acc:.4f}")
    logger.info(f"Saved best model name to {best_model_txt}")
    logger.info(f"===============================================================")
    
    print("\nModel Comparison Table (Sorted by Accuracy):")
    print(df_compare[["Model", "accuracy", "f1_macro", "f1_weighted"]])


if __name__ == "__main__":
    main()
