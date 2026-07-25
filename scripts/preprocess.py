"""
CLI script to preprocess data, engineer features, and perform train/test splitting.
"""

import argparse
from pathlib import Path

from src.data import (
    run_preprocessing_pipeline,
    run_feature_engineering_pipeline,
    run_splitting_pipeline,
)
from src.utils.helpers import load_yaml, get_project_root, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MHEALTH Preprocessing, Feature Engineering, and Splitting Script")
    parser.add_argument("--paths-config", type=str, default="configs/paths.yaml", help="Path to paths config file")
    parser.add_argument("--train-config", type=str, default="configs/training.yaml", help="Path to training/general config file")
    args = parser.parse_args()

    root = get_project_root()
    paths_cfg = load_yaml(resolve_path(args.paths_config, root))
    train_cfg = load_yaml(resolve_path(args.train_config, root))
    
    # Extract settings
    raw_csv = resolve_path(paths_cfg["data"]["raw_csv"], root)
    cleaned_csv = resolve_path("data/interim/mhealth_cleaned.csv", root) # default cleaned path
    scaler_path = resolve_path(paths_cfg["artifacts"]["scaler"], root)
    processed_dir = resolve_path(paths_cfg["data"]["processed_dir"], root)
    feat_names_path = resolve_path(paths_cfg["data"]["feature_names"], root)
    
    window_size = train_cfg.get("lstm", {}).get("window_size", 128) # or default to 128
    # Let's check window size from training config
    # In training.yaml, lstm/gru window size is 50. But for CNN/MLP/General, let's use 128 (default) or check if specified.
    # Let's check training.yaml window_size:
    # lstm: window_size: 50, gru: window_size: 50
    # Let's support window_size as 128 by default or take 128 since it's the standard for feature engineering.
    window_size = 128
    step_size = 64
    
    logger.info("========== Starting Data Pipeline ==========")
    
    # 1. Preprocessing
    cleaned_df = run_preprocessing_pipeline(
        raw_path=raw_csv,
        cleaned_output_path=cleaned_csv,
        scaler_path=scaler_path
    )
    
    # 2. Feature Engineering
    X_raw, X_eng, y, feat_names = run_feature_engineering_pipeline(
        cleaned_df=cleaned_df,
        window_size=window_size,
        step_size=step_size,
        output_dir=processed_dir,
        feature_names_path=feat_names_path
    )
    
    # 3. Train/Test Splitting
    run_splitting_pipeline(
        X_raw=X_raw,
        X_eng=X_eng,
        y=y,
        paths_cfg=paths_cfg,
        test_size=0.2,
        random_state=train_cfg.get("general", {}).get("random_seed", 42)
    )
    
    logger.info("========== Data Pipeline Completed Successfully! ==========")


if __name__ == "__main__":
    main()
