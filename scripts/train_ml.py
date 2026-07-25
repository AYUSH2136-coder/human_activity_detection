"""
Script to train, evaluate, and save traditional Machine Learning models.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json

from src.models import get_ml_model
from src.evaluation import calculate_metrics, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from src.data.loader import get_activity_names
from src.utils.helpers import load_yaml, get_project_root, resolve_path, ensure_dirs
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Traditional Machine Learning Models")
    parser.add_argument("--paths-config", type=str, default="configs/paths.yaml", help="Path to paths config file")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Path to model hyperparams config file")
    parser.add_argument("--train-config", type=str, default="configs/training.yaml", help="Path to training config for seed")
    args = parser.parse_args()

    root = get_project_root()
    paths_cfg = load_yaml(resolve_path(args.paths_config, root))
    model_cfg = load_yaml(resolve_path(args.model_config, root))
    train_cfg = load_yaml(resolve_path(args.train_config, root))
    
    # Set seed
    seed = train_cfg.get("general", {}).get("random_seed", 42)
    set_seed(seed)
    
    # Ensure directories exist
    ensure_dirs(
        resolve_path(paths_cfg["models"]["ml_dir"], root),
        resolve_path(paths_cfg["outputs"]["figures_dir"], root)
    )
    
    # Load data
    logger.info("Loading training and testing data...")
    X_train = np.load(resolve_path(paths_cfg["data"]["X_train_eng"], root))
    X_test = np.load(resolve_path(paths_cfg["data"]["X_test_eng"], root))
    y_train = np.load(resolve_path(paths_cfg["data"]["y_train"], root))
    y_test = np.load(resolve_path(paths_cfg["data"]["y_test"], root))
    
    # Load feature names
    with open(resolve_path(paths_cfg["data"]["feature_names"], root), "r") as f:
        feature_names = json.load(f)
        
    # Get activity names (classes 0-11)
    activity_names = get_activity_names()
    
    logger.info(f"Loaded {X_train.shape[0]:,} train samples, {X_test.shape[0]:,} test samples.")
    
    results = []
    
    # Models to train
    ml_models = ["logistic_regression", "decision_tree", "random_forest", "svm", "xgboost", "lightgbm", "catboost"]
    
    for name in ml_models:
        logger.info(f"\n==================== Training {name.upper()} ====================")
        cfg = model_cfg.get(name, {}).copy()
        
        # Instantiate model
        model = get_ml_model(name, cfg)
        
        # Prepare training data (handle SVM subsampling if necessary)
        X_train_fit, y_train_fit = X_train, y_train
        if name == "svm":
            max_samples = model_cfg.get("svm", {}).get("max_train_samples", 100000)
            if len(X_train) > max_samples:
                logger.info(f"Subsampling training set for SVM: {len(X_train)} → {max_samples}")
                np.random.seed(seed)
                sub_indices = np.random.choice(len(X_train), max_samples, replace=False)
                X_train_fit = X_train[sub_indices]
                y_train_fit = y_train[sub_indices]
                
        # Train
        logger.info(f"Fitting {name}...")
        model.fit(X_train_fit, y_train_fit)
        
        # Predict
        preds = model.predict(X_test)
        
        # Compute metrics
        metrics = calculate_metrics(y_test, preds, activity_names)
        metrics["Model"] = name
        results.append(metrics)
        
        # Save model
        ml_dir = resolve_path(paths_cfg["models"]["ml_dir"], root)
        if name == "xgboost":
            model_path = ml_dir / "xgboost.json"
            model.save_model(model_path)
        elif name == "catboost":
            model_path = ml_dir / "catboost.bin"
            model.save_model(str(model_path))
        else:
            model_path = ml_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
        logger.info(f"Saved {name} model checkpoint.")
        
        # Generate figures
        fig_dir = resolve_path(paths_cfg["outputs"]["figures_dir"], root)
        
        # Confusion Matrix
        plot_confusion_matrix(
            y_true=y_test,
            y_pred=preds,
            activity_names=activity_names,
            save_path=fig_dir / f"confusion_matrix_{name}.png",
            title=f"Confusion Matrix - {name.replace('_', ' ').title()}"
        )
        
        # Feature Importance
        plot_feature_importance(
            model=model,
            feature_names=feature_names,
            save_path=fig_dir / f"feature_importance_{name}.png",
            title=f"Feature Importance - {name.replace('_', ' ').title()}"
        )
        
        # ROC Curve (if predict_proba is supported)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            plot_roc_curve(
                y_true=y_test,
                y_score=probs,
                activity_names=activity_names,
                save_path=fig_dir / f"roc_{name}.png",
                title=f"ROC Curve - {name.replace('_', ' ').title()}"
            )
            
    # Save comparison table
    df_results = pd.DataFrame(results)
    results_csv = resolve_path(paths_cfg["outputs"]["ml_results"], root)
    df_results.to_csv(results_csv, index=False)
    logger.info(f"\nAll ML training finished. Saved results comparison to {results_csv}")
    print(df_results[["Model", "accuracy", "f1_macro", "f1_weighted"]])


if __name__ == "__main__":
    main()
