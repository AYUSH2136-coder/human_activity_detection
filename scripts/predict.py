"""
CLI script to run inference using a trained model.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.inference import HADInferencePipeline
from src.utils.helpers import load_yaml, get_project_root, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Inference using Trained HAD Models")
    parser.add_argument("--model-name", type=str, required=True, help="Name of model to use (e.g. random_forest, cnn1d)")
    parser.add_argument("--model-type", type=str, choices=["ml", "dl"], required=True, help="Model type: ml or dl")
    parser.add_argument("--input-file", type=str, required=True, help="Path to input file (.npy window array or .csv raw sensor data)")
    parser.add_argument("--output-file", type=str, default="outputs/predictions.csv", help="Path to save prediction results")
    parser.add_argument("--paths-config", type=str, default="configs/paths.yaml", help="Path to paths config file")
    parser.add_argument("--train-config", type=str, default="configs/training.yaml", help="Path to training config file")
    args = parser.parse_args()

    root = get_project_root()
    paths_cfg = load_yaml(resolve_path(args.paths_config, root))
    train_cfg = load_yaml(resolve_path(args.train_config, root))
    
    # Resolve paths
    scaler_path = resolve_path(paths_cfg["artifacts"]["scaler"], root)
    output_path = resolve_path(args.output_file, root)
    input_path = resolve_path(args.input_file, root)
    
    # Find model file path
    model_name_clean = args.model_name.lower().replace("_", "").replace("-", "")
    if args.model_type == "ml":
        ml_dir = resolve_path(paths_cfg["models"]["ml_dir"], root)
        if model_name_clean == "xgboost":
            model_path = ml_dir / "xgboost.json"
        elif model_name_clean == "catboost":
            model_path = ml_dir / "catboost.bin"
        else:
            model_path = ml_dir / f"{args.model_name}.joblib"
    else:
        dl_dir = resolve_path(paths_cfg["models"]["dl_dir"], root)
        model_path = dl_dir / f"{args.model_name}_model.pth"
        
    if not model_path.exists():
        logger.error(f"Model checkpoint not found at {model_path}")
        return
        
    # Initialize pipeline
    pipeline = HADInferencePipeline(
        model_name=args.model_name,
        model_type=args.model_type,
        model_path=model_path,
        scaler_path=scaler_path,
        training_cfg=train_cfg,
    )
    
    # Load input and predict
    logger.info(f"Loading input file from {input_path}...")
    if input_path.suffix == ".npy":
        X = np.load(input_path)
        logger.info(f"Loaded numpy array of shape {X.shape}. Running window-based prediction...")
        # If input has shape (window_size, num_features), add batch dimension
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)
        preds, probs = pipeline.predict(X)
        
        # Save results
        df_out = pd.DataFrame({
            "window_index": np.arange(len(preds)),
            "predicted_class": preds,
            "confidence": probs
        })
    elif input_path.suffix == ".csv":
        df_in = pd.read_csv(input_path)
        # Verify columns
        from src.data.loader import get_sensor_columns
        sensor_cols = get_sensor_columns()
        missing = [c for c in sensor_cols if c not in df_in.columns]
        if missing:
            raise ValueError(f"Input CSV is missing required sensor columns: {missing}")
            
        raw_sequence = df_in[sensor_cols].values
        
        window_size = train_cfg.get("lstm", {}).get("window_size", 128)
        step_size = window_size // 2
        
        preds, probs = pipeline.predict_raw_sequence(
            raw_sequence=raw_sequence,
            window_size=window_size,
            step_size=step_size
        )
        
        df_out = pd.DataFrame({
            "window_index": np.arange(len(preds)),
            "start_sample": np.arange(0, len(raw_sequence) - window_size + 1, step_size),
            "end_sample": np.arange(window_size, len(raw_sequence) + 1, step_size),
            "predicted_class": preds,
            "confidence": probs
        })
    else:
        logger.error("Unsupported input file format. Use .npy or .csv")
        return
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    logger.info(f"Predictions written to {output_path}")
    print(df_out.head())


if __name__ == "__main__":
    main()
