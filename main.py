"""
Main entry point for the Human Activity Detection (HAD) project.
Supports running the entire pipeline or specific modules via CLI commands.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to avoid module import issues when running from root
sys.path.append(str(Path(__file__).resolve().parent))

# Import scripts directly by calling their main functions or equivalent wrappers
from scripts import preprocess, train_ml, train_dl, evaluate, predict
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline():
    """Run the entire end-to-end pipeline sequentially."""
    logger.info("\n\n" + "=" * 50)
    logger.info("   RUNNING END-TO-END HUMAN ACTIVITY DETECTION PIPELINE   ")
    logger.info("=" * 50)
    
    # 1. Preprocessing
    logger.info("\n>>> STAGE 1: Running Preprocessing & Feature Engineering")
    try:
        sys.argv = ["main.py"]
        preprocess.main()
    except Exception as e:
        logger.error(f"Preprocessing stage failed: {e}")
        raise e
        
    # 2. ML Training
    logger.info("\n>>> STAGE 2: Training Traditional ML Models")
    try:
        sys.argv = ["main.py"]
        train_ml.main()
    except Exception as e:
        logger.error(f"ML training stage failed: {e}")
        raise e
        
    # 3. DL Training
    logger.info("\n>>> STAGE 3: Training PyTorch Deep Learning Models")
    try:
        sys.argv = ["main.py"]
        train_dl.main()
    except Exception as e:
        logger.error(f"DL training stage failed: {e}")
        raise e
        
    # 4. Evaluation
    logger.info("\n>>> STAGE 4: Generating Aggregated Model Comparison Reports")
    try:
        sys.argv = ["main.py"]
        evaluate.main()
    except Exception as e:
        logger.error(f"Evaluation stage failed: {e}")
        raise e
        
    logger.info("\n" + "=" * 50)
    logger.info("   PIPELINE COMPLETED SUCCESSFULLY!   ")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Human Activity Detection (HAD) Modular Pipeline Command Line Tool",
        usage="%(prog)s [command] [options]"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands to execute specific steps")
    
    # 1. Pipeline parser
    subparsers.add_parser("pipeline", help="Run the entire pipeline (preprocess, train, evaluate)")
    
    # 2. Preprocess parser
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess, engineer features, and split the raw dataset")
    preprocess_parser.add_argument("--paths-config", type=str, default="configs/paths.yaml")
    preprocess_parser.add_argument("--train-config", type=str, default="configs/training.yaml")
    
    # 3. Train ML parser
    train_ml_parser = subparsers.add_parser("train-ml", help="Train and evaluate traditional machine learning models")
    train_ml_parser.add_argument("--paths-config", type=str, default="configs/paths.yaml")
    train_ml_parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    train_ml_parser.add_argument("--train-config", type=str, default="configs/training.yaml")
    
    # 4. Train DL parser
    train_dl_parser = subparsers.add_parser("train-dl", help="Train and evaluate PyTorch deep learning models")
    train_dl_parser.add_argument("--paths-config", type=str, default="configs/paths.yaml")
    train_dl_parser.add_argument("--train-config", type=str, default="configs/training.yaml")
    
    # 5. Evaluate parser
    evaluate_parser = subparsers.add_parser("evaluate", help="Aggregate all training results and compare performance")
    evaluate_parser.add_argument("--paths-config", type=str, default="configs/paths.yaml")
    
    # 6. Predict parser
    predict_parser = subparsers.add_parser("predict", help="Run inference on raw data or windows using a trained model")
    predict_parser.add_argument("--model-name", type=str, required=True, help="Model name (e.g. random_forest, cnn1d)")
    predict_parser.add_argument("--model-type", type=str, choices=["ml", "dl"], required=True, help="Model type: ml or dl")
    predict_parser.add_argument("--input-file", type=str, required=True, help="Input data file (.npy or .csv)")
    predict_parser.add_argument("--output-file", type=str, default="outputs/predictions.csv", help="Output prediction CSV path")
    predict_parser.add_argument("--paths-config", type=str, default="configs/paths.yaml")
    predict_parser.add_argument("--train-config", type=str, default="configs/training.yaml")
    
    # If no argument is passed, run pipeline by default
    if len(sys.argv) == 1:
        run_pipeline()
        return

    # Parse args
    args, unknown = parser.parse_known_args()
    
    if args.command == "pipeline":
        run_pipeline()
    elif args.command == "preprocess":
        sys.argv = ["main.py"] + sys.argv[2:]
        preprocess.main()
    elif args.command == "train-ml":
        sys.argv = ["main.py"] + sys.argv[2:]
        train_ml.main()
    elif args.command == "train-dl":
        sys.argv = ["main.py"] + sys.argv[2:]
        train_dl.main()
    elif args.command == "evaluate":
        sys.argv = ["main.py"] + sys.argv[2:]
        evaluate.main()
    elif args.command == "predict":
        sys.argv = ["main.py"] + sys.argv[2:]
        predict.main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
