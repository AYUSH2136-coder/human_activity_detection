# Human Activity Detection (HAD) using Wearable Sensors

An end-to-end, production-grade modular machine learning and deep learning pipeline to clean, preprocess, segment, train, and evaluate activity classifiers using the MHEALTH (Mobile Health) wearable sensor dataset.

---

## 📂 Repository Structure

The project has a modular package structure:

```text
├── configs/                  # Configuration files
│   ├── config.yaml           # General configurations
│   ├── paths.yaml            # File system paths (data, models, outputs)
│   ├── model.yaml            # Hyperparameters for ML classifiers
│   └── training.yaml         # Training settings for PyTorch DL models
├── data/                     # Dataset storage (Ignored by Git)
│   ├── raw/                  # Raw MHEALTH csv data
│   ├── interim/              # Preprocessed scaled and cleaned csv
│   └── processed/            # Segmented window arrays (*.npy)
├── checkpoints/              # Saved model weights & scaler (Ignored by Git)
│   ├── ml/                   # Joblib/JSON files for ML models
│   └── dl/                   # PyTorch state-dicts (.pth)
├── docs/                     # Detailed project documentation
│   ├── dataset.md            # MHEALTH dataset specifications
│   ├── methodology.md        # Pipeline engineering details
│   ├── results.md            # Experimental performance metrics
│   └── report.md             # B.Tech Academic Project Report
├── outputs/                  # Results, figures, and logs (Ignored by Git)
│   ├── figures/              # Confusion matrices, ROC curves, training loss
│   ├── logs/                 # Standard runtime log files
│   └── comparison_table.csv  # Combined performance summary
├── src/                      # Source code package
│   ├── data/                 # Processing, windowing, and splitting modules
│   ├── models/               # Classifiers (Traditional and Deep Learning)
│   ├── evaluation/           # Performance reports and visualizations
│   ├── inference/            # Prediction pipeline for streaming data
│   └── utils/                # Helper modules (logger, seeding, paths)
├── scripts/                  # Executable CLI scripts
├── main.py                   # Central CLI entry point orchestrator
├── README.md                 # Project landing page
├── pyproject.toml            # Package metadata & lock details
└── .gitignore                # Git ignore configuration
```

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Setup Environment & Dependencies
Clone the repository and install the dependencies:
```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Place Raw Dataset
Download the MHEALTH dataset and place `mhealth_raw_data.csv` in `data/raw/`.

---

## 🚀 Execution & Command-Line Interfaces (CLI)

The `main.py` script at the root serves as the centralized CLI. You can execute specific pipeline steps or run the entire pipeline from scratch.

### Run the Entire End-to-End Pipeline
Runs preprocessing, feature engineering, train-test splitting, trains all 7 ML models and 4 DL models, generates all plots, and produces comparison reports:
```bash
python main.py
```
*(Or specify the `pipeline` subcommand explicitly)*:
```bash
python main.py pipeline
```

### Individual Subcommands
You can also run specific parts of the pipeline:
```bash
# Preprocess data, extract features, and split
python main.py preprocess

# Train traditional ML models (XGBoost, Random Forest, etc.)
python main.py train-ml

# Train PyTorch deep learning models (MLP, 1D CNN, LSTM, GRU)
python main.py train-dl

# Run aggregated evaluation and comparisons
python main.py evaluate

# Predict activities on raw sensor stream CSV or windowed arrays
python main.py predict --model-name xgboost --model-type ml --input-file data/processed/X_test.npy
```

---

## 📊 Performance Summary

The classifiers are ranked below by their test accuracy on the MHEALTH test set:

| Rank | Model Name | Test Accuracy | F1-Score (Macro) | F1-Score (Weighted) | Type |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost** | **0.9972** | **0.9965** | **0.9972** | Machine Learning |
| 🥈 | **LightGBM** | **0.9953** | **0.9939** | **0.9953** | Machine Learning |
| 🥉 | **Random Forest** | **0.9953** | **0.9956** | **0.9953** | Machine Learning |
| 4 | **CatBoost** | 0.9953 | 0.9956 | 0.9953 | Machine Learning |
| 5 | **Logistic Regression** | 0.9935 | 0.9921 | 0.9935 | Machine Learning |
| 6 | **Support Vector Machine (SVM)** | 0.9897 | 0.9903 | 0.9897 | Machine Learning |
| 7 | **GRU (Gated Recurrent Unit)** | 0.9879 | 0.9885 | 0.9878 | Deep Learning |
| 8 | **Decision Tree** | 0.9879 | 0.9860 | 0.9879 | Machine Learning |
| 9 | **1D CNN (Convolutional)** | 0.9860 | 0.9868 | 0.9860 | Deep Learning |
| 10 | **MLP (Multi-Layer Perceptron)** | 0.9701 | 0.9681 | 0.9699 | Deep Learning |
| 11 | **LSTM (Long Short-Term Memory)** | 0.9617 | 0.9592 | 0.9610 | Deep Learning |

---

## 🛠️ Python Integration (Production Inference)

You can easily run predictions on raw sequences from Python using the `HADInferencePipeline` class:

```python
import numpy as np
from src.inference import HADInferencePipeline
from src.utils.helpers import load_yaml

# 1. Load configuration
train_cfg = load_yaml("configs/training.yaml")

# 2. Instantiate the pipeline with your target model
pipeline = HADInferencePipeline(
    model_name="xgboost",
    model_type="ml",
    model_path="checkpoints/ml/xgboost.json",
    scaler_path="checkpoints/scaler.joblib",
    training_cfg=train_cfg,
    device="auto"  # Detects and runs on CUDA if available
)

# 3. Simulate a raw sensor stream: shape (seq_len, 12_sensor_channels)
raw_sequence = np.random.randn(1000, 12)

# 4. Run prediction (segmenting, scaling, and predicting)
preds, confidence = pipeline.predict_raw_sequence(
    raw_sequence=raw_sequence,
    window_size=128,
    step_size=64
)

print(f"Predicted activities: {preds}")
print(f"Confidence scores: {confidence}")
```
