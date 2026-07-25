"""
Inference module — loading models and running predictions on single or batch inputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.data.preprocessing import load_scaler
from src.data.feature_engineering import extract_statistical_features
from src.models import MLPClassifier, CNN1DClassifier, LSTMClassifier, GRUClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HADInferencePipeline:
    """
    End-to-end inference pipeline for Human Activity Detection.
    Loads the trained model and scaler to make predictions on raw sensor sequences.
    """
    def __init__(
        self,
        model_name: str,
        model_type: str,  # "ml" or "dl"
        model_path: str | Path,
        scaler_path: str | Path,
        training_cfg: dict | None = None,
        device: str = "auto",
    ):
        self.model_name = model_name.lower().replace("_", "").replace("-", "")
        self.model_type = model_type.lower()
        self.model_path = Path(model_path)
        
        # Load scaler
        self.scaler = load_scaler(scaler_path)
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.training_cfg = training_cfg or {}
        
        # Load model
        self.model = self._load_model()
        logger.info(f"Inference pipeline initialized successfully for {model_name} ({model_type}) on {self.device}.")

    def _load_model(self):
        """Load the model based on type and name."""
        if self.model_type == "ml":
            logger.info(f"Loading ML model from {self.model_path}...")
            if self.model_name == "xgboost":
                # XGBClassifier
                model = xgb.XGBClassifier()
                model.load_model(self.model_path)
                return model
            elif self.model_name == "lightgbm":
                return joblib.load(self.model_path)
            elif self.model_name == "catboost":
                model = CatBoostClassifier()
                model.load_model(str(self.model_path))
                return model
            else:
                # joblib load for Random Forest, SVM, Decision Tree, Logistic Regression
                return joblib.load(self.model_path)
                
        elif self.model_type == "dl":
            logger.info(f"Loading DL model from {self.model_path}...")
            num_classes = self.training_cfg.get("num_classes", 12)
            num_features = self.training_cfg.get("num_features", 12)
            
            # Instantiate architecture
            if self.model_name == "mlp":
                cfg = self.training_cfg.get("mlp", {})
                model = MLPClassifier(
                    input_dim=num_features * self.training_cfg.get("window_size", 128),
                    num_classes=num_classes,
                    hidden_layers=cfg.get("hidden_layers", [256, 128, 64]),
                    dropout=cfg.get("dropout", 0.3),
                    activation=cfg.get("activation", "relu"),
                )
            elif self.model_name == "cnn1d":
                cfg = self.training_cfg.get("cnn1d", {})
                model = CNN1DClassifier(
                    num_features=num_features,
                    num_classes=num_classes,
                    num_filters=cfg.get("num_filters", [64, 128, 256]),
                    kernel_size=cfg.get("kernel_size", 3),
                    dropout=cfg.get("dropout", 0.3),
                )
            elif self.model_name == "lstm":
                cfg = self.training_cfg.get("lstm", {})
                model = LSTMClassifier(
                    num_features=num_features,
                    num_classes=num_classes,
                    hidden_size=cfg.get("hidden_size", 128),
                    num_layers=cfg.get("num_layers", 2),
                    dropout=cfg.get("dropout", 0.3),
                    bidirectional=cfg.get("bidirectional", False),
                )
            elif self.model_name == "gru":
                cfg = self.training_cfg.get("gru", {})
                model = GRUClassifier(
                    num_features=num_features,
                    num_classes=num_classes,
                    hidden_size=cfg.get("hidden_size", 128),
                    num_layers=cfg.get("num_layers", 2),
                    dropout=cfg.get("dropout", 0.3),
                    bidirectional=cfg.get("bidirectional", False),
                )
            else:
                raise ValueError(f"Unknown DL model name: {self.model_name}")
                
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict activity classes and confidence scores for a batch of sliding windows.

        Parameters
        ----------
        X : np.ndarray
            3D array of shape (num_windows, window_size, num_features) containing
            preprocessed (scaled) sensor windows.

        Returns
        -------
        preds : np.ndarray
            1D array of predicted class labels (0-indexed).
        probs : np.ndarray
            1D array of class probabilities (confidence scores) for the predicted labels.
        """
        if self.model_type == "ml":
            # Extract statistical features
            X_eng, _ = extract_statistical_features(X)
            
            # Predict
            probs_matrix = self.model.predict_proba(X_eng)
            preds = np.argmax(probs_matrix, axis=1)
            probs = np.max(probs_matrix, axis=1)
            return preds, probs
            
        elif self.model_type == "dl":
            # Convert to torch tensor
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                logits = self.model(X_tensor)
                probs_matrix = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs_matrix, axis=1)
                probs = np.max(probs_matrix, axis=1)
            return preds, probs

    def predict_raw_sequence(
        self,
        raw_sequence: np.ndarray,
        window_size: int = 128,
        step_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Segment a continuous unscaled sequence of sensor measurements, apply scaler,
        and predict activity for each window.

        Parameters
        ----------
        raw_sequence : np.ndarray
            2D array of shape (seq_len, num_features). Unscaled raw sensor sequence.
        window_size : int
            Number of samples per window.
        step_size : int
            Shift size between windows.

        Returns
        -------
        preds : np.ndarray
            Predicted activity labels (0-indexed) for each segmented window.
        probs : np.ndarray
            Confidence scores for the predictions.
        """
        logger.info(f"Running inference on raw sequence of length {len(raw_sequence)}...")
        
        # Scale
        scaled_sequence = self.scaler.transform(raw_sequence)
        
        # Segment into sliding windows
        X_list = []
        for start in range(0, len(scaled_sequence) - window_size + 1, step_size):
            end = start + window_size
            X_list.append(scaled_sequence[start:end])
            
        if not X_list:
            raise ValueError(f"Sequence length ({len(raw_sequence)}) is shorter than window_size ({window_size}).")
            
        X = np.array(X_list, dtype=np.float32)
        return self.predict(X)
