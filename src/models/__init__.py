"""
Models package initialization — exposing model architectures and getters.
"""

from src.models.logistic_regression import get_model as get_logistic_regression
from src.models.decision_tree import get_model as get_decision_tree
from src.models.random_forest import get_model as get_random_forest
from src.models.svm import get_model as get_svm
from src.models.xgboost_model import get_model as get_xgboost
from src.models.lightgbm_model import get_model as get_lightgbm
from src.models.catboost_model import get_model as get_catboost

from src.models.mlp import MLPClassifier
from src.models.cnn1d import CNN1DClassifier
from src.models.lstm import LSTMClassifier
from src.models.gru import GRUClassifier


def get_ml_model(name: str, config: dict):
    """
    Factory function to retrieve a configured Machine Learning estimator by name.
    """
    model_key = name.lower().replace("_", "").replace("-", "")
    if model_key == "logisticregression":
        return get_logistic_regression(config)
    elif model_key == "decisiontree":
        return get_decision_tree(config)
    elif model_key in ["randomforest", "rf"]:
        return get_random_forest(config)
    elif model_key == "svm":
        return get_svm(config)
    elif model_key == "xgboost":
        return get_xgboost(config)
    elif model_key == "lightgbm":
        return get_lightgbm(config)
    elif model_key == "catboost":
        return get_catboost(config)
    else:
        raise ValueError(f"Unknown ML model name: {name}")


__all__ = [
    "get_logistic_regression",
    "get_decision_tree",
    "get_random_forest",
    "get_svm",
    "get_xgboost",
    "get_lightgbm",
    "get_catboost",
    "MLPClassifier",
    "CNN1DClassifier",
    "LSTMClassifier",
    "GRUClassifier",
    "get_ml_model",
]
