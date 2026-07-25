"""
CatBoost model builder.
"""

from catboost import CatBoostClassifier


def get_model(config: dict) -> CatBoostClassifier:
    """
    Construct and return a CatBoostClassifier estimator.
    """
    return CatBoostClassifier(**config)
