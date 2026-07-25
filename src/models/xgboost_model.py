"""
XGBoost model builder.
"""

from xgboost import XGBClassifier


def get_model(config: dict) -> XGBClassifier:
    """
    Construct and return an XGBClassifier estimator.
    """
    return XGBClassifier(**config)
