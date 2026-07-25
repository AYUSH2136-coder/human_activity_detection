"""
LightGBM model builder.
"""

from lightgbm import LGBMClassifier


def get_model(config: dict) -> LGBMClassifier:
    """
    Construct and return an LGBMClassifier estimator.
    """
    return LGBMClassifier(**config)
