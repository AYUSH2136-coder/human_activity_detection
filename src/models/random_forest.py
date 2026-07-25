"""
Random Forest model builder.
"""

import inspect
from sklearn.ensemble import RandomForestClassifier


def get_model(config: dict) -> RandomForestClassifier:
    """
    Construct and return a RandomForestClassifier estimator with the provided config.
    Filters out unsupported parameters dynamically to prevent scikit-learn version issues.
    """
    sig = inspect.signature(RandomForestClassifier.__init__)
    valid_params = set(sig.parameters.keys())
    cfg = {k: v for k, v in config.items() if k in valid_params}
    return RandomForestClassifier(**cfg)
