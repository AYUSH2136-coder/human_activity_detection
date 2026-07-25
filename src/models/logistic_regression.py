"""
Logistic Regression model builder.
"""

import inspect
from sklearn.linear_model import LogisticRegression


def get_model(config: dict) -> LogisticRegression:
    """
    Construct and return a LogisticRegression estimator with the provided config.
    Filters out unsupported parameters dynamically to prevent scikit-learn version issues.
    """
    sig = inspect.signature(LogisticRegression.__init__)
    valid_params = set(sig.parameters.keys())
    cfg = {k: v for k, v in config.items() if k in valid_params}
    return LogisticRegression(**cfg)
