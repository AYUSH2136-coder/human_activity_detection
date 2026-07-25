"""
Decision Tree model builder.
"""

import inspect
from sklearn.tree import DecisionTreeClassifier


def get_model(config: dict) -> DecisionTreeClassifier:
    """
    Construct and return a DecisionTreeClassifier estimator with the provided config.
    Filters out unsupported parameters dynamically to prevent scikit-learn version issues.
    """
    sig = inspect.signature(DecisionTreeClassifier.__init__)
    valid_params = set(sig.parameters.keys())
    cfg = {k: v for k, v in config.items() if k in valid_params}
    return DecisionTreeClassifier(**cfg)
