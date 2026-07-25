"""
Support Vector Machine (SVM) model builder.
"""

import inspect
from sklearn.svm import SVC


def get_model(config: dict) -> SVC:
    """
    Construct and return a Support Vector Classifier (SVC) estimator.
    Pops custom configuration keys like 'max_train_samples' and filters out
    unsupported parameters dynamically to prevent version issues.
    """
    cfg = config.copy()
    cfg.pop("max_train_samples", None)
    
    sig = inspect.signature(SVC.__init__)
    valid_params = set(sig.parameters.keys())
    cfg = {k: v for k, v in cfg.items() if k in valid_params}
    return SVC(**cfg)
