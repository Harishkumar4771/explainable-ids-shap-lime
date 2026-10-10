"""LIME integration preserving configured feature order."""

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from .. import config


def explain(adapter, training_values: np.ndarray, values: np.ndarray, seed: int = 42, num_samples: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    explainer = LimeTabularExplainer(training_values, feature_names=config.FEATURES, class_names=["benign", "attack"], mode="classification", discretize_continuous=False, random_state=seed)
    attributions = np.zeros((len(values), len(config.FEATURES)), dtype=float)
    scores = np.zeros(len(values), dtype=float)
    intercepts = np.zeros(len(values), dtype=float)
    for row_index, value in enumerate(values):
        explanation = explainer.explain_instance(value, adapter.predict_proba, num_features=len(config.FEATURES), num_samples=num_samples, labels=(1,))
        for feature_index, attribution in explanation.local_exp[1]:
            attributions[row_index, feature_index] = attribution
        scores[row_index] = explanation.score
        intercepts[row_index] = explanation.intercept[1]
    return attributions, scores, intercepts