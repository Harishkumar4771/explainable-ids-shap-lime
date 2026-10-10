"""SHAP integration with a stable attack-attribution shape."""

import numpy as np
import shap


def explain(adapter, background: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, str]:
    explainer = shap.Explainer(adapter.predict_attack_proba, background)
    result = explainer(values)
    output = np.asarray(result.values)
    if output.ndim == 3:
        output = output[:, :, -1]
    return output.reshape(len(values), -1), type(explainer).__name__