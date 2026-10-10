"""Decision-tree surrogate explanations for black-box model predictions."""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from .. import config


def fit_surrogate(values, black_box_labels, max_depth: int = 5, seed: int = 42) -> DecisionTreeClassifier:
    """Fit a balanced tree to hard black-box predictions."""
    surrogate = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=seed)
    return surrogate.fit(values, black_box_labels)


def rule_path_attribution(surrogate: DecisionTreeClassifier, values):
    """Return split-feature path attributions for each input row."""
    paths = surrogate.decision_path(values)
    importances = surrogate.tree_.compute_feature_importances(normalize=False)
    output = []
    for row in range(values.shape[0]):
        nodes = paths.indices[paths.indptr[row] : paths.indptr[row + 1]]
        used = set()
        for node in nodes:
            feature = surrogate.tree_.feature[node]
            if feature >= 0:
                used.add(feature)
        output.append([float(importances[index]) if index in used else 0.0 for index in range(values.shape[1])])
    return output
