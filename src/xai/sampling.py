"""Deterministic, class-balanced explanation instance sampling."""

import json
from .. import config
from ..data_loader import load_dataset
import numpy as np


def sample_instances(experiment: str, target: str, seed: int = 42, force: bool = False) -> dict:
    path = config.RESULTS_DIR / "xai" / f"instances_{experiment}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    _, labels = load_dataset(target, "test")
    rng = np.random.default_rng(seed)
    indices = []
    weights = []
    for label, count in ((0, 500), (1, 500)):
        candidates = np.flatnonzero(labels == label)
        chosen = rng.choice(candidates, size=min(count, len(candidates)), replace=False)
        indices.extend(chosen.tolist())
        weights.extend([1.0 / max(1, len(candidates))] * len(chosen))
    result = {"experiment": experiment, "indices": indices, "labels": labels[indices].tolist(), "weights": weights}
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result