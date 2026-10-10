"""Train the configured IDS model zoo from the command line."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .data_loader import get_train_val
from .model import ModelAdapter, build_adapter

LOGGER = logging.getLogger(__name__)
MODELS = ("mlp_a", "mlp_b", "mlp_c", "rf", "dt")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.05, np.linspace(0.1, 0.9, 17), 0.95])
    scores = [f1_score(y_true, probabilities >= value, zero_division=0) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def _smoke_subset(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.concatenate([np.flatnonzero(y == label)[:5000] for label in (0, 1) if np.any(y == label)])
    rng.shuffle(indices)
    return x[indices], y[indices]


def train_one(dataset: str, model_name: str, seed: int, scaling_mode: str, smoke: bool, force: bool) -> None:
    extension = ".keras" if model_name.startswith("mlp") else ".joblib"
    output = config.MODELS_DIR / dataset / model_name / f"seed{seed}{extension}"
    sidecar = output.with_suffix(".json")
    if output.exists() and sidecar.exists() and not force:
        LOGGER.info("skip %s %s seed=%d", dataset, model_name, seed)
        return
    tf.keras.utils.set_random_seed(seed)
    x_train, x_val, y_train, y_val = get_train_val(dataset, seed, scaling_mode)
    if smoke:
        x_train, y_train = _smoke_subset(x_train, y_train, seed)
        x_val, y_val = _smoke_subset(x_val, y_val, seed + 1)
    adapter = build_adapter(model_name, seed)
    started = time.perf_counter()
    history: dict[str, list[float]] = {}
    if adapter.family == "keras":
        adapter.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc"), tf.keras.metrics.AUC(curve="ROC", name="roc_auc")])
        weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
        fit_history = adapter.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2 if smoke else 50, batch_size=256, verbose=0, class_weight={0: float(weights[0]), 1: float(weights[1])}, callbacks=[])
        history = fit_history.history
    else:
        adapter.model.fit(x_train, y_train)
    val_probabilities = adapter.predict_attack_proba(x_val)
    threshold = _threshold(y_val, val_probabilities)
    adapter.save(output)
    config.RESULTS_DIR.joinpath("training").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(config.RESULTS_DIR / "training" / f"{dataset}_{model_name}_seed{seed}_history.csv", index=False)
    metadata = {"dataset": dataset, "model": model_name, "seed": seed, "scaling_mode": scaling_mode, "features": config.FEATURES, "val_threshold": threshold, "val_roc_auc": float(roc_auc_score(y_val, val_probabilities)) if len(np.unique(y_val)) > 1 else None, "epochs_run": len(next(iter(history.values()))) if history else None, "wall_clock_seconds": time.perf_counter() - started, "git_commit": _git_commit()}
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("finished %s %s seed=%d threshold=%.3f", dataset, model_name, seed, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cicids", "unsw", "all"], default="all")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    parser.add_argument("--scaling-mode", choices=config.ALLOWED_SCALING_MODES, default=config.SCALING_MODE)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    datasets = config.DATASETS if args.dataset == "all" else (args.dataset,)
    models = MODELS if args.model == "all" else (args.model,)
    for dataset in datasets:
        for model_name in models:
            for seed in ([args.seeds[0]] if args.smoke else args.seeds):
                train_one(dataset, model_name, seed, args.scaling_mode, args.smoke, args.force)


if __name__ == "__main__":
    main()