"""Evaluate trained models on in-domain and cross-dataset test splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score, f1_score, auc)

from . import config
from .data_loader import get_cross_dataset_inputs, load_dataset
from .model import ModelAdapter

LOGGER = logging.getLogger(__name__)
EXPERIMENTS = {"A": ("cicids", "cicids"), "B": ("unsw", "unsw"), "C": ("cicids", "unsw"), "D": ("unsw", "cicids")}


def _model_path(source: str, model: str, seed: int) -> Path:
    root = config.MODELS_DIR / source / model
    for suffix in (".keras", ".joblib"):
        path = root / f"seed{seed}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No model found for {source}/{model}/seed{seed}")


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    prevalence = float(labels.mean())
    if len(np.unique(labels)) > 1:
        roc = float(roc_auc_score(labels, probabilities))
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        pr_auc = float(auc(recall[::-1], precision[::-1]))
    else:
        roc = float("nan")
        pr_auc = float("nan")
    return {"accuracy": accuracy_score(labels, predictions), "precision": precision_score(labels, predictions, zero_division=0), "recall": recall_score(labels, predictions, zero_division=0), "f1": f1_score(labels, predictions, zero_division=0), "roc_auc": roc, "pr_auc": pr_auc, "prevalence": prevalence, "pr_auc_lift": pr_auc / prevalence if prevalence else float("nan"), "n_test": len(labels), "n_attack": int(labels.sum()), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "balanced_accuracy": balanced_accuracy_score(labels, predictions), "mcc": matthews_corrcoef(labels, predictions)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--models", nargs="+", default=["mlp_a", "mlp_b", "mlp_c", "rf", "dt"])
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    parser.add_argument("--scaling-mode", choices=config.ALLOWED_SCALING_MODES, default=config.SCALING_MODE)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows: list[dict] = []
    seeds = [args.seeds[0]] if args.smoke else args.seeds
    for experiment in args.experiments:
        source, target = EXPERIMENTS[experiment]
        for model_name in args.models:
            for seed in seeds:
                try:
                    path = _model_path(source, model_name, seed)
                except FileNotFoundError:
                    LOGGER.warning("missing model %s/%s seed=%d", source, model_name, seed)
                    continue
                adapter = ModelAdapter.load(path, name=model_name)
                labels = load_dataset(target, "test", scaling_mode=args.scaling_mode)[1]
                features = load_dataset(target, "test", scaling_mode=args.scaling_mode)[0]
                if source != target:
                    features = get_cross_dataset_inputs(source, target, args.scaling_mode)
                probabilities = adapter.predict_attack_proba(features.to_numpy())
                sidecar = path.with_suffix(".json")
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                thresholds = {"fixed_0_5": 0.5, "validation": float(metadata.get("val_threshold", 0.5))}
                for threshold_type, threshold in thresholds.items():
                    row = {"experiment": experiment, "source": source, "target": target, "model": model_name, "seed": seed, "threshold_type": threshold_type, "threshold": threshold, "scaling_mode": args.scaling_mode}
                    row.update(_metrics(labels, probabilities, threshold))
                    rows.append(row)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    if not metrics.empty:
        summary = metrics.groupby(["experiment", "source", "target", "model", "threshold_type"], dropna=False).agg({"roc_auc": ["mean", "std"], "pr_auc": ["mean", "std"], "recall": ["mean", "std"]}).reset_index()
        summary.to_csv(config.RESULTS_DIR / "metrics_summary.csv", index=False)
    LOGGER.info("wrote %d metric rows", len(metrics))


if __name__ == "__main__":
    main()