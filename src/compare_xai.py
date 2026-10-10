"""Summarize cached explanation importance without rerunning explainers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--models", nargs="+", default=["mlp_a", "mlp_b", "mlp_c", "rf", "dt"])
    parser.add_argument("--methods", nargs="+", default=["shap", "lime"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--smoke", action="store_true")
    rows = []
    for experiment in parser.parse_args().experiments:
        for model in parser.parse_args().models:
            for method in parser.parse_args().methods:
                for seed in parser.parse_args().seeds:
                    path = config.XAI_CACHE_DIR / experiment / model / f"seed{seed}" / f"{method}.npz"
                    if not path.exists():
                        continue
                    values = np.load(path)["attributions"]
                    for index, feature in enumerate(config.FEATURES):
                        rows.append({"experiment": experiment, "model": model, "method": method, "seed": seed, "feature": feature, "mean_abs_attribution": float(np.abs(values[:, index]).mean())})
    output = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    output.to_csv(config.RESULTS_DIR / "xai_metrics.csv", index=False)
    output.groupby(["experiment", "model", "method", "feature"], as_index=False)["mean_abs_attribution"].mean().to_csv(config.RESULTS_DIR / "xai_summary.csv", index=False)


if __name__ == "__main__":
    main()