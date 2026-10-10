"""Compute Spearman Rank Correlation and Top-k Jaccard Overlap for XAI reliability."""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_FILE = RESULTS_DIR / "xai_summary.csv"
OUTPUT_FILE = RESULTS_DIR / "explanation_reliability_metrics.csv"


def jaccard_overlap(list1: list[str], list2: list[str]) -> float:
    s1, s2 = set(list1), set(list2)
    union = len(s1.union(s2))
    return len(s1.intersection(s2)) / union if union > 0 else 0.0


def main() -> None:
    if not SUMMARY_FILE.exists():
        print(f"Error: {SUMMARY_FILE} not found.")
        return

    df = pd.read_csv(SUMMARY_FILE)
    records = []

    models = df["model"].unique()
    methods = df["method"].unique()

    # 1. SHAP vs. LIME Agreement within each Experiment
    for exp in ["A", "B", "C", "D"]:
        for model in models:
            sub_shap = df[(df["experiment"] == exp) & (df["model"] == model) & (df["method"] == "shap")]
            sub_lime = df[(df["experiment"] == exp) & (df["model"] == model) & (df["method"] == "lime")]

            if sub_shap.empty or sub_lime.empty:
                continue

            merged = pd.merge(sub_shap, sub_lime, on="feature", suffixes=("_shap", "_lime"))
            rho, pval = spearmanr(merged["mean_abs_attribution_shap"], merged["mean_abs_attribution_lime"])

            top3_shap = sub_shap.sort_values(by="mean_abs_attribution", ascending=False)["feature"].head(3).tolist()
            top3_lime = sub_lime.sort_values(by="mean_abs_attribution", ascending=False)["feature"].head(3).tolist()

            top5_shap = sub_shap.sort_values(by="mean_abs_attribution", ascending=False)["feature"].head(5).tolist()
            top5_lime = sub_lime.sort_values(by="mean_abs_attribution", ascending=False)["feature"].head(5).tolist()

            jaccard_3 = jaccard_overlap(top3_shap, top3_lime)
            jaccard_5 = jaccard_overlap(top5_shap, top5_lime)

            records.append({
                "comparison_type": "SHAP_vs_LIME_Agreement",
                "experiment_1": exp,
                "experiment_2": exp,
                "model": model,
                "method": "both",
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(pval), 4),
                "top3_jaccard": round(jaccard_3, 4),
                "top5_jaccard": round(jaccard_5, 4),
            })

    # 2. In-Distribution vs. Cross-Dataset Stability
    # Pair 1: Exp A (CIC -> CIC) vs. Exp C (CIC -> UNSW)
    # Pair 2: Exp B (UNSW -> UNSW) vs. Exp D (UNSW -> CIC)
    domain_pairs = [("A", "C", "CICIDS_source"), ("B", "D", "UNSW_source")]

    for exp1, exp2, desc in domain_pairs:
        for model in models:
            for method in methods:
                sub1 = df[(df["experiment"] == exp1) & (df["model"] == model) & (df["method"] == method)]
                sub2 = df[(df["experiment"] == exp2) & (df["model"] == model) & (df["method"] == method)]

                if sub1.empty or sub2.empty:
                    continue

                merged = pd.merge(sub1, sub2, on="feature", suffixes=("_in", "_cross"))
                rho, pval = spearmanr(merged["mean_abs_attribution_in"], merged["mean_abs_attribution_cross"])

                top3_1 = sub1.sort_values(by="mean_abs_attribution", ascending=False)["feature"].head(3).tolist()
                top3_2 = sub2.sort_values(by="mean_abs_attribution", ascending=False)["feature"].head(3).tolist()

                top5_1 = sub1.sort_values(by="mean_abs_attribution", ascending=False)["feature"].head(5).tolist()
                top5_2 = sub2.sort_values(by="mean_abs_attribution", ascending=False)["feature"].head(5).tolist()

                jaccard_3 = jaccard_overlap(top3_1, top3_2)
                jaccard_5 = jaccard_overlap(top5_1, top5_2)

                records.append({
                    "comparison_type": f"In_Domain_vs_Cross_Domain ({desc})",
                    "experiment_1": exp1,
                    "experiment_2": exp2,
                    "model": model,
                    "method": method,
                    "spearman_rho": round(float(rho), 4),
                    "p_value": round(float(pval), 4),
                    "top3_jaccard": round(jaccard_3, 4),
                    "top5_jaccard": round(jaccard_5, 4),
                })

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Explanation reliability metrics saved to {OUTPUT_FILE}")
    print("\nSummary Results Preview:")
    print(out_df.head(15).to_string())


if __name__ == "__main__":
    main()
