"""Publication-ready visualizations for Intrusion Detection Cross-Dataset Evaluation,
Explainability Reliability (SHAP vs. LIME), and Neuro-Symbolic Surrogate Comparison.
Corresponds to Phase 1 PPT Slide 6 evaluation components.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "figures"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_detection_performance() -> None:
    """Plot detection AUC and F1 across experiments A, B, C, D comparing ML/DL vs Neuro-Symbolic."""
    metrics_path = RESULTS_DIR / "metrics_summary.csv"
    ns_path = RESULTS_DIR / "neurosymbolic_metrics.csv"
    if not metrics_path.exists() or not ns_path.exists():
        return

    # Base models
    df_base = pd.read_csv(metrics_path, header=[0, 1])
    # Flatten columns
    new_cols = []
    for c in df_base.columns:
        if "Unnamed:" in c[1]:
            new_cols.append(c[0])
        else:
            new_cols.append(f"{c[0]}_{c[1]}")
    df_base.columns = new_cols
    # Filter for validation threshold type
    df_val = df_base[df_base["threshold_type"] == "validation"].copy()
    
    # Select key models
    sub = df_val[["experiment", "model", "roc_auc_mean", "recall_mean"]].copy()
    sub.rename(columns={"roc_auc_mean": "roc_auc", "recall_mean": "recall"}, inplace=True)

    # Neuro-symbolic
    df_ns = pd.read_csv(ns_path)
    ns_sub = df_ns[["experiment", "model", "roc_auc", "recall"]].copy()
    ns_sub["model"] = "neuro_symbolic"

    combined = pd.concat([sub, ns_sub], ignore_index=True)
    
    # Plot ROC AUC across Experiments
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("muted", n_colors=combined["model"].nunique())
    sns.barplot(
        data=combined,
        x="experiment",
        y="roc_auc",
        hue="model",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_title("Cross-Dataset Generalization Drop (ROC-AUC across Exp A, B, C, D)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Experiment Setup (A: CIC->CIC, B: UNSW->UNSW, C: CIC->UNSW, D: UNSW->CIC)", fontsize=11)
    ax.set_ylabel("ROC AUC Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random Guess (0.5)")
    ax.legend(title="Model Architecture", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig1_generalization_roc_auc.png", dpi=300)
    plt.close()


def plot_xai_agreement_heatmap() -> None:
    """Plot Spearman correlation heatmap for SHAP vs LIME agreement and Domain Shift stability."""
    rel_path = RESULTS_DIR / "explanation_reliability_metrics.csv"
    if not rel_path.exists():
        return

    df = pd.read_csv(rel_path)
    
    # 1. SHAP vs LIME Agreement
    df_agree = df[df["comparison_type"] == "SHAP_vs_LIME_Agreement"].copy()
    pivot_agree = df_agree.pivot(index="model", columns="experiment_1", values="spearman_rho")
    pivot_agree.columns = [f"Exp {c}" for c in pivot_agree.columns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        pivot_agree,
        annot=True,
        cmap="coolwarm",
        vmin=-0.5,
        vmax=1.0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Spearman Rank Correlation (ρ)"},
        ax=ax1,
    )
    ax1.set_title("SHAP vs. LIME Agreement\n(Feature Attribution Ranking)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Model Architecture")
    ax1.set_xlabel("Experiment")

    # 2. Domain Shift Stability (In-Domain vs Cross-Domain)
    df_shift = df[df["comparison_type"].str.contains("In_Domain_vs_Cross_Domain")].copy()
    df_shift["shift_label"] = df_shift.apply(
        lambda r: f"{r['method'].upper()} ({'A vs C' if r['experiment_1'] == 'A' else 'B vs D'})", axis=1
    )
    pivot_shift = df_shift.pivot(index="model", columns="shift_label", values="spearman_rho")

    sns.heatmap(
        pivot_shift,
        annot=True,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Spearman Rank Correlation (ρ)"},
        ax=ax2,
    )
    ax2.set_title("Explanation Stability Under Domain Shift\n(In-Domain vs. Cross-Domain)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("")
    ax2.set_xlabel("Explanation Method & Dataset Shift")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig2_xai_reliability_heatmaps.png", dpi=300)
    plt.close()


def plot_neurosymbolic_comparison() -> None:
    """Plot agreement of Neuro-Symbolic Surrogate rules against Post-Hoc SHAP and LIME."""
    comp_path = RESULTS_DIR / "neurosymbolic_vs_posthoc_comparison.csv"
    if not comp_path.exists():
        return

    df = pd.read_csv(comp_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    sns.barplot(
        data=df,
        x="experiment",
        y="spearman_rho",
        hue="comparison",
        palette=["#2b5c8f", "#d95f02"],
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_title("Neuro-Symbolic Surrogate vs. Post-Hoc Attributions (Spearman ρ)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Experiment Setup", fontsize=11)
    ax.set_ylabel("Spearman Rank Correlation (ρ)", fontsize=11)
    ax.set_ylim(-0.6, 1.0)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(title="Comparison Pair", loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig3_neurosymbolic_vs_posthoc.png", dpi=300)
    plt.close()


def plot_feature_importance_comparison() -> None:
    """Plot Top Feature Importance across Models & Methods for in-domain baseline (Exp A)."""
    xai_path = RESULTS_DIR / "xai_metrics.csv"
    ns_attr_path = RESULTS_DIR / "neurosymbolic_attributions.csv"
    if not xai_path.exists() or not ns_attr_path.exists():
        return

    df_xai = pd.read_csv(xai_path)
    df_ns = pd.read_csv(ns_attr_path)

    # Filter for Experiment A, model mlp_a
    exp_a_shap = df_xai[(df_xai["experiment"] == "A") & (df_xai["model"] == "mlp_a") & (df_xai["method"] == "shap")]
    exp_a_lime = df_xai[(df_xai["experiment"] == "A") & (df_xai["model"] == "mlp_a") & (df_xai["method"] == "lime")]
    exp_a_ns = df_ns[df_ns["experiment"] == "A"]

    if exp_a_shap.empty or exp_a_lime.empty or exp_a_ns.empty:
        return

    # Combine top features
    shap_s = exp_a_shap.groupby("feature")["mean_abs_attribution"].mean()
    shap_attr = shap_s / (shap_s.sum() if shap_s.sum() > 0 else 1.0)

    lime_s = exp_a_lime.groupby("feature")["mean_abs_attribution"].mean()
    lime_attr = lime_s / (lime_s.sum() if lime_s.sum() > 0 else 1.0)

    ns_s = exp_a_ns.set_index("feature")["importance"]
    ns_attr = ns_s / (ns_s.sum() if ns_s.sum() > 0 else 1.0)

    df_feats = pd.DataFrame({
        "SHAP (MLP-A)": shap_attr,
        "LIME (MLP-A)": lime_attr,
        "Neuro-Symbolic Rules": ns_attr,
    }).fillna(0)

    # Sort by average attribution
    df_feats["mean"] = df_feats.mean(axis=1)
    df_feats = df_feats.sort_values(by="mean", ascending=True).drop(columns=["mean"])

    fig, ax = plt.subplots(figsize=(10, 6))
    df_feats.plot(kind="barh", ax=ax, width=0.8, colormap="viridis")
    ax.set_title("Experiment A (CICIDS): Feature Attribution Ranking Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("Normalized Feature Importance / Attribution Weight", fontsize=11)
    ax.set_ylabel("Network Flow Features", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig4_feature_attribution_rankings.png", dpi=300)
    plt.close()


def main() -> None:
    print("Generating publication-ready figures in results/figures/...")
    plot_detection_performance()
    plot_xai_agreement_heatmap()
    plot_neurosymbolic_comparison()
    plot_feature_importance_comparison()
    print("All figures successfully generated!")


if __name__ == "__main__":
    main()
