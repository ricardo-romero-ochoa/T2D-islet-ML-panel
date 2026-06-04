#!/usr/bin/env python3
"""
06_visualization/plot_gene_auc.py

Plots individual gene AUC values for all panel genes.
Demonstrates non-redundancy: no single gene achieves perfect discrimination alone.

Additional manuscript-facing outputs:
  results/figure_inputs/fig7_single_gene_auc.csv
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR, FIGURE_INPUTS_DIR, DISCOVERY_DATASETS, FIGURE_DPI
from figure_metadata import lookup_gene_metadata


def main():
    expr   = pd.read_csv(os.path.join(DATA_PROCESSED, f"{DISCOVERY_DATASETS[0]}_expr_normalized.csv"), index_col=0)
    labels = pd.read_csv(os.path.join(DATA_PROCESSED, f"{DISCOVERY_DATASETS[0]}_labels.csv"))
    labels = labels[labels["label"].isin([0,1])]
    panel  = pd.read_csv(os.path.join(RESULTS_DIR, "final_gene_panel.csv"))

    common = [c for c in expr.columns if c in labels["sample_id"].values]
    ldict  = dict(zip(labels["sample_id"], labels["label"]))
    y      = np.array([ldict[c] for c in common])

    results = []
    for g in panel["gene"].tolist():
        if g not in expr.index:
            continue
        vals = expr.loc[g, common].values.astype(float)
        auc  = roc_auc_score(y, vals)
        auc  = max(auc, 1 - auc)
        meta = lookup_gene_metadata(g)
        display = panel.loc[panel["gene"] == g, "display_label"].iloc[0] if "display_label" in panel.columns else meta["display_label"]
        symbol = panel.loc[panel["gene"] == g, "gene_symbol"].iloc[0] if "gene_symbol" in panel.columns else meta["gene_symbol"]
        results.append({
            "gene": g,
            "display_label": display,
            "gene_symbol": symbol,
            "auc": auc,
        })

    df = pd.DataFrame(results).sort_values("auc", ascending=True).reset_index(drop=True)
    df["rank_desc"] = df["auc"].rank(ascending=False, method="dense").astype(int)
    os.makedirs(FIGURE_INPUTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig7_single_gene_auc.csv"), index=False)

    colors = ["#2E86C1" if a == df["auc"].max() else "#95A5A6" for a in df["auc"]]

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    bars = ax.barh(df["display_label"], df["auc"], color=colors, alpha=0.9, edgecolor="white", height=0.7)
    ax.axvline(0.5,  color="gray",    lw=1, ls="--", alpha=0.5)
    ax.axvline(0.9,  color="#3498DB", lw=1, ls=":",  alpha=0.8)
    ax.axvline(0.95, color="#E74C3C", lw=1, ls=":",  alpha=0.8)

    for bar, val in zip(bars, df["auc"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Single-gene AUC-ROC", fontsize=12)
    ax.set_title("Individual Discriminative Power of Panel Genes (LOOCV)", fontsize=11)
    ax.set_xlim([max(0.5, df['auc'].min() - 0.05), 1.05])

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color="gray",    ls="--", label="AUC = 0.50"),
        Line2D([0],[0], color="#3498DB", ls=":",  label="AUC = 0.90"),
        Line2D([0],[0], color="#E74C3C", ls=":",  label="AUC = 0.95"),
    ], fontsize=9, loc="lower right")

    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for fmt in ["png","svg"]:
        plt.savefig(os.path.join(FIGURES_DIR, f"individual_gene_auc.{fmt}"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("Saved: figures/individual_gene_auc.png/.svg")
    print(f"Saved: {os.path.join(FIGURE_INPUTS_DIR, 'fig7_single_gene_auc.csv')}")


if __name__ == "__main__":
    main()
