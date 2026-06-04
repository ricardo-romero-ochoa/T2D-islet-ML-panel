#!/usr/bin/env python3
"""
03_feature_selection/deg_analysis.py

Differential expression analysis using empirical Bayes moderated t-test
(Python equivalent of limma eBayes). Applies Benjamini-Hochberg FDR correction.

Outputs:
  results/deg_results.csv   — full statistics for all genes
  results/deg_list.csv      — significant DEGs (FDR < cutoff, |log2FC| >= cutoff)
  figures/volcano_<gse>.png/.svg
"""

import os, sys, logging, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR, LOGS_DIR,
                    DISCOVERY_DATASETS, DEG_FDR_CUTOFF, DEG_LOGFC_CUTOFF,
                    PALETTE_T2D, PALETTE_CTRL, FIGURE_DPI)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "03_deg.log")), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

def bh_correction(pvals):
    n = len(pvals); order = np.argsort(pvals); rank = np.empty_like(order)
    rank[order] = np.arange(1, n + 1)
    adj = np.minimum(1.0, pvals * n / rank)
    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return adj

def ebayes_moderated_t(X_t2d, X_ctrl):
    n1, n2 = X_t2d.shape[0], X_ctrl.shape[0]
    mean1, mean2 = X_t2d.mean(axis=0), X_ctrl.mean(axis=0)
    log2fc = mean1 - mean2
    var1   = X_t2d.var(axis=0, ddof=1)
    var2   = X_ctrl.var(axis=0, ddof=1)
    df_pool = n1 + n2 - 2
    sp2     = ((n1-1)*var1 + (n2-1)*var2) / df_pool
    log_sp2 = np.log(sp2 + 1e-8)
    prior_var = np.exp(np.median(log_sp2))
    prior_df  = max(3, len(sp2) / 10)
    sp2_shrunk = (prior_df * prior_var + df_pool * sp2) / (df_pool + prior_df)
    se     = np.sqrt(sp2_shrunk * (1/n1 + 1/n2))
    t_stat = log2fc / (se + 1e-10)
    pvals  = 2 * stats.t.sf(np.abs(t_stat), df=df_pool + prior_df)
    return log2fc, t_stat, pvals

def run_deg(expr, labels, gse_id):
    common = [c for c in expr.columns if c in labels.index and labels[c] in [0, 1]]
    expr_sub = expr[common]
    y = labels[common].values.astype(int)
    X = expr_sub.values.T
    gene_names = expr_sub.index.tolist()
    X_t2d  = X[y == 1]; X_ctrl = X[y == 0]
    log.info(f"  DEG: {X_t2d.shape[0]} T2D vs {X_ctrl.shape[0]} Ctrl | {X.shape[1]} genes")
    if X_t2d.shape[0] < 2 or X_ctrl.shape[0] < 2:
        log.error("  Not enough samples"); return pd.DataFrame()
    log2fc, t_stat, pvals = ebayes_moderated_t(X_t2d, X_ctrl)
    adj = bh_correction(pvals)
    df  = pd.DataFrame({"gene": gene_names, "log2FC": log2fc, "t_stat": t_stat,
                        "pvalue": pvals, "adj_pvalue": adj,
                        "mean_T2D": X_t2d.mean(axis=0), "mean_Ctrl": X_ctrl.mean(axis=0),
                        "gse_id": gse_id}).sort_values("adj_pvalue")
    sig = df[(df["adj_pvalue"] <= DEG_FDR_CUTOFF) & (df["log2FC"].abs() >= DEG_LOGFC_CUTOFF)]
    log.info(f"  DEGs: {len(sig)} (up={( sig['log2FC']>0).sum()}, down={(sig['log2FC']<0).sum()})")
    return df

def volcano_plot(results, gse_id, out_dir, top_n=20):
    sig_up   = (results["adj_pvalue"] <= DEG_FDR_CUTOFF) & (results["log2FC"] >=  DEG_LOGFC_CUTOFF)
    sig_down = (results["adj_pvalue"] <= DEG_FDR_CUTOFF) & (results["log2FC"] <= -DEG_LOGFC_CUTOFF)
    ns       = ~(sig_up | sig_down)
    neg_log_p = -np.log10(results["pvalue"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(results["log2FC"][ns],       neg_log_p[ns],       c="#CCCCCC", s=10, alpha=0.4, lw=0)
    ax.scatter(results["log2FC"][sig_up],   neg_log_p[sig_up],   c=PALETTE_T2D, s=18, alpha=0.7, lw=0, label=f"Up in T2D ({sig_up.sum()})")
    ax.scatter(results["log2FC"][sig_down], neg_log_p[sig_down], c=PALETTE_CTRL, s=18, alpha=0.7, lw=0, label=f"Down in T2D ({sig_down.sum()})")
    ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.axvline( DEG_LOGFC_CUTOFF, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(-DEG_LOGFC_CUTOFF, color="gray", ls="--", lw=0.8, alpha=0.7)
    top = pd.concat([results[sig_up].nsmallest(top_n//2,"adj_pvalue"),
                     results[sig_down].nsmallest(top_n//2,"adj_pvalue")])
    for _, row in top.iterrows():
        ax.annotate(row["gene"], xy=(row["log2FC"], -np.log10(row["pvalue"])),
                    fontsize=6.5, ha="center", xytext=(0,5), textcoords="offset points",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.set_xlabel("log₂ Fold Change (T2D / Control)", fontsize=12)
    ax.set_ylabel("−log₁₀(p-value)", fontsize=12)
    ax.set_title(f"Differential Expression — {gse_id}\nFDR ≤ {DEG_FDR_CUTOFF}, |log₂FC| ≥ {DEG_LOGFC_CUTOFF}", fontsize=11)
    ax.legend(fontsize=9); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for fmt in ["png","svg"]:
        plt.savefig(os.path.join(out_dir, f"volcano_{gse_id}.{fmt}"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DISCOVERY_DATASETS[0])
    args = ap.parse_args()

    expr_path   = os.path.join(DATA_PROCESSED, f"{args.dataset}_expr_normalized.csv")
    labels_path = os.path.join(DATA_PROCESSED, f"{args.dataset}_labels.csv")
    if not os.path.exists(expr_path):
        log.error(f"Not found: {expr_path}"); sys.exit(1)

    expr   = pd.read_csv(expr_path, index_col=0)
    ldf    = pd.read_csv(labels_path)
    labels = pd.Series(ldf["label"].values, index=ldf["sample_id"].values).dropna()

    results = run_deg(expr, labels, args.dataset)
    if results.empty: sys.exit(1)

    results.to_csv(os.path.join(RESULTS_DIR, "deg_results.csv"), index=False)
    sig = results[(results["adj_pvalue"] <= DEG_FDR_CUTOFF) & (results["log2FC"].abs() >= DEG_LOGFC_CUTOFF)]
    sig.to_csv(os.path.join(RESULTS_DIR, "deg_list.csv"), index=False)
    log.info(f"Saved: deg_results.csv ({len(results)} genes), deg_list.csv ({len(sig)} DEGs)")
    volcano_plot(results, args.dataset, FIGURES_DIR)

if __name__ == "__main__":
    main()
