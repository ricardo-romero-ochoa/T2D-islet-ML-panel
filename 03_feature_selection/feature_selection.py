#!/usr/bin/env python3
"""
03_feature_selection/feature_selection.py

Three-method consensus feature selection:
  1. LASSO (L1 logistic regression)
  2. SVM-RFE (recursive feature elimination with linear SVM)
  3. Random Forest importance (mean decrease in impurity, 5-fold averaged)

Consensus panel: genes selected by >= CONSENSUS_MIN_VOTES methods.
Final panel: top FINAL_PANEL_SIZE genes by composite score.

In addition to the original outputs, this script now saves manuscript-facing CSVs
needed to build Figures 7–9 reproducibly.
"""

import json
import os, sys, logging, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR, FIGURE_INPUTS_DIR, LOGS_DIR,
                    DISCOVERY_DATASETS, LASSO_CV_FOLDS, SVM_RFE_STEP,
                    SVM_RFE_CV, RF_N_ESTIMATORS, TOP_N_FEATURES,
                    CONSENSUS_MIN_VOTES, FINAL_PANEL_SIZE, RANDOM_STATE,
                    FIGURE_DPI, PALETTE_T2D, PALETTE_CTRL)
from figure_metadata import lookup_gene_metadata

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "03_features.log")), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


def load_data(gse_id, deg_genes=None):
    expr   = pd.read_csv(os.path.join(DATA_PROCESSED, f"{gse_id}_expr_normalized.csv"), index_col=0)
    ldf    = pd.read_csv(os.path.join(DATA_PROCESSED, f"{gse_id}_labels.csv"))
    labels = pd.Series(ldf["label"].values, index=ldf["sample_id"].values).dropna()
    common = [c for c in expr.columns if c in labels.index and labels[c] in [0,1]]
    expr   = expr[common]; y = labels[common].astype(int)
    if deg_genes:
        overlap = [g for g in deg_genes if g in expr.index]
        expr    = expr.loc[overlap]
        log.info(f"  Pre-filtered to {len(overlap)} DEG genes")
    X = expr.T.fillna(0)
    log.info(f"  X: {X.shape}  T2D={( y==1).sum()}  Ctrl={(y==0).sum()}")
    return X, y


def run_lasso(X, y):
    log.info("LASSO...")
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    cv     = StratifiedKFold(n_splits=LASSO_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    clf    = LogisticRegressionCV(Cs=np.logspace(-4,2,30), cv=cv, penalty="l1",
                                   solver="liblinear", max_iter=2000,
                                   random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
    clf.fit(X_sc, y)
    coef = clf.coef_[0]
    df   = pd.DataFrame({"gene": X.columns, "lasso_coef": coef, "abs_coef": np.abs(coef)}).sort_values("abs_coef", ascending=False)
    df["lasso_selected"] = df["abs_coef"] > 0
    log.info(f"  Selected: {df['lasso_selected'].sum()}")
    return df


def run_svm_rfe(X, y):
    log.info("SVM-RFE...")
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    cv     = StratifiedKFold(n_splits=SVM_RFE_CV, shuffle=True, random_state=RANDOM_STATE)
    rfe    = RFECV(LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE),
                   step=max(1, int(X.shape[1]*SVM_RFE_STEP)), cv=cv,
                   scoring="roc_auc", min_features_to_select=5, n_jobs=-1)
    rfe.fit(X_sc, y)
    df = pd.DataFrame({"gene": X.columns, "svm_support": rfe.support_, "svm_ranking": rfe.ranking_}).sort_values("svm_ranking")
    df["svm_selected"] = df["svm_support"]
    log.info(f"  Optimal features: {rfe.n_features_}")
    return df


def run_rf(X, y):
    log.info("Random Forest...")
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    imps   = []
    for fold, (tr, _) in enumerate(cv.split(X_sc, y)):
        rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_features="sqrt",
                                     n_jobs=-1, random_state=RANDOM_STATE+fold, class_weight="balanced")
        rf.fit(X_sc[tr], y.iloc[tr])
        imps.append(rf.feature_importances_)
    mean_imp = np.mean(imps, axis=0); std_imp = np.std(imps, axis=0)
    df = pd.DataFrame({"gene": X.columns, "rf_importance": mean_imp, "rf_std": std_imp}).sort_values("rf_importance", ascending=False)
    df["rf_rank"]     = range(1, len(df)+1)
    df["rf_selected"] = df["rf_rank"] <= TOP_N_FEATURES
    log.info(f"  Top {TOP_N_FEATURES} genes retained")
    return df


def build_consensus(lasso_df, svm_df, rf_df):
    m = (lasso_df[["gene","lasso_selected","abs_coef"]].rename(columns={"abs_coef":"lasso_score"})
         .merge(svm_df[["gene","svm_selected","svm_ranking"]].rename(columns={"svm_ranking":"svm_score"}), on="gene")
         .merge(rf_df[["gene","rf_selected","rf_importance"]].rename(columns={"rf_importance":"rf_score"}), on="gene"))
    m["votes"]     = m["lasso_selected"].astype(int) + m["svm_selected"].astype(int) + m["rf_selected"].astype(int)
    m["consensus"] = m["votes"] >= CONSENSUS_MIN_VOTES
    for col in ["lasso_score","rf_score"]:
        mx = m[col].abs().max(); m[col+"_norm"] = m[col].abs() / (mx + 1e-10)
    svm_mx = m["svm_score"].max()
    m["svm_score_norm"] = 1 - (m["svm_score"] - 1) / (svm_mx + 1e-10)
    m["composite_score"] = (m["lasso_score_norm"] + m["svm_score_norm"] + m["rf_score_norm"]) / 3
    consensus   = m[m["consensus"]].sort_values("composite_score", ascending=False)
    final_panel = consensus.head(FINAL_PANEL_SIZE).copy()
    for col in ["gene"]:
        pass
    final_panel["display_label"] = final_panel["gene"].map(lambda g: lookup_gene_metadata(g)["display_label"])
    final_panel["gene_symbol"] = final_panel["gene"].map(lambda g: lookup_gene_metadata(g)["gene_symbol"])
    final_panel["biotype"] = final_panel["gene"].map(lambda g: lookup_gene_metadata(g)["biotype"])
    final_panel["direction"] = final_panel["gene"].map(lambda g: lookup_gene_metadata(g)["direction"])
    log.info(f"  Consensus genes: {len(consensus)}  Final panel: {len(final_panel)}")
    log.info(f"  Panel: {final_panel['display_label'].tolist()}")
    return m, final_panel


def save_manuscript_inputs(lasso_df, svm_df, rf_df, merged_df, final_panel):
    os.makedirs(FIGURE_INPUTS_DIR, exist_ok=True)

    lasso_sel = lasso_df[lasso_df["lasso_selected"]].copy()
    lasso_sel.to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig7_lasso_selected_genes.csv"), index=False)

    svm_sel = svm_df[svm_df["svm_selected"]].copy()
    svm_sel.to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig7_svmrfe_selected_genes.csv"), index=False)

    rf_sel = rf_df[rf_df["rf_selected"]].copy()
    rf_sel.to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig7_rf_selected_genes.csv"), index=False)

    membership = merged_df[["gene","lasso_selected","svm_selected","rf_selected","votes","consensus","composite_score"]].copy()
    membership["display_label"] = membership["gene"].map(lambda g: lookup_gene_metadata(g)["display_label"])
    membership["gene_symbol"] = membership["gene"].map(lambda g: lookup_gene_metadata(g)["gene_symbol"])
    membership.to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig7_feature_selection_membership.csv"), index=False)

    venn_counts = {
        "lasso_only": int(len(set(lasso_sel["gene"]) - set(svm_sel["gene"]) - set(rf_sel["gene"]))),
        "svm_only": int(len(set(svm_sel["gene"]) - set(lasso_sel["gene"]) - set(rf_sel["gene"]))),
        "rf_only": int(len(set(rf_sel["gene"]) - set(lasso_sel["gene"]) - set(svm_sel["gene"]))),
        "lasso_svm": int(len((set(lasso_sel["gene"]) & set(svm_sel["gene"])) - set(rf_sel["gene"]))),
        "lasso_rf": int(len((set(lasso_sel["gene"]) & set(rf_sel["gene"])) - set(svm_sel["gene"]))),
        "svm_rf": int(len((set(svm_sel["gene"]) & set(rf_sel["gene"])) - set(lasso_sel["gene"]))),
        "all_three": int(len(set(lasso_sel["gene"]) & set(svm_sel["gene"]) & set(rf_sel["gene"]))),
    }
    with open(os.path.join(FIGURE_INPUTS_DIR, "fig7_venn_counts.json"), "w", encoding="utf-8") as fh:
        json.dump(venn_counts, fh, indent=2)

    final_panel.to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig7_final_panel.csv"), index=False)


def plot_lasso(df, out_dir):
    top = df[df["lasso_selected"]].head(30).sort_values("lasso_coef")
    if top.empty: return
    fig, ax = plt.subplots(figsize=(7, max(5, len(top)*0.3)))
    colors = [PALETTE_T2D if c > 0 else PALETTE_CTRL for c in top["lasso_coef"]]
    ax.hlines(top["gene"], 0, top["lasso_coef"], colors=colors, lw=1.5, alpha=0.7)
    ax.scatter(top["lasso_coef"], top["gene"], c=colors, s=60, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("LASSO Coefficient"); ax.set_title("LASSO Feature Importance")
    ax.legend(handles=[Patch(color=PALETTE_T2D, label="Up in T2D"), Patch(color=PALETTE_CTRL, label="Down in T2D")])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for fmt in ["png","svg"]: plt.savefig(os.path.join(out_dir, f"feature_importance_lasso.{fmt}"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_rf(df, out_dir):
    top = df.head(30).sort_values("rf_importance")
    if top.empty: return
    fig, ax = plt.subplots(figsize=(7, max(5, len(top)*0.3)))
    colors = ["#E74C3C" if s else "#95A5A6" for s in top["rf_selected"]]
    ax.barh(top["gene"], top["rf_importance"], xerr=top["rf_std"], color=colors, alpha=0.75,
            error_kw={"elinewidth":0.8,"capsize":3})
    ax.set_xlabel("Mean Decrease in Impurity (±SD)"); ax.set_title("Random Forest Feature Importance")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for fmt in ["png","svg"]: plt.savefig(os.path.join(out_dir, f"feature_importance_rf.{fmt}"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_venn(lasso_g, svm_g, rf_g, out_dir):
    sets = [lasso_g, svm_g, rf_g]; names = ["LASSO","SVM-RFE","Random Forest"]
    centers = [(3.5,4.0),(5.5,4.0),(4.5,5.7)]; colors = ["#AED6F1","#A9DFBF","#F9E79F"]
    fig, ax = plt.subplots(figsize=(7,6))
    for (cx,cy), color, name in zip(centers, colors, names):
        ax.add_patch(plt.Circle((cx,cy), 1.8, alpha=0.4, color=color))
        ax.text(cx, cy-2.15, name, ha="center", fontsize=11, fontweight="bold")
    regions = [(2.7,4.0,len(lasso_g-svm_g-rf_g)),(6.3,4.0,len(svm_g-lasso_g-rf_g)),
               (4.5,6.2,len(rf_g-lasso_g-svm_g)),(4.4,4.05,len((lasso_g&svm_g)-rf_g)),
               (3.6,5.0,len((lasso_g&rf_g)-svm_g)),(5.4,5.0,len((svm_g&rf_g)-lasso_g)),
               (4.5,4.82,len(lasso_g&svm_g&rf_g))]
    for (x,y,n) in regions:
        ax.text(x, y, str(n), ha="center", fontsize=13, fontweight="bold")
    ax.set_xlim(0,9); ax.set_ylim(1,8.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Feature Selection Overlap (3 Methods)", fontsize=13, pad=10)
    plt.tight_layout()
    for fmt in ["png","svg"]: plt.savefig(os.path.join(out_dir, f"venn_feature_overlap.{fmt}"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DISCOVERY_DATASETS[0])
    args = ap.parse_args()

    deg_path  = os.path.join(RESULTS_DIR, "deg_list.csv")
    deg_genes = pd.read_csv(deg_path)["gene"].tolist() if os.path.exists(deg_path) else None
    if deg_genes: log.info(f"Using {len(deg_genes)} DEG genes as input")

    X, y = load_data(args.dataset, deg_genes)

    lasso_df = run_lasso(X, y)
    svm_df   = run_svm_rfe(X, y)
    rf_df    = run_rf(X, y)

    lasso_df.to_csv(os.path.join(RESULTS_DIR, "lasso_features.csv"), index=False)
    svm_df.to_csv(os.path.join(RESULTS_DIR, "svm_rfe_features.csv"), index=False)
    rf_df.to_csv(os.path.join(RESULTS_DIR, "rf_importance.csv"), index=False)

    full, panel = build_consensus(lasso_df, svm_df, rf_df)
    full.to_csv(os.path.join(RESULTS_DIR, "feature_importance_all.csv"), index=False)
    panel.to_csv(os.path.join(RESULTS_DIR, "consensus_features.csv"), index=False)
    panel[["gene","display_label","gene_symbol","biotype","direction","votes","composite_score"]].to_csv(os.path.join(RESULTS_DIR, "final_gene_panel.csv"), index=False)

    save_manuscript_inputs(lasso_df, svm_df, rf_df, full, panel)

    log.info("\nFINAL PANEL:")
    for _, r in panel.iterrows():
        log.info(f"  {r['display_label']:<25} votes={r['votes']}  score={r['composite_score']:.4f}")

    plot_lasso(lasso_df, FIGURES_DIR)
    plot_rf(rf_df, FIGURES_DIR)
    plot_venn(set(lasso_df[lasso_df["lasso_selected"]]["gene"]),
              set(svm_df[svm_df["svm_selected"]]["gene"]),
              set(rf_df[rf_df["rf_selected"]]["gene"]), FIGURES_DIR)
    log.info("Feature selection complete.")


if __name__ == "__main__":
    main()
