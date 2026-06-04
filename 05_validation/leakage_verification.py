#!/usr/bin/env python3
"""
05_validation/leakage_verification.py

Leakage-verification sensitivity analysis.

Compares the proper LOOCV workflow against an improper workflow in which global
quantile normalization is applied to the full cohort before cross-validation.
The resulting CSVs feed Figure 7E directly.
"""

import os, sys, logging, warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score, matthews_corrcoef, accuracy_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR, FIGURE_INPUTS_DIR, LOGS_DIR, DISCOVERY_DATASETS, RANDOM_STATE, FIGURE_DPI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "05_leakage_verification.log")), logging.StreamHandler()]
)
log = logging.getLogger(__name__)


def load_data(gse_id, panel_genes):
    expr   = pd.read_csv(os.path.join(DATA_PROCESSED, f"{gse_id}_expr_normalized.csv"), index_col=0)
    ldf    = pd.read_csv(os.path.join(DATA_PROCESSED, f"{gse_id}_labels.csv"))
    labels = pd.Series(ldf["label"].values, index=ldf["sample_id"].values).dropna()
    common = [c for c in expr.columns if c in labels.index and labels[c] in [0,1]]
    y      = labels[common].astype(int)
    avail  = [g for g in panel_genes if g in expr.index]
    X      = expr.loc[avail, common].T.fillna(0)
    return X, y, common


def quantile_normalize_rows(X):
    """Equalize sample distributions across features using full-cohort information."""
    X = np.asarray(X, dtype=float)
    sorted_rows = np.sort(X, axis=1)
    mean_ranks = sorted_rows.mean(axis=0)
    ranks = np.argsort(np.argsort(X, axis=1), axis=1)
    X_qn = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_qn[i, :] = mean_ranks[ranks[i, :]]
    return X_qn


def build_ensemble():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", VotingClassifier(estimators=[
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            ("rf",  RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ("lr",  LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
            ("gb",  GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE)),
        ], voting="soft"))
    ])


def compute_metrics(y_true, probas, threshold=0.5):
    preds = (probas >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, probas)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, preds)) if len(np.unique(preds)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, preds)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def save_curve(prefix, y_true, probas):
    fpr, tpr, thr = roc_curve(y_true, probas)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
        os.path.join(FIGURE_INPUTS_DIR, f"{prefix}_roc_points.csv"), index=False
    )


def plot_comparison(proper_auc, leak_auc, proper_curve, leak_curve):
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot(proper_curve[0], proper_curve[1], lw=2.4, color="#1F4E79", label=f"Proper LOOCV (AUC = {proper_auc:.3f})")
    ax.plot(leak_curve[0], leak_curve[1], lw=2.4, color="#C0392B", label=f"Global quantile norm before CV (AUC = {leak_auc:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.6, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Leakage-verification sensitivity analysis")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for fmt in ["png", "svg"]:
        plt.savefig(os.path.join(FIGURES_DIR, f"leakage_verification_roc.{fmt}"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(FIGURE_INPUTS_DIR, exist_ok=True)
    panel_path = os.path.join(RESULTS_DIR, "final_gene_panel.csv")
    if not os.path.exists(panel_path):
        log.error("Run feature_selection.py first")
        sys.exit(1)
    panel_genes = pd.read_csv(panel_path)["gene"].tolist()
    X_df, y, sample_ids = load_data(DISCOVERY_DATASETS[0], panel_genes)
    X = X_df.values.astype(float)
    y_arr = y.values.astype(int)

    loo = LeaveOneOut()
    model = build_ensemble()

    proper_p = cross_val_predict(model, X, y_arr, cv=loo, method="predict_proba")[:, 1]
    proper_metrics = compute_metrics(y_arr, proper_p)
    save_curve("fig7_proper_loocv", y_arr, proper_p)

    # Improper workflow: normalize the full cohort before CV
    X_leak = quantile_normalize_rows(X)
    leak_p = cross_val_predict(model, X_leak, y_arr, cv=loo, method="predict_proba")[:, 1]
    leak_metrics = compute_metrics(y_arr, leak_p)
    save_curve("fig7_leakage_global_qn", y_arr, leak_p)

    pd.DataFrame([
        {"workflow": "proper_loocv", **proper_metrics},
        {"workflow": "global_qn_before_cv", **leak_metrics},
    ]).to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig7_leakage_verification_metrics.csv"), index=False)

    pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": y_arr,
        "proper_loocv_proba": proper_p,
        "global_qn_before_cv_proba": leak_p,
    }).to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig7_leakage_verification_predictions.csv"), index=False)

    fpr1, tpr1, _ = roc_curve(y_arr, proper_p)
    fpr2, tpr2, _ = roc_curve(y_arr, leak_p)
    plot_comparison(proper_metrics["auc"], leak_metrics["auc"], (fpr1, tpr1), (fpr2, tpr2))

    log.info(f"Proper LOOCV AUC: {proper_metrics['auc']:.3f}")
    log.info(f"Leakage-control AUC: {leak_metrics['auc']:.3f}")


if __name__ == "__main__":
    main()
