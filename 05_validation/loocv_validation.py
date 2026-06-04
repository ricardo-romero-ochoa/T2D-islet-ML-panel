#!/usr/bin/env python3
"""
05_validation/loocv_validation.py

PRIMARY VALIDATION — Leave-One-Out Cross-Validation (LOOCV).

Additional manuscript-facing outputs:
  results/figure_inputs/fig7_loocv_predictions.csv
  results/figure_inputs/fig7_loocv_roc_points.csv
  results/figure_inputs/fig7_loocv_metrics.csv
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
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score, matthews_corrcoef,
                              confusion_matrix, brier_score_loss, average_precision_score,
                              accuracy_score)
from sklearn.calibration import calibration_curve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR, FIGURE_INPUTS_DIR, LOGS_DIR,
                    DISCOVERY_DATASETS, RANDOM_STATE, PALETTE_T2D, FIGURE_DPI)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "05_loocv.log")), logging.StreamHandler()]
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
    log.info(f"  X: {X.shape}  T2D={( y==1).sum()}  ND={(y==0).sum()}")
    return X, y, common


def build_ensemble():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", VotingClassifier(estimators=[
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            ("rf",  RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=RANDOM_STATE, n_jobs=-1)),
            ("lr",  LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000,
                                        random_state=RANDOM_STATE)),
            ("gb",  GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE)),
        ], voting="soft"))
    ])


def compute_metrics(y_true, probas, threshold=0.5):
    preds = (probas >= threshold).astype(int)
    cm    = confusion_matrix(y_true, preds, labels=[0,1])
    tn,fp,fn,tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    auc  = roc_auc_score(y_true, probas) if len(np.unique(y_true)) > 1 else np.nan
    n1,n0 = int((y_true==1).sum()), int((y_true==0).sum())
    q1 = auc / (2-auc) if not np.isnan(auc) else np.nan
    q2 = 2*auc**2 / (1+auc) if not np.isnan(auc) else np.nan
    se_auc = np.sqrt((auc*(1-auc)+(n1-1)*(q1-auc**2)+(n0-1)*(q2-auc**2))/(n1*n0)) if not np.isnan(auc) and n1>0 and n0>0 else np.nan
    return {
        "n_samples": len(y_true), "n_t2d": int(n1), "n_nd": int(n0),
        "auc":  round(float(auc), 4), "auc_ci_lo": round(float(auc-1.96*se_auc), 4) if np.isfinite(se_auc) else np.nan,
        "auc_ci_hi": round(float(auc+1.96*se_auc), 4) if np.isfinite(se_auc) else np.nan,
        "sensitivity": round(tp/(tp+fn) if tp+fn>0 else np.nan, 4),
        "specificity": round(tn/(tn+fp) if tn+fp>0 else np.nan, 4),
        "ppv":   round(tp/(tp+fp) if tp+fp>0 else np.nan, 4),
        "npv":   round(tn/(tn+fn) if tn+fn>0 else np.nan, 4),
        "f1":    round(f1_score(y_true, preds, zero_division=0), 4),
        "mcc":   round(matthews_corrcoef(y_true, preds) if len(np.unique(preds))>1 else np.nan, 4),
        "accuracy": round(accuracy_score(y_true, preds), 4),
        "brier": round(brier_score_loss(y_true, probas), 4),
        "ap":    round(average_precision_score(y_true, probas) if len(np.unique(y_true))>1 else np.nan, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def plot_roc(y_true, probas, metrics, out_dir):
    fpr,tpr,_ = roc_curve(y_true, probas)
    auc = metrics["auc"]; ci_lo = metrics["auc_ci_lo"]; ci_hi = metrics["auc_ci_hi"]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(fpr,tpr, color="#1F4E79", lw=2.5,
            label=f"Ensemble (AUC = {auc:.3f})")
    ax.fill_between(fpr,tpr, alpha=0.08, color="#1F4E79")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Random")
    ax.set_xlabel("False Positive Rate",fontsize=13)
    ax.set_ylabel("True Positive Rate",fontsize=13)
    ax.set_title(f"LOOCV ROC Curve\n10-gene T2D Diagnostic Panel (n={len(y_true)})",fontsize=12)
    ax.legend(fontsize=10,loc="lower right",framealpha=0.9)
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.02])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for fmt in ["png","svg"]:
        plt.savefig(os.path.join(out_dir,f"loocv_roc_curve.{fmt}"),dpi=FIGURE_DPI,bbox_inches="tight")
    plt.close()
    log.info("  LOOCV ROC curve saved")


def plot_calibration(y_true, probas, out_dir):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot([0,1],[0,1],"k--",label="Perfect calibration")
    try:
        fp, mp = calibration_curve(y_true, probas, n_bins=8, strategy="uniform")
        brier  = brier_score_loss(y_true, probas)
        ax.plot(mp, fp, "s-", color="#1F4E79", lw=2, label=f"Ensemble (Brier={brier:.3f})")
    except Exception as e:
        log.warning(f"  Calibration plot: {e}")
    ax.set_xlabel("Mean Predicted Probability",fontsize=11)
    ax.set_ylabel("Fraction of Positives",fontsize=11)
    ax.set_title("Calibration Curve (LOOCV)",fontsize=11); ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"loocv_calibration.png"),dpi=FIGURE_DPI,bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(FIGURE_INPUTS_DIR, exist_ok=True)
    panel_path = os.path.join(RESULTS_DIR, "final_gene_panel.csv")
    if not os.path.exists(panel_path):
        log.error("Run feature_selection.py first"); sys.exit(1)
    panel_genes = pd.read_csv(panel_path)["gene"].tolist()

    X_df, y, sample_ids = load_data(DISCOVERY_DATASETS[0], panel_genes)
    X = X_df.values.astype(float); y_arr = y.values.astype(int)

    log.info(f"\nRunning LOOCV (n={len(y_arr)})...")
    ensemble = build_ensemble()
    loo      = LeaveOneOut()
    probas   = cross_val_predict(ensemble, X, y_arr, cv=loo, method="predict_proba")[:,1]
    preds    = (probas >= 0.5).astype(int)

    metrics = compute_metrics(y_arr, probas)

    log.info("\n" + "="*55)
    log.info("LOOCV PERFORMANCE (PRIMARY VALIDATION)")
    log.info("="*55)
    log.info(f"  n = {metrics['n_samples']} ({metrics['n_t2d']} T2D, {metrics['n_nd']} ND)")
    log.info(f"  AUC:         {metrics['auc']:.4f}  [{metrics['auc_ci_lo']:.3f}–{metrics['auc_ci_hi']:.3f}]")
    log.info(f"  Sensitivity: {metrics['sensitivity']:.4f}  ({metrics['tp']} T2D correct / {metrics['n_t2d']} total)")
    log.info(f"  Specificity: {metrics['specificity']:.4f}  ({metrics['tn']} ND correct / {metrics['n_nd']} total)")
    log.info(f"  F1:          {metrics['f1']:.4f}")
    log.info(f"  MCC:         {metrics['mcc']:.4f}")
    log.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
    log.info(f"  Brier score: {metrics['brier']:.4f}")
    log.info(f"  Misclassified: {metrics['fp'] + metrics['fn']}")
    log.info("="*55)

    perf_df = pd.DataFrame([metrics])
    perf_df.to_csv(os.path.join(RESULTS_DIR,"loocv_performance.csv"), index=False)
    perf_df.to_csv(os.path.join(FIGURE_INPUTS_DIR,"fig7_loocv_metrics.csv"), index=False)

    per_sample = pd.DataFrame({
        "sample_id": sample_ids, "true_label": y_arr,
        "predicted_proba_T2D": probas, "predicted_label": preds,
        "correct": (preds == y_arr).astype(int)
    })
    per_sample.to_csv(os.path.join(RESULTS_DIR,"loocv_per_sample.csv"), index=False)
    per_sample.to_csv(os.path.join(FIGURE_INPUTS_DIR,"fig7_loocv_predictions.csv"), index=False)

    fpr, tpr, thresholds = roc_curve(y_arr, probas)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        os.path.join(FIGURE_INPUTS_DIR, "fig7_loocv_roc_points.csv"), index=False
    )

    plot_roc(y_arr, probas, metrics, FIGURES_DIR)
    plot_calibration(y_arr, probas, FIGURES_DIR)
    log.info("LOOCV complete.")


if __name__ == "__main__":
    main()
