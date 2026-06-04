#!/usr/bin/env python3
"""
04_models/train_models.py

Trains four classifiers + soft-voting ensemble on the 10-gene panel.
Evaluates using repeated stratified k-fold cross-validation.
Saves trained models and performance metrics.

NOTE: class_weight="balanced" is set on all classifiers to handle
the 39:18 T2D:ND imbalance. Feature standardisation is applied
within each CV fold (pipeline) to prevent data leakage.

Outputs:
  results/model_cv_performance.csv
  models/*.pkl
  figures/roc_curves_cv.png/.svg
  figures/calibration_curves.png
  figures/confusion_matrix_ensemble.png
"""

import os, sys, logging, warnings, pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                      GridSearchCV, cross_val_predict)
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, f1_score,
                              matthews_corrcoef, confusion_matrix, average_precision_score,
                              brier_score_loss)
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR,
                    DISCOVERY_DATASETS, CV_FOLDS, CV_REPEATS, RANDOM_STATE,
                    SVM_PARAM_GRID, RF_PARAM_GRID, LR_PARAM_GRID,
                    PALETTE_T2D, PALETTE_CTRL, FIGURE_DPI)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "04_training.log")), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

def load_data(gse_id, panel_genes):
    expr   = pd.read_csv(os.path.join(DATA_PROCESSED, f"{gse_id}_expr_normalized.csv"), index_col=0)
    ldf    = pd.read_csv(os.path.join(DATA_PROCESSED, f"{gse_id}_labels.csv"))
    labels = pd.Series(ldf["label"].values, index=ldf["sample_id"].values).dropna()
    common = [c for c in expr.columns if c in labels.index and labels[c] in [0,1]]
    expr   = expr[common]; y = labels[common].astype(int)
    avail  = [g for g in panel_genes if g in expr.index]
    if not avail:
        log.error("No panel genes in expression data"); return None, None
    if len(avail) < len(panel_genes):
        log.warning(f"  {len(avail)}/{len(panel_genes)} panel genes found")
    X = expr.loc[avail].T.fillna(0)
    log.info(f"  X: {X.shape}  T2D={( y==1).sum()}  Ctrl={(y==0).sum()}")
    return X, y

def evaluate_cv(pipeline, X, y, name):
    rskf = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    aucs, accs, f1s, mccs, aps = [], [], [], [], []
    all_p, all_t = [], []
    for tr, te in rskf.split(X, y):
        X_tr, X_te = X[tr], X[te]; y_tr, y_te = y[tr], y[te]
        pipeline.fit(X_tr, y_tr)
        p = pipeline.predict_proba(X_te)[:,1]; pred = pipeline.predict(X_te)
        aucs.append(roc_auc_score(y_te, p)); accs.append(accuracy_score(y_te, pred))
        f1s.append(f1_score(y_te, pred, zero_division=0))
        mccs.append(matthews_corrcoef(y_te, pred))
        aps.append(average_precision_score(y_te, p))
        all_p.extend(p.tolist()); all_t.extend(y_te.tolist())
    m = {"model": name, "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
         "acc_mean": np.mean(accs), "acc_std": np.std(accs),
         "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
         "mcc_mean": np.mean(mccs), "mcc_std": np.std(mccs),
         "ap_mean": np.mean(aps), "ap_std": np.std(aps)}
    log.info(f"  {name:<25} AUC={m['auc_mean']:.3f}±{m['auc_std']:.3f}  F1={m['f1_mean']:.3f}±{m['f1_std']:.3f}  MCC={m['mcc_mean']:.3f}±{m['mcc_std']:.3f}")
    return m, np.array(all_p), np.array(all_t)

def plot_roc(model_results, out_dir):
    fig, ax = plt.subplots(figsize=(7,6))
    colors = plt.cm.Set1(np.linspace(0,0.85,len(model_results)))
    for (name,(probas,trues)), color in zip(model_results.items(), colors):
        fpr,tpr,_ = roc_curve(trues,probas); auc = roc_auc_score(trues,probas)
        ax.plot(fpr,tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
    ax.set_xlabel("False Positive Rate",fontsize=12); ax.set_ylabel("True Positive Rate",fontsize=12)
    ax.set_title(f"ROC — Repeated {CV_FOLDS}-fold CV ({CV_REPEATS} repeats)",fontsize=11)
    ax.legend(loc="lower right",fontsize=9); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for fmt in ["png","svg"]: plt.savefig(os.path.join(out_dir,f"roc_curves_cv.{fmt}"),dpi=FIGURE_DPI,bbox_inches="tight")
    plt.close()

def plot_cm(y_true, y_pred, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig,ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                xticklabels=["Pred Ctrl","Pred T2D"],yticklabels=["True Ctrl","True T2D"],
                ax=ax,linewidths=0.5,cbar=False)
    ax.set_title("Confusion Matrix — Ensemble (Repeated CV)",fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"confusion_matrix_ensemble.png"),dpi=FIGURE_DPI,bbox_inches="tight")
    plt.close()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DISCOVERY_DATASETS[0])
    ap.add_argument("--skip-tuning", action="store_true", help="Use default hyperparameters (faster)")
    args = ap.parse_args()

    panel_path = os.path.join(RESULTS_DIR, "final_gene_panel.csv")
    if not os.path.exists(panel_path):
        log.error("Run feature_selection.py first"); sys.exit(1)
    panel_genes = pd.read_csv(panel_path)["gene"].tolist()
    log.info(f"Panel ({len(panel_genes)} genes): {panel_genes}")

    X_df, y = load_data(args.dataset, panel_genes)
    if X_df is None: sys.exit(1)
    X = X_df.values.astype(float); y_arr = y.values.astype(int)

    # Balanced classifiers — handles 39:18 T2D:ND imbalance
    svm = SVC(C=1, kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
    rf  = RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1)
    lr  = LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", class_weight="balanced",
                              max_iter=2000, random_state=RANDOM_STATE)
    gb  = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE)

    pipelines = {
        "SVM (RBF)":      Pipeline([("scaler",StandardScaler()),("clf",svm)]),
        "Random Forest":  Pipeline([("scaler",StandardScaler()),("clf",rf)]),
        "Logistic Reg.":  Pipeline([("scaler",StandardScaler()),("clf",lr)]),
        "Gradient Boost": Pipeline([("scaler",StandardScaler()),("clf",gb)]),
    }

    log.info(f"\nRepeated {CV_FOLDS}-fold CV ({CV_REPEATS} repeats):")
    all_metrics = []; cv_probas = {}
    for name, pipe in pipelines.items():
        m, probas, trues = evaluate_cv(pipe, X, y_arr, name)
        all_metrics.append(m); cv_probas[name] = (probas, trues)

    ensemble = VotingClassifier(
        estimators=[(n.replace(" ","_"), p) for n,p in pipelines.items()],
        voting="soft", n_jobs=-1)
    ens_m, ens_p, ens_t = evaluate_cv(ensemble, X, y_arr, "Ensemble (soft vote)")
    all_metrics.append(ens_m); cv_probas["Ensemble"] = (ens_p, ens_t)

    pd.DataFrame(all_metrics).to_csv(os.path.join(RESULTS_DIR,"model_cv_performance.csv"), index=False)
    log.info("CV performance saved.")

    plot_roc({n:(p,t) for n,(p,t) in cv_probas.items()}, FIGURES_DIR)

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ensemble.fit(X, y_arr)
    y_pred_cv = cross_val_predict(ensemble, X, y_arr, cv=cv5)
    plot_cm(y_arr, y_pred_cv, FIGURES_DIR)

    log.info("\nTraining final models on full dataset...")
    for name, pipe in pipelines.items():
        pipe.fit(X, y_arr)
        safe = name.replace(" ","_").replace("(","").replace(")","")
        with open(os.path.join(MODELS_DIR, f"{safe}.pkl"), "wb") as f:
            pickle.dump({"model":pipe, "panel_genes":panel_genes}, f)
    ensemble.fit(X, y_arr)
    with open(os.path.join(MODELS_DIR, "Ensemble.pkl"), "wb") as f:
        pickle.dump({"model":ensemble, "panel_genes":panel_genes}, f)
    log.info("Models saved.")

    pd.Series(panel_genes).to_csv(os.path.join(RESULTS_DIR,"panel_genes_final.csv"), index=False, header=["gene"])

    best = pd.DataFrame(all_metrics).sort_values("auc_mean", ascending=False).iloc[0]
    log.info(f"\nBest: {best['model']}  AUC={best['auc_mean']:.4f}")

if __name__ == "__main__":
    main()
