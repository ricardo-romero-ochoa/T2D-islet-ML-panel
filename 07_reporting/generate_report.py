#!/usr/bin/env python3
"""
07_reporting/generate_report.py

Generates all manuscript-ready tables and prints key statistics
for copy-paste into the Methods and Results sections.

Outputs (in manuscript_tables/):
  Table1_dataset_characteristics.csv
  Table2_deg_summary.csv
  Table3_model_cv_performance.csv
  Table4_loocv_performance.csv
  Table5_final_gene_panel.csv
"""

import os, sys, logging
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_PROCESSED, RESULTS_DIR, LOGS_DIR,
                    DISCOVERY_DATASETS, VALIDATION_DATASETS, ALL_DATASETS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

TABLES_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "manuscript_tables")
os.makedirs(TABLES_DIR, exist_ok=True)

def table1():
    rows = []
    for gse_id in ALL_DATASETS:
        lp = os.path.join(DATA_PROCESSED, f"{gse_id}_labels.csv")
        ep = os.path.join(DATA_PROCESSED, f"{gse_id}_expr_normalized.csv")
        if not os.path.exists(lp): continue
        ldf   = pd.read_csv(lp)
        n_t2d = (ldf["label"]==1).sum(); n_ctrl = (ldf["label"]==0).sum()
        n_excl = (ldf["label"]==-1).sum()
        n_genes = "N/A"
        if os.path.exists(ep):
            try: n_genes = str(pd.read_csv(ep, index_col=0, nrows=0).shape[0] or pd.read_csv(ep, index_col=0).shape[0])
            except: pass
        role = "Discovery" if gse_id in DISCOVERY_DATASETS else "Validation" if gse_id in VALIDATION_DATASETS else "Supplementary"
        rows.append({"GEO Accession":gse_id,"Role":role,"N T2D":n_t2d,"N Control":n_ctrl,
                     "N Excluded":n_excl,"Genes/Probes":n_genes})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR,"Table1_dataset_characteristics.csv"), index=False)
    log.info("Table 1 saved")
    return df

def table2():
    dp = os.path.join(RESULTS_DIR, "deg_list.csv")
    rp = os.path.join(RESULTS_DIR, "deg_results.csv")
    if not os.path.exists(dp): log.warning("DEG list not found"); return None
    sig  = pd.read_csv(dp); full = pd.read_csv(rp) if os.path.exists(rp) else sig
    rows = [
        ("Total genes tested",       str(len(full)),        "FDR threshold",      "0.01"),
        ("Significant DEGs",         str(len(sig)),         "|log₂FC| threshold", "1.5"),
        ("Up-regulated in T2D",      str((sig["log2FC"]>0).sum()), "Method",      "Moderated t-test (eBayes)"),
        ("Down-regulated in T2D",    str((sig["log2FC"]<0).sum()), "Correction",  "Benjamini-Hochberg"),
    ]
    df = pd.DataFrame(rows, columns=["Parameter","Value","Parameter_2","Value_2"])
    df.to_csv(os.path.join(TABLES_DIR,"Table2_deg_summary.csv"), index=False)
    log.info("Table 2 saved")
    return df

def table3():
    cp = os.path.join(RESULTS_DIR, "model_cv_performance.csv")
    if not os.path.exists(cp): log.warning("CV performance not found"); return None
    df = pd.read_csv(cp)
    df["AUC (mean±SD)"] = df.apply(lambda r: f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}", axis=1)
    df["F1 (mean±SD)"]  = df.apply(lambda r: f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}", axis=1)
    df["MCC (mean±SD)"] = df.apply(lambda r: f"{r['mcc_mean']:.3f} ± {r['mcc_std']:.3f}", axis=1)
    out = df[["model","AUC (mean±SD)","F1 (mean±SD)","MCC (mean±SD)"]].rename(columns={"model":"Model"})
    out.to_csv(os.path.join(TABLES_DIR,"Table3_model_cv_performance.csv"), index=False)
    log.info("Table 3 saved")
    return out

def table4():
    lp = os.path.join(RESULTS_DIR, "loocv_performance.csv")
    if not os.path.exists(lp): log.warning("LOOCV performance not found — run loocv_validation.py"); return None
    df = pd.read_csv(lp)
    df["AUC (95% CI)"] = df.apply(lambda r: f"{r['auc']:.3f} [{r['auc_ci_lo']:.3f}–{r['auc_ci_hi']:.3f}]", axis=1)
    cols = ["n_samples","n_t2d","n_nd","AUC (95% CI)","sensitivity","specificity","ppv","npv","f1","mcc","accuracy","brier"]
    out = df[[c for c in cols if c in df.columns]].rename(columns={
        "n_samples":"N","n_t2d":"N T2D","n_nd":"N ND",
        "sensitivity":"Sensitivity","specificity":"Specificity",
        "ppv":"PPV","npv":"NPV","f1":"F1","mcc":"MCC","accuracy":"Accuracy","brier":"Brier Score"
    })
    for col in ["Sensitivity","Specificity","PPV","NPV","F1","MCC","Accuracy","Brier Score"]:
        if col in out.columns: out[col] = out[col].round(4)
    out.to_csv(os.path.join(TABLES_DIR,"Table4_loocv_performance.csv"), index=False)
    log.info("Table 4 saved")
    return out

def table5():
    pp  = os.path.join(RESULTS_DIR, "final_gene_panel.csv")
    dp  = os.path.join(RESULTS_DIR, "deg_results.csv")
    lp  = os.path.join(RESULTS_DIR, "lasso_features.csv")
    rfp = os.path.join(RESULTS_DIR, "rf_importance.csv")
    svmp= os.path.join(RESULTS_DIR, "svm_rfe_features.csv")
    if not os.path.exists(pp): log.warning("Gene panel not found"); return None

    panel = pd.read_csv(pp)
    if os.path.exists(dp):
        deg   = pd.read_csv(dp)[["gene","log2FC","adj_pvalue","mean_T2D","mean_Ctrl"]]
        panel = panel.merge(deg, on="gene", how="left")
    if os.path.exists(lp):
        panel = panel.merge(pd.read_csv(lp)[["gene","lasso_coef"]], on="gene", how="left")
    if os.path.exists(rfp):
        panel = panel.merge(pd.read_csv(rfp)[["gene","rf_importance","rf_rank"]], on="gene", how="left")
    if os.path.exists(svmp):
        panel = panel.merge(pd.read_csv(svmp)[["gene","svm_ranking"]], on="gene", how="left")

    if "log2FC" in panel.columns:
        panel["Direction"] = panel["log2FC"].apply(lambda x: "Up in T2D" if x>0 else "Down in T2D" if x<0 else "N/A")

    panel = panel.rename(columns={
        "gene":"Ensembl ID","gene_symbol":"Gene Symbol","votes":"Votes","composite_score":"Composite Score",
        "log2FC":"log2FC (T2D/ND)","adj_pvalue":"FDR adj. p","lasso_coef":"LASSO Coef.",
        "rf_importance":"RF Importance","rf_rank":"RF Rank","svm_ranking":"SVM-RFE Rank"
    })
    panel.to_csv(os.path.join(TABLES_DIR,"Table5_final_gene_panel.csv"), index=False)
    log.info("Table 5 saved")
    return panel

def print_summary():
    log.info("\n" + "="*60)
    log.info("KEY STATISTICS FOR MANUSCRIPT")
    log.info("="*60)

    total_t2d = total_ctrl = 0
    for gse_id in ALL_DATASETS:
        lp = os.path.join(DATA_PROCESSED, f"{gse_id}_labels.csv")
        if os.path.exists(lp):
            ldf = pd.read_csv(lp)
            total_t2d  += (ldf["label"]==1).sum()
            total_ctrl += (ldf["label"]==0).sum()

    log.info(f"\nDataset: {DISCOVERY_DATASETS[0]}")
    log.info(f"  Total samples analyzed: {total_t2d + total_ctrl}")
    log.info(f"  T2D: {total_t2d}  |  Non-diabetic: {total_ctrl}")

    dp = os.path.join(RESULTS_DIR, "deg_list.csv")
    if os.path.exists(dp):
        deg = pd.read_csv(dp)
        log.info(f"\nDifferential Expression (FDR<0.01, |log2FC|≥1.5):")
        log.info(f"  Total DEGs: {len(deg)}")
        log.info(f"  Up: {(deg['log2FC']>0).sum()}  Down: {(deg['log2FC']<0).sum()}")

    pp = os.path.join(RESULTS_DIR, "final_gene_panel.csv")
    if os.path.exists(pp):
        panel = pd.read_csv(pp)
        log.info(f"\nFeature Selection:")
        log.info(f"  Panel size: {len(panel)} genes")
        sym_col = "gene_symbol" if "gene_symbol" in panel.columns else "gene"
        log.info(f"  Genes: {panel[sym_col].tolist()}")

    lp = os.path.join(RESULTS_DIR, "loocv_performance.csv")
    if os.path.exists(lp):
        loocv = pd.read_csv(lp).iloc[0]
        log.info(f"\nPrimary Validation (LOOCV, n={loocv['n_samples']}):")
        log.info(f"  AUC:         {loocv['auc']:.4f} [{loocv['auc_ci_lo']:.3f}–{loocv['auc_ci_hi']:.3f}]")
        log.info(f"  Sensitivity: {loocv['sensitivity']:.4f}")
        log.info(f"  Specificity: {loocv['specificity']:.4f}")
        log.info(f"  F1:          {loocv['f1']:.4f}")
        log.info(f"  MCC:         {loocv['mcc']:.4f}")
        log.info(f"  Misclassified: {int(loocv['fp'] + loocv['fn'])}")

    cp = os.path.join(RESULTS_DIR, "model_cv_performance.csv")
    if os.path.exists(cp):
        cv  = pd.read_csv(cp)
        best = cv.sort_values("auc_mean", ascending=False).iloc[0]
        log.info(f"\nRepeated CV (best model: {best['model']}):")
        log.info(f"  AUC: {best['auc_mean']:.3f}±{best['auc_std']:.3f}")

    log.info("\n" + "="*60)

def main():
    log.info("Generating manuscript tables...\n")
    table1(); table2(); table3(); table4(); table5()
    print_summary()
    log.info(f"\nAll tables saved to: {TABLES_DIR}/")
    for f in sorted(os.listdir(TABLES_DIR)):
        log.info(f"  {f}")

if __name__ == "__main__":
    main()
