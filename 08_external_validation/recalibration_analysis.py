#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


def parse_args():
    p = argparse.ArgumentParser(description="Recalibration sensitivity analysis for reduced-score external validation.")
    p.add_argument("--discovery-scores", required=True, type=Path)
    p.add_argument("--external-scores", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    return p.parse_args()


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_table_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", compression="infer")


def clean_group(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"1", "t2d", "type 2 diabetes", "type2diabetes", "diabetic"}:
        return "T2D"
    if s in {"0", "nd", "control", "non-diabetic", "nondiabetic", "non diabetic"}:
        return "ND"
    return None


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    score_candidates = ["reduced_score", "score"]
    group_candidates = ["group", "label", "class"]

    score_col = next((c for c in score_candidates if c in df.columns), None)
    group_col = next((c for c in group_candidates if c in df.columns), None)

    if score_col is None:
        numeric_cols = []
        for c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            frac = vals.notna().mean()
            nunique = vals.dropna().nunique()
            if frac > 0.8 and nunique > 5:
                numeric_cols.append(c)
        if not numeric_cols:
            raise ValueError("Could not detect score column.")
        score_col = numeric_cols[0]

    if group_col is None:
        for c in df.columns:
            mapped = df[c].map(clean_group)
            if mapped.notna().sum() >= len(df) * 0.8:
                group_col = c
                break
        if group_col is None:
            raise ValueError("Could not detect group column.")

    return score_col, group_col


def load_scores(path: Path) -> pd.DataFrame:
    df = read_table_auto(path).copy()
    score_col, group_col = detect_columns(df)
    out = pd.DataFrame({
        "score": pd.to_numeric(df[score_col], errors="coerce"),
        "group": df[group_col].map(clean_group),
    })
    if "sample" in df.columns:
        out["sample"] = df["sample"].astype(str)
    else:
        out["sample"] = [f"S{i+1}" for i in range(len(out))]
    out = out.dropna(subset=["score", "group"]).reset_index(drop=True)
    return out


def youden_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, score)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def metrics_from_score(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "AUC": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan,
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) == 2 else np.nan,
        "MCC": float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) == 2 else np.nan,
        "Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def metrics_from_prob(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "AUC": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) == 2 else np.nan,
        "MCC": float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) == 2 else np.nan,
        "Threshold": float(threshold),
        "Brier": float(brier_score_loss(y_true, prob)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def loocv_logistic_prob(score: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    n = len(score)
    prob = np.zeros(n, dtype=float)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        x_train = score[mask].reshape(-1, 1)
        y_train = y_true[mask]

        model = LogisticRegression(max_iter=5000, solver="lbfgs")
        model.fit(x_train, y_train)
        prob[i] = model.predict_proba(score[i].reshape(1, -1))[0, 1]

    return prob


def group_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    for grp in ["ND", "T2D"]:
        x = df.loc[df["group"] == grp, "score"].values.astype(float)
        out[grp] = {
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "median": float(np.median(x)),
        }
    return out


def make_plots(disc: pd.DataFrame, ext: pd.DataFrame, ext_scores: pd.DataFrame, frozen_thr: float, metrics_map: dict, outdir: Path):
    plt.figure(figsize=(8, 5.5))
    positions = [1, 2, 4, 5]
    data = [
        disc.loc[disc["group"] == "ND", "score"].values,
        disc.loc[disc["group"] == "T2D", "score"].values,
        ext.loc[ext["group"] == "ND", "score"].values,
        ext.loc[ext["group"] == "T2D", "score"].values,
    ]
    plt.boxplot(data, positions=positions, widths=0.7, tick_labels=["Disc ND", "Disc T2D", "Ext ND", "Ext T2D"])
    plt.axhline(frozen_thr, linestyle="--", linewidth=1, label=f"Frozen threshold = {frozen_thr:.3f}")
    plt.ylabel("Reduced score")
    plt.title("Raw reduced-score distributions")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "plot_score_distributions.png", dpi=300)
    plt.close()

    methods = ["baseline_frozen_threshold", "unsupervised_mean_centering", "unsupervised_location_scale_alignment", "external_youden_threshold"]
    sens = [metrics_map[m]["Sensitivity"] for m in methods]
    spec = [metrics_map[m]["Specificity"] for m in methods]
    x = np.arange(len(methods))
    width = 0.36

    plt.figure(figsize=(9, 5.5))
    plt.bar(x - width/2, sens, width, label="Sensitivity")
    plt.bar(x + width/2, spec, width, label="Specificity")
    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Metric")
    plt.title("External threshold-dependent performance by recalibration scenario")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "plot_threshold_scenarios.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6.5, 5.5))
    nd = ext_scores.loc[ext_scores["group"] == "ND", "loocv_recalibrated_prob"].values
    t2d = ext_scores.loc[ext_scores["group"] == "T2D", "loocv_recalibrated_prob"].values
    plt.boxplot([nd, t2d], tick_labels=["Ext ND", "Ext T2D"])
    plt.axhline(0.5, linestyle="--", linewidth=1, label="Probability threshold = 0.5")
    plt.ylabel("LOOCV recalibrated probability")
    plt.title("External LOOCV logistic recalibration")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "plot_recalibration_probabilities.png", dpi=300)
    plt.close()


def main():
    args = parse_args()
    safe_mkdir(args.outdir)

    disc = load_scores(args.discovery_scores)
    ext = load_scores(args.external_scores)

    y_disc = (disc["group"].values == "T2D").astype(int)
    y_ext = (ext["group"].values == "T2D").astype(int)

    frozen_thr = youden_threshold(y_disc, disc["score"].values)

    baseline = metrics_from_score(y_ext, ext["score"].values, frozen_thr)

    disc_mean_all = float(disc["score"].mean())
    ext_mean_all = float(ext["score"].mean())
    mean_shift = ext_mean_all - disc_mean_all
    ext_mean_centered = ext["score"].values - mean_shift
    mean_centered = metrics_from_score(y_ext, ext_mean_centered, frozen_thr)

    disc_sd_all = float(disc["score"].std(ddof=1))
    ext_sd_all = float(ext["score"].std(ddof=1))
    if ext_sd_all == 0 or not np.isfinite(ext_sd_all):
        ext_locscale = ext["score"].values.copy()
    else:
        ext_locscale = ((ext["score"].values - ext_mean_all) / ext_sd_all) * disc_sd_all + disc_mean_all
    locscale = metrics_from_score(y_ext, ext_locscale, frozen_thr)

    ext_youden_thr = youden_threshold(y_ext, ext["score"].values)
    ext_youden = metrics_from_score(y_ext, ext["score"].values, ext_youden_thr)

    loocv_prob = loocv_logistic_prob(ext["score"].values, y_ext)
    loocv_recal = metrics_from_prob(y_ext, loocv_prob, threshold=0.5)

    ext_scores = ext.copy()
    ext_scores["baseline_score"] = ext["score"].values
    ext_scores["mean_centered_score"] = ext_mean_centered
    ext_scores["locscale_aligned_score"] = ext_locscale
    ext_scores["loocv_recalibrated_prob"] = loocv_prob
    ext_scores.to_csv(args.outdir / "external_scores_recalibrated.csv", index=False)

    rows = [
        {"method": "baseline_frozen_threshold", "uses_external_labels": False, **baseline},
        {"method": "unsupervised_mean_centering", "uses_external_labels": False, **mean_centered},
        {"method": "unsupervised_location_scale_alignment", "uses_external_labels": False, **locscale},
        {"method": "external_youden_threshold", "uses_external_labels": True, **ext_youden},
        {"method": "external_loocv_logistic_recalibration", "uses_external_labels": True, **loocv_recal},
    ]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(args.outdir / "recalibration_metrics.csv", index=False)

    summary = {
        "frozen_discovery_threshold": frozen_thr,
        "external_youden_threshold": ext_youden_thr,
        "overall_mean_shift_external_minus_discovery": mean_shift,
        "discovery_overall_mean": disc_mean_all,
        "external_overall_mean": ext_mean_all,
        "discovery_overall_sd": disc_sd_all,
        "external_overall_sd": ext_sd_all,
        "discovery_group_stats": group_stats(disc),
        "external_group_stats": group_stats(ext),
        "metrics": rows,
        "notes": {
            "baseline_frozen_threshold": "Primary external validation operating point; no external labels used.",
            "unsupervised_mean_centering": "Sensitivity analysis using unlabeled external score mean-shift correction.",
            "unsupervised_location_scale_alignment": "Sensitivity analysis aligning external score mean and SD to discovery.",
            "external_youden_threshold": "Descriptive upper bound; uses external labels and is not independent validation.",
            "external_loocv_logistic_recalibration": "Sensitivity analysis using external labels with LOOCV to reduce optimism.",
        },
    }
    with open(args.outdir / "recalibration_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    report = [
        "Recalibration sensitivity analysis of reduced score",
        "===============================================",
        "",
        f"Frozen discovery threshold: {frozen_thr:.5f}",
        f"External Youden threshold: {ext_youden_thr:.5f}",
        "",
        f"Discovery overall mean / SD: {disc_mean_all:.5f} / {disc_sd_all:.5f}",
        f"External overall mean / SD:  {ext_mean_all:.5f} / {ext_sd_all:.5f}",
        f"Overall mean shift (External - Discovery): {mean_shift:.5f}",
        "",
        "Discovery group means:",
        f"  ND:  {summary['discovery_group_stats']['ND']['mean']:.5f}",
        f"  T2D: {summary['discovery_group_stats']['T2D']['mean']:.5f}",
        "External group means:",
        f"  ND:  {summary['external_group_stats']['ND']['mean']:.5f}",
        f"  T2D: {summary['external_group_stats']['T2D']['mean']:.5f}",
        "",
        "Metrics by method:",
    ]
    for row in rows:
        report.extend([
            f"- {row['method']}",
            f"    Uses external labels: {row['uses_external_labels']}",
            f"    AUC: {row['AUC']:.3f}" if pd.notna(row['AUC']) else "    AUC: nan",
            f"    Sensitivity: {row['Sensitivity']:.3f}" if pd.notna(row['Sensitivity']) else "    Sensitivity: nan",
            f"    Specificity: {row['Specificity']:.3f}" if pd.notna(row['Specificity']) else "    Specificity: nan",
            f"    Balanced accuracy: {row['BalancedAccuracy']:.3f}" if pd.notna(row['BalancedAccuracy']) else "    Balanced accuracy: nan",
            f"    MCC: {row['MCC']:.3f}" if pd.notna(row['MCC']) else "    MCC: nan",
            f"    Threshold: {row['Threshold']:.5f}" if 'Threshold' in row and pd.notna(row['Threshold']) else "    Threshold: n/a",
            f"    Brier: {row['Brier']:.5f}" if 'Brier' in row and pd.notna(row.get('Brier', np.nan)) else "",
        ])
    report.extend([
        "",
        "Interpretation guide:",
        "  - baseline_frozen_threshold is the strict external validation operating point.",
        "  - unsupervised_* methods do not use external labels; they test whether simple cohort-level normalization can rescue threshold portability.",
        "  - external_youden_threshold is descriptive only.",
        "  - external_loocv_logistic_recalibration uses external labels and supports interpretation, not independent validation.",
    ])
    with open(args.outdir / "recalibration_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join([x for x in report if x != ""]) + "\n")

    metrics_map = {row["method"]: row for row in rows}
    make_plots(disc, ext, ext_scores, frozen_thr, metrics_map, args.outdir)
    print(f"Wrote outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
