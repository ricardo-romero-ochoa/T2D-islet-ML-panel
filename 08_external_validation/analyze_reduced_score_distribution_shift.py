#!/usr/bin/env python3
"""
Distribution-shift analysis for reduced-score external validation outputs.

Purpose
-------
Given discovery and external reduced-score CSVs from the fixed-threshold workflow,
this script quantifies and visualizes score distribution shift across cohorts.
It is designed to explain patterns such as:
  - strong external AUC but failed frozen-threshold transfer
  - upward or downward displacement of the external score distribution
  - group-specific shifts (ND and T2D separately)

Expected input CSVs
-------------------
Discovery and external score files should each contain at least:
  - a group column (default: 'group') with ND/T2D labels
  - a score column (default: 'reduced_score')

Typical examples are the outputs from:
  validate_gse50244_reduced_score_fixed_threshold.py
which produce:
  - discovery_reduced_scores.csv
  - external_reduced_scores.csv

Outputs
-------
Written to --outdir:
  - summary_by_cohort_group.csv
  - shift_statistics.csv
  - threshold_position.csv
  - distribution_shift_report.txt
  - distribution_shift_report.json
  - plot_hist_density.png
  - plot_boxplot.png
  - plot_ecdf.png
  - plot_threshold_fraction.png

Usage
-----
python analyze_reduced_score_distribution_shift.py \
  --discovery-scores discovery_reduced_scores.csv \
  --external-scores external_reduced_scores.csv \
  --metrics-summary metrics_summary.txt \
  --outdir distribution_shift_results

If --metrics-summary is omitted, provide --threshold explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance


def eprint(*args, **kwargs):
    print(*args, **kwargs)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_table_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", compression="infer")


def normalize_group(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"nd", "control", "non-diabetic", "nondiabetic", "non diabetic"}:
        return "ND"
    if s in {"t2d", "type 2 diabetes", "type2diabetes", "diabetic"}:
        return "T2D"
    return None


def infer_score_col(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested and requested in df.columns:
        return requested
    candidates = [
        "reduced_score",
        "score",
        "signed_score",
        "reducedScore",
        "ReducedScore",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find score column. Available columns: {list(df.columns)}"
    )


def infer_group_col(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested and requested in df.columns:
        return requested
    candidates = ["group", "label", "class", "phenotype"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find group column. Available columns: {list(df.columns)}"
    )


def load_scores(path: Path, cohort_name: str, score_col: Optional[str], group_col: Optional[str]) -> pd.DataFrame:
    df = read_table_auto(path).copy()
    sc = infer_score_col(df, score_col)
    gc = infer_group_col(df, group_col)

    out = pd.DataFrame({
        "cohort": cohort_name,
        "group": df[gc].map(normalize_group),
        "score": pd.to_numeric(df[sc], errors="coerce"),
    })

    if "sample" in df.columns:
        out["sample"] = df["sample"].astype(str)
    elif "sample_id" in df.columns:
        out["sample"] = df["sample_id"].astype(str)
    else:
        out["sample"] = [f"{cohort_name}_{i+1}" for i in range(len(df))]

    out = out[out["group"].isin(["ND", "T2D"])].copy()
    out = out[np.isfinite(out["score"])].copy()

    if out.empty:
        raise ValueError(f"No valid ND/T2D scored rows found in {path}")

    return out.reset_index(drop=True)


def parse_threshold_from_metrics_summary(path: Path) -> Optional[float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Frozen threshold:\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    return None


def summarize_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in sorted(df["cohort"].unique()):
        for group in ["ND", "T2D"]:
            sub = df[(df["cohort"] == cohort) & (df["group"] == group)]
            if sub.empty:
                continue
            vals = sub["score"].values.astype(float)
            q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
            rows.append({
                "cohort": cohort,
                "group": group,
                "n": int(len(vals)),
                "mean": float(np.mean(vals)),
                "median": float(med),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "min": float(np.min(vals)),
                "q1": float(q1),
                "q3": float(q3),
                "max": float(np.max(vals)),
            })
    return pd.DataFrame(rows)


def compare_distributions(discovery: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, d_sub, e_sub in [
        ("ALL", discovery, external),
        ("ND", discovery[discovery["group"] == "ND"], external[external["group"] == "ND"]),
        ("T2D", discovery[discovery["group"] == "T2D"], external[external["group"] == "T2D"]),
    ]:
        dv = d_sub["score"].values.astype(float)
        ev = e_sub["score"].values.astype(float)
        if len(dv) == 0 or len(ev) == 0:
            rows.append({
                "comparison": label,
                "discovery_n": int(len(dv)),
                "external_n": int(len(ev)),
                "mean_shift_external_minus_discovery": np.nan,
                "median_shift_external_minus_discovery": np.nan,
                "ks_statistic": np.nan,
                "ks_pvalue": np.nan,
                "wasserstein_distance": np.nan,
            })
            continue
        ks = ks_2samp(dv, ev)
        rows.append({
            "comparison": label,
            "discovery_n": int(len(dv)),
            "external_n": int(len(ev)),
            "mean_shift_external_minus_discovery": float(np.mean(ev) - np.mean(dv)),
            "median_shift_external_minus_discovery": float(np.median(ev) - np.median(dv)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "wasserstein_distance": float(wasserstein_distance(dv, ev)),
        })
    return pd.DataFrame(rows)


def threshold_position(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for cohort in sorted(df["cohort"].unique()):
        for group in ["ALL", "ND", "T2D"]:
            sub = df[df["cohort"] == cohort] if group == "ALL" else df[(df["cohort"] == cohort) & (df["group"] == group)]
            vals = sub["score"].values.astype(float)
            if len(vals) == 0:
                continue
            rows.append({
                "cohort": cohort,
                "group": group,
                "n": int(len(vals)),
                "fraction_above_or_equal_threshold": float(np.mean(vals >= threshold)),
                "fraction_below_threshold": float(np.mean(vals < threshold)),
                "threshold_percentile_within_distribution": float(np.mean(vals <= threshold)),
                "mean_minus_threshold": float(np.mean(vals) - threshold),
                "median_minus_threshold": float(np.median(vals) - threshold),
            })
    return pd.DataFrame(rows)


def compute_confusion_metrics(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for cohort in sorted(df["cohort"].unique()):
        sub = df[df["cohort"] == cohort].copy()
        y_true = (sub["group"] == "T2D").astype(int).values
        pred = (sub["score"].values >= threshold).astype(int)
        # force 2x2 layout
        tn = int(np.sum((y_true == 0) & (pred == 0)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        bal_acc = np.nan if np.isnan(sens) or np.isnan(spec) else 0.5 * (sens + spec)
        rows.append({
            "cohort": cohort,
            "threshold": float(threshold),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Sensitivity": float(sens),
            "Specificity": float(spec),
            "BalancedAccuracy": float(bal_acc),
        })
    return pd.DataFrame(rows)


def pretty_label(cohort: str, group: str) -> str:
    return f"{cohort} {group}"


def plot_hist_density(df: pd.DataFrame, threshold: float, outpath: Path) -> None:
    plt.figure(figsize=(8, 6))
    combos = [
        ("Discovery", "ND"),
        ("Discovery", "T2D"),
        ("External", "ND"),
        ("External", "T2D"),
    ]

    all_scores = df["score"].values.astype(float)
    xmin, xmax = float(np.min(all_scores)), float(np.max(all_scores))
    pad = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
    xgrid = np.linspace(xmin - pad, xmax + pad, 400)

    for cohort, group in combos:
        sub = df[(df["cohort"] == cohort) & (df["group"] == group)]
        vals = sub["score"].values.astype(float)
        if len(vals) == 0:
            continue
        if len(np.unique(vals)) > 1 and len(vals) > 2:
            dens = gaussian_kde(vals)(xgrid)
            plt.plot(xgrid, dens, lw=2, label=pretty_label(cohort, group))
        else:
            plt.axvline(float(vals[0]), lw=2, label=pretty_label(cohort, group))

    plt.axvline(threshold, linestyle="--", lw=2, label=f"Frozen threshold = {threshold:.5f}")
    plt.xlabel("Reduced score")
    plt.ylabel("Density")
    plt.title("Reduced-score distributions across cohorts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_boxplot(df: pd.DataFrame, threshold: float, outpath: Path) -> None:
    order = [("Discovery", "ND"), ("Discovery", "T2D"), ("External", "ND"), ("External", "T2D")]
    data = []
    labels = []
    for cohort, group in order:
        sub = df[(df["cohort"] == cohort) & (df["group"] == group)]
        if len(sub) == 0:
            continue
        data.append(sub["score"].values.astype(float))
        labels.append(f"{cohort}\n{group}")

    plt.figure(figsize=(7, 6))
    plt.boxplot(data, tick_labels=labels)
    plt.axhline(threshold, linestyle="--", lw=2, label=f"Frozen threshold = {threshold:.5f}")
    plt.ylabel("Reduced score")
    plt.title("Score distribution shift by cohort and class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_ecdf(df: pd.DataFrame, threshold: float, outpath: Path) -> None:
    plt.figure(figsize=(8, 6))
    combos = [
        ("Discovery", "ND"),
        ("Discovery", "T2D"),
        ("External", "ND"),
        ("External", "T2D"),
    ]
    for cohort, group in combos:
        sub = df[(df["cohort"] == cohort) & (df["group"] == group)]
        vals = np.sort(sub["score"].values.astype(float))
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        plt.step(vals, y, where="post", lw=2, label=pretty_label(cohort, group))
    plt.axvline(threshold, linestyle="--", lw=2, label=f"Frozen threshold = {threshold:.5f}")
    plt.xlabel("Reduced score")
    plt.ylabel("ECDF")
    plt.title("Empirical cumulative distributions of reduced score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_threshold_fraction(thr_df: pd.DataFrame, outpath: Path) -> None:
    sub = thr_df[thr_df["group"].isin(["ND", "T2D"])].copy()
    labels = [f"{r['cohort']}\n{r['group']}" for _, r in sub.iterrows()]
    values = sub["fraction_above_or_equal_threshold"].values.astype(float)

    plt.figure(figsize=(7, 5.5))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Fraction predicted T2D\n(score ≥ frozen threshold)")
    plt.title("Threshold behavior across cohorts and classes")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def build_report(
    summary_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    thr_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    threshold: float,
) -> str:
    def get_stat(cohort: str, group: str, col: str):
        row = summary_df[(summary_df["cohort"] == cohort) & (summary_df["group"] == group)]
        if row.empty:
            return math.nan
        return float(row.iloc[0][col])

    nd_shift = shift_df.loc[shift_df["comparison"] == "ND", "mean_shift_external_minus_discovery"]
    t2d_shift = shift_df.loc[shift_df["comparison"] == "T2D", "mean_shift_external_minus_discovery"]
    all_shift = shift_df.loc[shift_df["comparison"] == "ALL", "mean_shift_external_minus_discovery"]

    ext_nd_above = thr_df[(thr_df["cohort"] == "External") & (thr_df["group"] == "ND")]["fraction_above_or_equal_threshold"]
    ext_t2d_above = thr_df[(thr_df["cohort"] == "External") & (thr_df["group"] == "T2D")]["fraction_above_or_equal_threshold"]

    lines = [
        "Distribution-shift analysis of reduced score",
        "===========================================",
        "",
        f"Frozen threshold analyzed: {threshold:.5f}",
        "",
        "Group means:",
        f"  Discovery ND mean:  {get_stat('Discovery', 'ND', 'mean'):.5f}",
        f"  Discovery T2D mean: {get_stat('Discovery', 'T2D', 'mean'):.5f}",
        f"  External ND mean:   {get_stat('External', 'ND', 'mean'):.5f}",
        f"  External T2D mean:  {get_stat('External', 'T2D', 'mean'):.5f}",
        "",
        "Mean shift (External - Discovery):",
        f"  ALL: {float(all_shift.iloc[0]) if len(all_shift) else math.nan:.5f}",
        f"  ND:  {float(nd_shift.iloc[0]) if len(nd_shift) else math.nan:.5f}",
        f"  T2D: {float(t2d_shift.iloc[0]) if len(t2d_shift) else math.nan:.5f}",
        "",
        "Threshold behavior externally:",
        f"  External ND fraction above threshold:  {float(ext_nd_above.iloc[0]) if len(ext_nd_above) else math.nan:.3f}",
        f"  External T2D fraction above threshold: {float(ext_t2d_above.iloc[0]) if len(ext_t2d_above) else math.nan:.3f}",
        "",
        "Interpretation:",
        "  If External ND and External T2D are both mostly above the frozen threshold,",
        "  the main issue is calibration/location shift rather than loss of rank-based separation.",
        "  Compare the boxplot, ECDF, and threshold-fraction plots together.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distribution-shift analysis for reduced-score external validation.")
    p.add_argument("--discovery-scores", required=True, type=Path)
    p.add_argument("--external-scores", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--metrics-summary", type=Path, default=None)
    p.add_argument("--score-col", default=None)
    p.add_argument("--group-col", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    safe_mkdir(args.outdir)

    discovery = load_scores(args.discovery_scores, "Discovery", args.score_col, args.group_col)
    external = load_scores(args.external_scores, "External", args.score_col, args.group_col)
    df = pd.concat([discovery, external], ignore_index=True)

    threshold = args.threshold
    if threshold is None and args.metrics_summary is not None:
        threshold = parse_threshold_from_metrics_summary(args.metrics_summary)
    if threshold is None:
        raise ValueError("Provide --threshold explicitly or supply --metrics-summary containing 'Frozen threshold'.")

    summary_df = summarize_scores(df)
    shift_df = compare_distributions(discovery, external)
    thr_df = threshold_position(df, threshold)
    confusion_df = compute_confusion_metrics(df, threshold)

    summary_df.to_csv(args.outdir / "summary_by_cohort_group.csv", index=False)
    shift_df.to_csv(args.outdir / "shift_statistics.csv", index=False)
    thr_df.to_csv(args.outdir / "threshold_position.csv", index=False)
    confusion_df.to_csv(args.outdir / "threshold_confusion_metrics.csv", index=False)

    plot_hist_density(df, threshold, args.outdir / "plot_hist_density.png")
    plot_boxplot(df, threshold, args.outdir / "plot_boxplot.png")
    plot_ecdf(df, threshold, args.outdir / "plot_ecdf.png")
    plot_threshold_fraction(thr_df, args.outdir / "plot_threshold_fraction.png")

    report = build_report(summary_df, shift_df, thr_df, confusion_df, threshold)
    (args.outdir / "distribution_shift_report.txt").write_text(report, encoding="utf-8")

    report_json = {
        "threshold": float(threshold),
        "summary_by_cohort_group": summary_df.to_dict(orient="records"),
        "shift_statistics": shift_df.to_dict(orient="records"),
        "threshold_position": thr_df.to_dict(orient="records"),
        "threshold_confusion_metrics": confusion_df.to_dict(orient="records"),
    }
    with open(args.outdir / "distribution_shift_report.json", "w", encoding="utf-8") as fh:
        json.dump(report_json, fh, indent=2)

    print(report)
    print("Output files written to:", args.outdir)


if __name__ == "__main__":
    main()
