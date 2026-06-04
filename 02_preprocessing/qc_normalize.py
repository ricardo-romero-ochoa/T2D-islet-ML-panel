#!/usr/bin/env python3
"""
02_preprocessing/qc_normalize.py

Per-dataset QC and normalization:
  1. Load raw expression CSV
  2. Log2(x+1) transform if data is not already log-scaled
  3. Remove low-detection probes (< MIN_DETECTION_FRACTION)
  4. Optionally apply quantile normalization (disabled for RNA-seq — see config)
  5. Map probe IDs → gene symbols via SOFT file (if available)
  6. Save normalized expression CSV + QC report + PCA plot

NOTE: QUANTILE_NORMALIZE is set to False in config.py for RNA-seq data.
Quantile normalization equalises sample distributions and destroys the
between-group expression differences required for classification.
"""

import os, sys, logging, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_RAW, DATA_PROCESSED, FIGURES_DIR, LOGS_DIR,
                    LOG2_TRANSFORM, QUANTILE_NORMALIZE, MIN_DETECTION_FRACTION,
                    DISCOVERY_DATASETS, VALIDATION_DATASETS,
                    PALETTE_T2D, PALETTE_CTRL, FIGURE_DPI)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "02_qc.log")), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

def is_log_scale(df): return df.values.max() < 100

def log2_transform(df): return np.log2(df.clip(lower=0) + 1)

def filter_low_detection(df, frac):
    mask = (df > 0).mean(axis=1) >= frac
    log.info(f"  Probe filter: {mask.sum()} / {len(mask)} pass")
    return df[mask]

def quantile_normalize(df):
    sorted_df = np.sort(df.values, axis=0)
    row_means  = sorted_df.mean(axis=1)
    rank_matrix = df.rank(method="min").astype(int) - 1
    return rank_matrix.map(lambda r: row_means[r] if r < len(row_means) else np.nan)

def load_gene_map(soft_file):
    probe_map = {}
    if not os.path.exists(soft_file): return probe_map
    in_platform = False; cols = {}
    with open(soft_file, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("^PLATFORM"): in_platform = True; continue
            if not in_platform: continue
            if line.startswith("!platform_table_begin"): cols = {}; continue
            if line.startswith("!platform_table_end"): break
            if line.startswith("#"): continue
            parts = line.split("\t")
            if not cols:
                cols = {n.upper(): i for i, n in enumerate(parts)}; continue
            probe_id = parts[0] if parts else None
            for gc in ["GENE_SYMBOL","SYMBOL","GENE","GENE_NAME"]:
                if gc in cols and cols[gc] < len(parts):
                    gene = parts[cols[gc]].strip()
                    if gene and gene not in ("---","NA",""):
                        probe_map[probe_id] = gene; break
    log.info(f"  Probe map: {len(probe_map)} entries")
    return probe_map

def collapse_probes(df, probe_map):
    if not probe_map: return df
    df = df.copy(); df["gene"] = df.index.map(probe_map)
    df = df.dropna(subset=["gene"])
    df["mean"] = df.drop(columns=["gene"]).mean(axis=1)
    df = df.sort_values("mean", ascending=False).drop_duplicates("gene").set_index("gene").drop(columns=["mean"])
    log.info(f"  After collapse: {len(df)} genes")
    return df

def make_pca(df, labels, gse_id, out_dir):
    X = df.T.fillna(0).values
    if X.shape[0] < 3: return
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    for lbl, color, name in [(1, PALETTE_T2D, "T2D"), (0, PALETTE_CTRL, "Control")]:
        idx = [i for i, c in enumerate(df.columns) if labels.get(c) == lbl]
        ax.scatter(coords[idx, 0], coords[idx, 1], c=color, label=name, s=55, alpha=0.8, edgecolors="white", lw=0.4)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"{gse_id} — PCA after normalization"); ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{gse_id}_qc_pca.png"), dpi=FIGURE_DPI)
    plt.close()

def process_dataset(gse_id):
    expr_path   = os.path.join(DATA_RAW, gse_id, f"{gse_id}_expression.csv")
    labels_path = os.path.join(DATA_PROCESSED, f"{gse_id}_labels.csv")
    if not os.path.exists(expr_path):
        log.warning(f"  {gse_id}: expression CSV not found — skipping"); return None

    df = pd.read_csv(expr_path, index_col=0).apply(pd.to_numeric, errors="coerce").dropna(how="all")
    log.info(f"  Loaded: {df.shape[0]} probes × {df.shape[1]} samples")

    labels = {}
    if os.path.exists(labels_path):
        ldf = pd.read_csv(labels_path)
        labels = dict(zip(ldf["sample_id"], ldf["label"]))

    if LOG2_TRANSFORM and not is_log_scale(df):
        df = log2_transform(df); log.info("  Applied log2(x+1)")
    else:
        log.info("  Data appears log-scaled — no log2 transform")

    df = filter_low_detection(df, MIN_DETECTION_FRACTION)

    if QUANTILE_NORMALIZE:
        df = quantile_normalize(df); log.info("  Applied quantile normalization")
    else:
        log.info("  Quantile normalization skipped (QUANTILE_NORMALIZE=False)")

    soft = os.path.join(DATA_RAW, gse_id, f"{gse_id}_family.soft")
    pm = load_gene_map(soft)
    if pm: df = collapse_probes(df, pm)

    out = os.path.join(DATA_PROCESSED, f"{gse_id}_expr_normalized.csv")
    df.to_csv(out)
    log.info(f"  Saved: {out}  shape={df.shape}")

    try: make_pca(df, labels, gse_id, FIGURES_DIR)
    except Exception as e: log.warning(f"  PCA plot failed: {e}")

    with open(os.path.join(DATA_PROCESSED, f"{gse_id}_qc_report.txt"), "w") as r:
        t2d_n = sum(1 for v in labels.values() if v == 1)
        ctrl_n = sum(1 for v in labels.values() if v == 0)
        r.write(f"QC Report: {gse_id}\nGenes: {df.shape[0]}\nSamples: {df.shape[1]}\n"
                f"T2D: {t2d_n}\nControl: {ctrl_n}\nMissing values: {df.isna().sum().sum()}\n")
    return df

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DISCOVERY_DATASETS + VALIDATION_DATASETS)
    args = ap.parse_args()
    for gse_id in args.datasets:
        log.info(f"\n{'─'*50}\nQC: {gse_id}")
        process_dataset(gse_id)
    log.info("\nQC/normalization complete.")

if __name__ == "__main__":
    main()
