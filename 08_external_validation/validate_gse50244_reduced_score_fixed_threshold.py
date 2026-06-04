#!/usr/bin/env python3
"""
External validation of the overlap-restricted reduced score in GSE50244,
with the classification threshold fixed from the discovery cohort.

Primary purpose
---------------
Recompute the external reduced-score evaluation so that the decision threshold
is determined using the discovery cohort only, then applied unchanged to GSE50244.

Discovery cohort assumptions
----------------------------
- Expression file is a matrix with first column = gene identifier.
- Your current discovery file uses Ensembl gene IDs and sample columns.
- Metadata file uses numeric labels:
      1  -> T2D
      0  -> ND
     -1  -> exclude (IGT, IFG, T3cD, unresolved)

Main outputs
------------
- overlap_table_used.csv
- discovery_reduced_scores.csv
- external_reduced_scores.csv
- metrics_summary.json
- metrics_summary.txt
- roc_discovery_reduced_score.png
- roc_external_reduced_score.png
- boxplot_discovery_reduced_score.png
- boxplot_external_reduced_score.png

Recommended use
---------------
Use the threshold method 'train_youden' as the primary externally portable threshold,
because the threshold is defined on exactly the same score scale later used externally.

Optional sensitivity analysis
-----------------------------
You may also run with --threshold-method loocv_youden, which derives the threshold
from discovery out-of-fold scores. This is stricter internally, but the threshold is
not guaranteed to be on exactly the same score scale as the full-discovery score used
externally, so it is best treated as a sensitivity analysis rather than the primary result.

Examples
--------
python validate_gse50244_reduced_score_fixed_threshold.py \
  --discovery-expr GSE164416_expr_normalized.csv \
  --discovery-meta GSE164416_labels.csv \
  --overlap-table gse50244_external_results/overlap_table.csv \
  --outdir gse50244_fixed_threshold_results \
  --sample-col sample_id \
  --group-col label \
  --discovery-transform none \
  --threshold-method train_youden

If you do not have overlap_table.csv from the previous run, the script can try to
resolve discovery Ensembl IDs automatically using Ensembl REST.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


PANEL_UP = ["DKK3", "PRIMA1", "TAFA4", "HHATL", "PARVG", "ENSG00000284653"]
PANEL_DOWN = ["GABRA2", "SLC2A2", "ARG2", "RNU1-70P"]
FULL_PANEL = PANEL_UP + PANEL_DOWN

GENE_ALIASES = {
    "TAFA4": ["TAFA4", "FAM19A4"],
    "GABRA2": ["GABRA2"],
    "SLC2A2": ["SLC2A2", "GLUT2"],
    "ARG2": ["ARG2"],
    "DKK3": ["DKK3"],
    "PRIMA1": ["PRIMA1"],
    "HHATL": ["HHATL"],
    "PARVG": ["PARVG"],
    "RNU1-70P": ["RNU1-70P"],
    "ENSG00000284653": ["ENSG00000284653"],
}

GSE50244_URLS = {
    "series_matrix": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE50nnn/GSE50244/matrix/"
        "GSE50244_series_matrix.txt.gz"
    ),
    "genes_expr": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE50nnn/GSE50244/suppl/"
        "GSE50244_Genes_counts_TMM_NormLength_atLeastMAF5_expressed.txt.gz"
    ),
}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_table_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", compression="infer")


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        eprint(f"[info] Using cached file: {dest}")
        return
    eprint(f"[info] Downloading: {url}")
    urllib.request.urlretrieve(url, dest)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {dest}")


def clean_token(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", "", str(x).strip().strip('"').strip("'")).lower()


def maybe_strip_version(x: str) -> str:
    return re.sub(r"\.\d+$", "", str(x))


def extract_hba1c(text) -> Optional[float]:
    """
    Parse strings like 'hba1c: 6.7' without mistakenly capturing the '1' in 'hba1c'.
    """
    if pd.isna(text):
        return None
    s = str(text).strip()
    m = re.search(r"hba1c\s*:?\s*(-?\d+(?:\.\d+)?)", s, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    # fallback: use the last numeric token in the string
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def cohen_d(nd: np.ndarray, t2d: np.ndarray) -> float:
    nd = np.asarray(nd, dtype=float)
    t2d = np.asarray(t2d, dtype=float)
    nd = nd[np.isfinite(nd)]
    t2d = t2d[np.isfinite(t2d)]
    if len(nd) < 2 or len(t2d) < 2:
        return np.nan
    n1, n2 = len(nd), len(t2d)
    v1, v2 = np.var(nd, ddof=1), np.var(t2d, ddof=1)
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return (np.mean(t2d) - np.mean(nd)) / np.sqrt(pooled)


def infer_transform(df: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, str]:
    vals = pd.to_numeric(df.stack(), errors="coerce").dropna().values
    if len(vals) == 0:
        raise ValueError("No numeric expression values found.")
    if mode == "none":
        return df, "none"
    if mode == "log2":
        return np.log2(df.clip(lower=0) + 1.0), "log2"
    q99 = float(np.quantile(vals, 0.99))
    vmin = float(np.min(vals))
    if vmin >= 0 and q99 > 50:
        return np.log2(df.clip(lower=0) + 1.0), "log2(auto)"
    return df, "none(auto)"


def collapse_duplicate_genes(expr: pd.DataFrame) -> pd.DataFrame:
    expr = expr.copy()
    expr.index = [maybe_strip_version(g) for g in expr.index]
    expr = expr.groupby(expr.index).mean(numeric_only=True)
    return expr


def detect_orientation(expr: pd.DataFrame, sample_ids: Sequence[str]) -> pd.DataFrame:
    sample_ids_clean = {clean_token(x) for x in sample_ids}
    col_matches = sum(clean_token(c) in sample_ids_clean for c in expr.columns)
    row_matches = sum(clean_token(r) in sample_ids_clean for r in expr.index)
    return expr.T if row_matches > col_matches else expr


def label_to_group(x) -> Optional[str]:
    if pd.isna(x):
        return None
    try:
        v = int(float(x))
        if v == 1:
            return "T2D"
        if v == 0:
            return "ND"
        return None
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"nd", "control", "non-diabetic", "nondiabetic", "non diabetic"}:
        return "ND"
    if s in {"t2d", "type 2 diabetes", "type2diabetes", "diabetic"}:
        return "T2D"
    return None


def parse_geo_series_matrix(series_matrix_gz: Path) -> pd.DataFrame:
    rows = {}
    with gzip.open(series_matrix_gz, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            parsed = next(csv.reader([line.rstrip("\n")], delimiter="\t", quotechar='"'))
            rows[parsed[0]] = parsed[1:]

    meta = pd.DataFrame({"geo_accession": rows["!Sample_geo_accession"]})
    meta["title"] = rows.get("!Sample_title", rows["!Sample_geo_accession"])
    if "!Sample_source_name_ch1" in rows:
        meta["source_name"] = rows["!Sample_source_name_ch1"]

    char_keys = sorted(k for k in rows if k.startswith("!Sample_characteristics_ch1"))
    used = set()

    def unique_name(name: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
        base = base or "characteristic"
        cur = base
        i = 2
        while cur in used:
            cur = f"{base}_{i}"
            i += 1
        used.add(cur)
        return cur

    for i, key in enumerate(char_keys, start=1):
        vals = rows[key]
        rawcol = f"characteristics_{i}"
        meta[rawcol] = vals

        prefixes, suffixes = [], []
        ok = True
        for v in vals:
            if ":" not in str(v):
                ok = False
                break
            p, s = str(v).split(":", 1)
            prefixes.append(p.strip())
            suffixes.append(s.strip())
        if ok and len(set(p.lower() for p in prefixes)) == 1:
            meta[unique_name(prefixes[0])] = suffixes

    return meta


def infer_groups_from_metadata(meta: pd.DataFrame, nd_max_hba1c: float, t2d_min_hba1c: float) -> pd.DataFrame:
    meta = meta.copy()
    explicit = []
    hba1c_values = []

    for _, row in meta.iterrows():
        text = " | ".join(str(v) for v in row.values if pd.notna(v)).lower()
        if "type 2 diabetes" in text or "type 2 diabetic" in text or re.search(r"\bt2d\b", text):
            explicit.append("T2D")
        elif "non-diabetic" in text or "nondiabetic" in text or "non diabetic" in text or "normoglycemic" in text:
            explicit.append("ND")
        else:
            explicit.append(None)

        h = None
        # first try named hba1c column
        for c in meta.columns:
            if "hba1c" in c.lower() and pd.notna(row.get(c, np.nan)):
                h = extract_hba1c(row[c])
                if h is not None:
                    break
        # fallback: scan characteristics columns
        if h is None:
            for c in meta.columns:
                if c.startswith("characteristics_") and pd.notna(row.get(c, np.nan)):
                    txt = str(row[c])
                    if "hba1c" in txt.lower():
                        h = extract_hba1c(txt)
                        if h is not None:
                            break
        hba1c_values.append(h)

    meta["group_explicit"] = explicit
    meta["hba1c_inferred"] = hba1c_values

    groups, reasons = [], []
    for _, row in meta.iterrows():
        if row["group_explicit"] in {"ND", "T2D"}:
            groups.append(row["group_explicit"])
            reasons.append("explicit_text")
            continue
        h = row["hba1c_inferred"]
        if pd.notna(h):
            if float(h) < nd_max_hba1c:
                groups.append("ND")
                reasons.append(f"hba1c<{nd_max_hba1c}")
            elif float(h) > t2d_min_hba1c:
                groups.append("T2D")
                reasons.append(f"hba1c>{t2d_min_hba1c}")
            else:
                groups.append("INTERMEDIATE")
                reasons.append("intermediate_hba1c")
        else:
            groups.append("UNKNOWN")
            reasons.append("no_label")

    meta["group"] = groups
    meta["group_reason"] = reasons
    return meta


def load_expression_generic(path: Path) -> pd.DataFrame:
    df = read_table_auto(path)
    if df.shape[1] < 2:
        raise ValueError(f"Too few columns in expression file: {path}")
    df = df.set_index(df.columns[0])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all")
    return df


def load_discovery_inputs(
    expr_path: Path,
    meta_path: Path,
    sample_col: str,
    group_col: str,
    transform_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    meta = read_table_auto(meta_path).copy()
    if sample_col not in meta.columns:
        raise ValueError(f"Missing sample column in discovery metadata: {sample_col}")
    if group_col not in meta.columns:
        raise ValueError(f"Missing group column in discovery metadata: {group_col}")

    meta = meta[[sample_col, group_col]].copy()
    meta.columns = ["sample_id", "group_raw"]
    meta["group"] = meta["group_raw"].map(label_to_group)
    meta = meta[meta["group"].isin(["ND", "T2D"])].copy()

    if meta.empty:
        raise ValueError("No ND/T2D discovery samples remained after filtering labels.")

    expr = load_expression_generic(expr_path)
    expr = detect_orientation(expr, meta["sample_id"].tolist())

    sample_map = {clean_token(c): c for c in expr.columns}
    keep, expr_cols = [], []
    for s in meta["sample_id"]:
        tok = clean_token(s)
        if tok in sample_map:
            keep.append(s)
            expr_cols.append(sample_map[tok])

    meta = meta[meta["sample_id"].isin(keep)].copy()
    meta = meta.set_index("sample_id").loc[keep].reset_index()

    expr = expr[expr_cols].copy()
    expr.columns = keep
    expr = collapse_duplicate_genes(expr)
    expr, transform_used = infer_transform(expr, transform_mode)

    return expr, meta, transform_used


def match_external_columns_to_metadata(expr_cols: Sequence[str], meta: pd.DataFrame) -> Tuple[List[str], pd.DataFrame, str]:
    expr_cols_list = list(expr_cols)
    candidates = []
    for meta_col in ["geo_accession", "title"]:
        if meta_col not in meta.columns:
            continue
        mp = {clean_token(v): i for i, v in meta[meta_col].items() if clean_token(v)}
        matches = [mp.get(clean_token(c), None) for c in expr_cols_list]
        candidates.append((sum(m is not None for m in matches), meta_col, matches))

    candidates.sort(reverse=True, key=lambda x: x[0])
    if not candidates or candidates[0][0] == 0:
        raise RuntimeError("Could not match GSE50244 expression columns to metadata.")

    _, best_col, best_matches = candidates[0]
    matched_expr_cols, matched_meta_rows = [], []
    for c, midx in zip(expr_cols_list, best_matches):
        if midx is not None:
            matched_expr_cols.append(c)
            matched_meta_rows.append(midx)

    out_meta = meta.loc[matched_meta_rows].copy().reset_index(drop=True)
    out_meta["matched_expr_col"] = matched_expr_cols
    return matched_expr_cols, out_meta, best_col


def load_gse50244_external(
    workdir: Path,
    external_meta_override: Optional[Path],
    nd_max_hba1c: float,
    t2d_min_hba1c: float,
    transform_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    safe_mkdir(workdir)
    series_path = workdir / "GSE50244_series_matrix.txt.gz"
    expr_path = workdir / "GSE50244_Genes_counts_TMM_NormLength_atLeastMAF5_expressed.txt.gz"

    download_file(GSE50244_URLS["series_matrix"], series_path)
    download_file(GSE50244_URLS["genes_expr"], expr_path)

    meta = parse_geo_series_matrix(series_path)

    if external_meta_override is not None:
        override = read_table_auto(external_meta_override).copy()
        if not {"sample_name", "group"}.issubset(override.columns):
            raise ValueError("Override file must contain columns: sample_name, group")
        override["group"] = override["group"].map(label_to_group)
        mp = {clean_token(r["sample_name"]): r["group"] for _, r in override.iterrows()}
        meta["group"] = meta["title"].map(lambda x: mp.get(clean_token(x), None))
        meta["group_reason"] = np.where(meta["group"].notna(), "override_file", "unlabeled")
        meta["hba1c_inferred"] = np.nan
    else:
        meta = infer_groups_from_metadata(meta, nd_max_hba1c, t2d_min_hba1c)

    raw_expr = read_table_auto(expr_path)
    gene_col = raw_expr.columns[0]
    expr_cols = list(raw_expr.columns[1:])

    matched_cols, meta_matched, match_mode = match_external_columns_to_metadata(expr_cols, meta)
    meta_matched = meta_matched[meta_matched["group"].isin(["ND", "T2D"])].copy().reset_index(drop=True)
    matched_cols = list(meta_matched["matched_expr_col"])

    expr = raw_expr[[gene_col] + matched_cols].copy()
    expr = expr.set_index(gene_col)
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=0, how="all")
    expr.columns = meta_matched["title"].astype(str).tolist()

    expr = collapse_duplicate_genes(expr)
    expr, transform_used = infer_transform(expr, transform_mode)
    return expr, meta_matched, transform_used, match_mode


def ensembl_xrefs_for_symbol(symbol: str, species: str = "homo_sapiens") -> List[str]:
    url = (
        f"https://rest.ensembl.org/xrefs/symbol/{species}/"
        f"{urllib.parse.quote(symbol)}?content-type=application/json"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    ids = []
    for item in data:
        _id = item.get("id", "")
        obj_type = str(item.get("type", "")).lower()
        if _id.startswith("ENSG") and ("gene" in obj_type or obj_type == ""):
            ids.append(maybe_strip_version(_id))

    seen, out = set(), []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_symbol_to_discovery_ensembl(discovery_index: Sequence[str]) -> Dict[str, List[str]]:
    discovery_ids = {maybe_strip_version(x) for x in discovery_index}
    mapping = {}
    for gene in FULL_PANEL:
        candidates = []
        if gene.startswith("ENSG"):
            g0 = maybe_strip_version(gene)
            if g0 in discovery_ids:
                candidates.append(g0)
        for alias in GENE_ALIASES.get(gene, [gene]):
            for ens in ensembl_xrefs_for_symbol(alias):
                ens = maybe_strip_version(ens)
                if ens in discovery_ids:
                    candidates.append(ens)
        seen, uniq = set(), []
        for x in candidates:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        mapping[gene] = uniq
    return mapping


def resolve_in_external_index(target_gene: str, external_index: Sequence[str]) -> Optional[str]:
    ext = [str(x) for x in external_index]
    exact = {x: x for x in ext}
    lower = {x.lower(): x for x in ext}
    stripped = {maybe_strip_version(x).lower(): x for x in ext}
    for probe in GENE_ALIASES.get(target_gene, [target_gene]):
        if probe in exact:
            return exact[probe]
        if probe.lower() in lower:
            return lower[probe.lower()]
        if maybe_strip_version(probe).lower() in stripped:
            return stripped[maybe_strip_version(probe).lower()]
    return None


def build_overlap_table_auto(discovery_expr: pd.DataFrame, external_expr: pd.DataFrame) -> pd.DataFrame:
    discovery_map = build_symbol_to_discovery_ensembl(discovery_expr.index)
    rows = []
    for gene in FULL_PANEL:
        d_candidates = discovery_map.get(gene, [])
        d_match = d_candidates[0] if d_candidates else None
        e_match = resolve_in_external_index(gene, external_expr.index)
        rows.append({
            "panel_gene": gene,
            "direction_in_discovery": "up" if gene in PANEL_UP else "down",
            "discovery_match": d_match,
            "external_match": e_match,
            "usable_for_transport": pd.notna(d_match) and pd.notna(e_match),
            "all_discovery_candidates": ";".join(d_candidates) if d_candidates else "",
        })
    return pd.DataFrame(rows)


def load_or_build_overlap(
    overlap_table_path: Optional[Path],
    discovery_expr: pd.DataFrame,
    external_expr: pd.DataFrame,
) -> pd.DataFrame:
    if overlap_table_path is not None and overlap_table_path.exists():
        tab = read_table_auto(overlap_table_path).copy()
        required = {"panel_gene", "discovery_match", "external_match", "usable_for_transport"}
        if not required.issubset(tab.columns):
            raise ValueError(
                "Provided overlap table is missing required columns: "
                + ", ".join(sorted(required - set(tab.columns)))
            )
        return tab
    eprint("[info] No overlap table provided; resolving discovery Ensembl IDs automatically via Ensembl REST.")
    return build_overlap_table_auto(discovery_expr, external_expr)


def build_feature_matrices(
    overlap_table: pd.DataFrame,
    discovery_expr: pd.DataFrame,
    discovery_meta: pd.DataFrame,
    external_expr: pd.DataFrame,
    external_meta: pd.DataFrame,
    min_overlap: int,
    include_genes: Optional[List[str]] = None,
):
    usable = overlap_table[overlap_table["usable_for_transport"] == True].copy()
    if include_genes is not None and len(include_genes) > 0:
        usable = usable[usable["panel_gene"].isin(include_genes)].copy()
    feature_genes = usable["panel_gene"].tolist()

    if len(feature_genes) < min_overlap:
        raise RuntimeError(
            f"Only {len(feature_genes)} overlapping panel genes found; minimum required is {min_overlap}."
        )

    discovery_rows, external_rows = [], []
    for g in feature_genes:
        row = usable[usable["panel_gene"] == g].iloc[0]
        discovery_rows.append(row["discovery_match"])
        external_rows.append(row["external_match"])

    X_train = discovery_expr.loc[discovery_rows, discovery_meta["sample_id"]].copy().T
    X_train.columns = feature_genes

    X_ext = external_expr.loc[external_rows, external_meta["title"]].copy().T
    X_ext.columns = feature_genes

    y_train = (discovery_meta["group"].values == "T2D").astype(int)
    y_ext = (external_meta["group"].values == "T2D").astype(int)
    return X_train, X_ext, feature_genes, y_train, y_ext


def fit_score_scaler(X_train: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame({
        "gene": X_train.columns,
        "mean": [float(X_train[g].mean()) for g in X_train.columns],
        "sd": [float(X_train[g].std(ddof=1)) for g in X_train.columns],
    }).set_index("gene")
    stats["sd"] = stats["sd"].replace(0, np.nan)
    stats["sd"] = stats["sd"].fillna(1.0)
    return stats


def apply_score_scaler(X: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    Z = X.copy()
    for g in Z.columns:
        Z[g] = (Z[g] - float(stats.loc[g, "mean"])) / float(stats.loc[g, "sd"])
    return Z


def compute_reduced_score_from_z(Z: pd.DataFrame) -> pd.Series:
    up = [g for g in Z.columns if g in PANEL_UP]
    down = [g for g in Z.columns if g in PANEL_DOWN]
    if len(up) == 0 or len(down) == 0:
        raise RuntimeError("Reduced score requires at least one up gene and one down gene.")
    return Z[up].mean(axis=1) - Z[down].mean(axis=1)


def compute_youden_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, score)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def reduced_score_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    auc_val = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) >= 2 else np.nan
    bal = float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) >= 2 else np.nan
    mcc = float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) >= 2 else np.nan
    return {
        "AUC": auc_val,
        "Threshold": float(threshold),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "BalancedAccuracy": bal,
        "MCC": mcc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def discovery_oof_scores_loocv(X_train: pd.DataFrame, y_train: np.ndarray) -> np.ndarray:
    n = X_train.shape[0]
    scores = np.full(n, np.nan, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr = X_train.iloc[mask, :]
        X_te = X_train.iloc[[i], :]
        stats = fit_score_scaler(X_tr)
        Z_te = apply_score_scaler(X_te, stats)
        scores[i] = float(compute_reduced_score_from_z(Z_te).iloc[0])
    return scores


def gene_level_replication(X_ext: pd.DataFrame, y_ext: np.ndarray, feature_genes: List[str]) -> pd.DataFrame:
    rows = []
    nd_mask = y_ext == 0
    t2d_mask = y_ext == 1
    for g in feature_genes:
        nd = X_ext.loc[nd_mask, g].values.astype(float)
        t2d = X_ext.loc[t2d_mask, g].values.astype(float)
        diff = float(np.mean(t2d) - np.mean(nd))
        ext_dir = "up" if diff > 0 else "down"
        disc_dir = "up" if g in PANEL_UP else "down"
        rows.append({
            "gene": g,
            "discovery_direction": disc_dir,
            "external_direction": ext_dir,
            "direction_concordant": disc_dir == ext_dir,
            "external_mean_ND": float(np.mean(nd)),
            "external_mean_T2D": float(np.mean(t2d)),
            "external_mean_diff_T2D_minus_ND": diff,
            "external_cohen_d": float(cohen_d(nd, t2d)),
        })
    return pd.DataFrame(rows)


def plot_roc(y_true: np.ndarray, score: np.ndarray, title: str, outpath: Path) -> None:
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, score)
    auc_val = roc_auc_score(y_true, score)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_box(values: np.ndarray, labels: Sequence[str], title: str, ylabel: str, outpath: Path) -> None:
    labels = np.asarray(labels)
    nd = values[labels == "ND"]
    t2d = values[labels == "T2D"]
    plt.figure(figsize=(5.5, 5.5))
    plt.boxplot([nd, t2d], tick_labels=["ND", "T2D"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--discovery-expr", required=True, type=Path)
    p.add_argument("--discovery-meta", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)

    p.add_argument("--sample-col", default="sample_id")
    p.add_argument("--group-col", default="label")

    p.add_argument("--overlap-table", type=Path, default=None,
                   help="Optional overlap_table.csv from the previous external-validation run.")
    p.add_argument("--external-meta-override", type=Path, default=None,
                   help="Optional CSV/TSV with columns: sample_name, group")

    p.add_argument("--discovery-transform", choices=["auto", "none", "log2"], default="none")
    p.add_argument("--external-transform", choices=["auto", "none", "log2"], default="auto")

    p.add_argument("--nd-max-hba1c", type=float, default=6.0)
    p.add_argument("--t2d-min-hba1c", type=float, default=6.5)
    p.add_argument("--min-overlap", type=int, default=6)

    p.add_argument("--threshold-method", choices=["train_youden", "loocv_youden"], default="train_youden")
    p.add_argument("--include-genes", type=str, default="",
                   help="Optional comma-separated subset of genes to include from the overlap table.")

    return p.parse_args()


def main():
    args = parse_args()
    safe_mkdir(args.outdir)
    cache_dir = args.outdir / "geo_cache"
    safe_mkdir(cache_dir)

    include_genes = [g.strip() for g in args.include_genes.split(",") if g.strip()] if args.include_genes else None

    eprint("[info] Loading discovery cohort...")
    disc_expr, disc_meta, disc_transform = load_discovery_inputs(
        expr_path=args.discovery_expr,
        meta_path=args.discovery_meta,
        sample_col=args.sample_col,
        group_col=args.group_col,
        transform_mode=args.discovery_transform,
    )
    y_train = (disc_meta["group"].values == "T2D").astype(int)
    eprint(f"[info] Discovery samples retained: {len(disc_meta)}")
    eprint(f"[info]   ND: {(y_train == 0).sum()}")
    eprint(f"[info]   T2D: {(y_train == 1).sum()}")
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Discovery cohort has only one class after filtering.")

    eprint("[info] Loading GSE50244...")
    ext_expr, ext_meta, ext_transform, match_mode = load_gse50244_external(
        workdir=cache_dir,
        external_meta_override=args.external_meta_override,
        nd_max_hba1c=args.nd_max_hba1c,
        t2d_min_hba1c=args.t2d_min_hba1c,
        transform_mode=args.external_transform,
    )
    ext_meta.to_csv(args.outdir / "GSE50244_parsed_metadata.csv", index=False)

    eprint("[info] External group counts by reason:")
    if "group_reason" in ext_meta.columns:
        eprint(ext_meta.groupby(["group", "group_reason"]).size().to_string())
    else:
        eprint(ext_meta["group"].value_counts(dropna=False).to_string())
    if "hba1c_inferred" in ext_meta.columns:
        eprint("[info] HbA1c summary in matched external samples:")
        eprint(ext_meta["hba1c_inferred"].describe().to_string())

    eprint("[info] Resolving overlap...")
    overlap = load_or_build_overlap(args.overlap_table, disc_expr, ext_expr)
    overlap.to_csv(args.outdir / "overlap_table_used.csv", index=False)

    X_train, X_ext, feature_genes, y_train, y_ext = build_feature_matrices(
        overlap_table=overlap,
        discovery_expr=disc_expr,
        discovery_meta=disc_meta,
        external_expr=ext_expr,
        external_meta=ext_meta,
        min_overlap=args.min_overlap,
        include_genes=include_genes,
    )
    eprint(f"[info] Overlap genes used: {feature_genes}")
    eprint(f"[info] External class counts after filtering: ND={(y_ext == 0).sum()}, T2D={(y_ext == 1).sum()}")
    if len(np.unique(y_ext)) < 2:
        raise RuntimeError(
            "External cohort has only one class after metadata matching and label filtering. "
            "Inspect GSE50244_parsed_metadata.csv and the HbA1c thresholds / sample matching."
        )

    # Final discovery score scale used for external deployment
    train_stats = fit_score_scaler(X_train)
    Z_train = apply_score_scaler(X_train, train_stats)
    Z_ext = apply_score_scaler(X_ext, train_stats)

    discovery_score = compute_reduced_score_from_z(Z_train)
    external_score = compute_reduced_score_from_z(Z_ext)

    # Threshold fixed from discovery cohort only
    if args.threshold_method == "train_youden":
        frozen_threshold = compute_youden_threshold(y_train, discovery_score.values)
        threshold_origin = "Discovery full-cohort reduced scores (Youden)"
        discovery_threshold_source_score = discovery_score.values
    elif args.threshold_method == "loocv_youden":
        oof_scores = discovery_oof_scores_loocv(X_train, y_train)
        frozen_threshold = compute_youden_threshold(y_train, oof_scores)
        threshold_origin = "Discovery LOOCV out-of-fold reduced scores (Youden)"
        discovery_threshold_source_score = oof_scores
    else:
        raise ValueError(f"Unsupported threshold method: {args.threshold_method}")

    discovery_metrics = reduced_score_metrics(y_train, discovery_score.values, frozen_threshold)
    external_metrics = reduced_score_metrics(y_ext, external_score.values, frozen_threshold)

    disc_scores_df = pd.DataFrame({
        "sample": X_train.index,
        "group": np.where(y_train == 1, "T2D", "ND"),
        "reduced_score": discovery_score.values,
        "predicted_label_at_frozen_threshold": np.where(discovery_score.values >= frozen_threshold, "T2D", "ND"),
    })
    if args.threshold_method == "loocv_youden":
        disc_scores_df["threshold_source_score_oof"] = discovery_threshold_source_score
    disc_scores_df.to_csv(args.outdir / "discovery_reduced_scores.csv", index=False)

    ext_scores_df = pd.DataFrame({
        "sample": X_ext.index,
        "group": np.where(y_ext == 1, "T2D", "ND"),
        "reduced_score": external_score.values,
        "predicted_label_at_frozen_threshold": np.where(external_score.values >= frozen_threshold, "T2D", "ND"),
    })
    ext_scores_df.to_csv(args.outdir / "external_reduced_scores.csv", index=False)

    rep = gene_level_replication(X_ext, y_ext, feature_genes)
    rep.to_csv(args.outdir / "external_gene_replication.csv", index=False)

    plot_roc(y_train, discovery_score.values, "Discovery reduced score ROC", args.outdir / "roc_discovery_reduced_score.png")
    plot_roc(y_ext, external_score.values, "GSE50244 reduced score ROC (frozen discovery threshold)", args.outdir / "roc_external_reduced_score.png")
    plot_box(discovery_score.values, np.where(y_train == 1, "T2D", "ND"),
             "Discovery reduced score", "Reduced score", args.outdir / "boxplot_discovery_reduced_score.png")
    plot_box(external_score.values, np.where(y_ext == 1, "T2D", "ND"),
             "GSE50244 reduced score", "Reduced score", args.outdir / "boxplot_external_reduced_score.png")

    summary = {
        "discovery_transform_used": disc_transform,
        "external_transform_used": ext_transform,
        "external_sample_match_mode": match_mode,
        "threshold_method": args.threshold_method,
        "threshold_origin": threshold_origin,
        "frozen_threshold": float(frozen_threshold),
        "n_discovery_samples": int(len(y_train)),
        "n_discovery_ND": int((y_train == 0).sum()),
        "n_discovery_T2D": int((y_train == 1).sum()),
        "n_external_samples": int(len(y_ext)),
        "n_external_ND": int((y_ext == 0).sum()),
        "n_external_T2D": int((y_ext == 1).sum()),
        "feature_overlap_genes": feature_genes,
        "n_feature_overlap": int(len(feature_genes)),
        "discovery_metrics_at_frozen_threshold": discovery_metrics,
        "external_metrics_at_frozen_threshold": external_metrics,
        "direction_concordance": {
            "n_concordant": int(rep["direction_concordant"].sum()),
            "n_total": int(len(rep)),
            "fraction_concordant": float(rep["direction_concordant"].mean()) if len(rep) else None,
        },
    }
    with open(args.outdir / "metrics_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    report = [
        "GSE50244 reduced-score external validation with frozen discovery threshold",
        "===================================================================",
        "",
        f"Discovery transform used: {disc_transform}",
        f"External transform used: {ext_transform}",
        f"External sample match mode: {match_mode}",
        "",
        f"Threshold method: {args.threshold_method}",
        f"Threshold origin: {threshold_origin}",
        f"Frozen threshold: {frozen_threshold:.5f}",
        "",
        f"Discovery samples: {len(y_train)}",
        f"  ND: {(y_train == 0).sum()}",
        f"  T2D: {(y_train == 1).sum()}",
        "",
        f"External labeled samples: {len(y_ext)}",
        f"  ND: {(y_ext == 0).sum()}",
        f"  T2D: {(y_ext == 1).sum()}",
        "",
        f"Feature overlap ({len(feature_genes)} genes): {', '.join(feature_genes)}",
        "",
        "Discovery reduced score at frozen threshold",
        f"  AUC: {discovery_metrics['AUC']:.3f}",
        f"  Sensitivity: {discovery_metrics['Sensitivity']:.3f}",
        f"  Specificity: {discovery_metrics['Specificity']:.3f}",
        f"  Balanced accuracy: {discovery_metrics['BalancedAccuracy']:.3f}",
        f"  MCC: {discovery_metrics['MCC']:.3f}",
        "",
        "External reduced score at the same frozen threshold",
        f"  AUC: {external_metrics['AUC']:.3f}",
        f"  Sensitivity: {external_metrics['Sensitivity']:.3f}",
        f"  Specificity: {external_metrics['Specificity']:.3f}",
        f"  Balanced accuracy: {external_metrics['BalancedAccuracy']:.3f}",
        f"  MCC: {external_metrics['MCC']:.3f}",
        "",
        f"Direction concordance: {int(rep['direction_concordant'].sum())}/{len(rep)}",
    ]
    with open(args.outdir / "metrics_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")

    print("\n".join(report))


if __name__ == "__main__":
    main()
