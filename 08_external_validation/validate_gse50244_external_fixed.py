#!/usr/bin/env python3
"""
External transportability validation of the T2D islet panel in GSE50244,
adapted for a discovery cohort where:

  label = 1   -> T2D
  label = 0   -> ND
  label < 0   -> exclude  (IGT, IFG, T3cD, unresolved)

Expected discovery inputs:
  - expression CSV with first column as gene identifier (e.g. Ensembl)
  - labels CSV with columns: sample_id, label

Example:
python validate_gse50244_external_fixed.py \
  --discovery-expr GSE164416_expr_normalized.csv \
  --discovery-meta GSE164416_labels.csv \
  --outdir gse50244_external_results \
  --sample-col sample_id \
  --group-col label \
  --discovery-transform none
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler


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


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        eprint(f"[info] Using cached file: {dest}")
        return
    eprint(f"[info] Downloading: {url}")
    urllib.request.urlretrieve(url, dest)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {dest}")


def read_table_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", compression="infer")


def clean_token(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", "", str(x).strip().strip('"').strip("'")).lower()


def maybe_strip_version(x: str) -> str:
    return re.sub(r"\.\d+$", "", str(x))


def extract_first_numeric(text) -> Optional[float]:
    """
    Robust numeric extractor for metadata strings like:
      'hba1c: 5.8'
      'HbA1c (%) : 7.1'
    Important: avoid grabbing the '1' from the word 'hba1c'.
    """
    if pd.isna(text):
        return None
    s = str(text).strip()

    # Prefer the content after the last colon if present
    if ":" in s:
        tail = s.rsplit(":", 1)[1]
        m = re.search(r"(-?\d+(?:\.\d+)?)", tail)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    # Otherwise take the last numeric token, not the first one
    nums = re.findall(r"(-?\d+(?:\.\d+)?)", s)
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
    if pooled <= 0:
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
    for _, row in meta.iterrows():
        text = " | ".join(str(v) for v in row.values if pd.notna(v)).lower()
        if "type 2 diabetes" in text or "type 2 diabetic" in text or re.search(r"\bt2d\b", text):
            explicit.append("T2D")
        elif "non-diabetic" in text or "nondiabetic" in text or "non diabetic" in text or "normoglycemic" in text:
            explicit.append("ND")
        else:
            explicit.append(None)
    meta["group_explicit"] = explicit

    hba1c_col = None
    for c in meta.columns:
        if "hba1c" in c.lower():
            hba1c_col = c
            break

    if hba1c_col is None:
        vals = []
        for _, row in meta.iterrows():
            found = None
            for c in meta.columns:
                if c.startswith("characteristics_") and "hba1c" in str(row[c]).lower():
                    found = extract_first_numeric(row[c])
                    break
            vals.append(found)
        meta["hba1c_inferred"] = vals
    else:
        meta["hba1c_inferred"] = meta[hba1c_col].map(extract_first_numeric)

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
        key = maybe_strip_version(probe).lower()
        if key in stripped:
            return stripped[key]
    return None


def build_overlap_table(discovery_expr: pd.DataFrame, external_expr: pd.DataFrame) -> pd.DataFrame:
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


def build_feature_matrices(
    overlap_table: pd.DataFrame,
    discovery_expr: pd.DataFrame,
    discovery_meta: pd.DataFrame,
    external_expr: pd.DataFrame,
    external_meta: pd.DataFrame,
    min_overlap: int,
):
    usable = overlap_table[overlap_table["usable_for_transport"]].copy()
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


def fit_locked_logistic(X_train: pd.DataFrame, y_train: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train.values)
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    clf.fit(Xs, y_train)
    return scaler, clf


def external_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan

    if len(np.unique(y_true)) < 2:
        auc_val = np.nan
        bal_acc = np.nan
        mcc = np.nan
    else:
        auc_val = float(roc_auc_score(y_true, score))
        bal_acc = float(balanced_accuracy_score(y_true, pred))
        mcc = float(matthews_corrcoef(y_true, pred))

    return {
        "AUC": auc_val,
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "BalancedAccuracy": bal_acc,
        "MCC": mcc,
        "Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def compute_reduced_score(X_ext: pd.DataFrame, feature_genes: List[str]) -> pd.Series:
    Z = X_ext.copy()
    for col in Z.columns:
        sd = float(np.std(Z[col], ddof=1))
        if sd == 0 or not np.isfinite(sd):
            Z[col] = 0.0
        else:
            Z[col] = (Z[col] - float(np.mean(Z[col]))) / sd

    up = [g for g in feature_genes if g in PANEL_UP]
    down = [g for g in feature_genes if g in PANEL_DOWN]
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


def gene_level_replication(X_ext: pd.DataFrame, ext_meta: pd.DataFrame, feature_genes: List[str]) -> pd.DataFrame:
    nd_mask = ext_meta["group"].values == "ND"
    t2d_mask = ext_meta["group"].values == "T2D"

    rows = []
    for g in feature_genes:
        nd = X_ext.loc[nd_mask, g].values.astype(float)
        t2d = X_ext.loc[t2d_mask, g].values.astype(float)
        diff = float(np.mean(t2d) - np.mean(nd)) if len(nd) and len(t2d) else np.nan
        ext_dir = "up" if pd.notna(diff) and diff > 0 else "down"
        disc_dir = "up" if g in PANEL_UP else "down"
        rows.append({
            "gene": g,
            "discovery_direction": disc_dir,
            "external_direction": ext_dir,
            "direction_concordant": disc_dir == ext_dir if pd.notna(diff) else np.nan,
            "external_mean_ND": float(np.mean(nd)) if len(nd) else np.nan,
            "external_mean_T2D": float(np.mean(t2d)) if len(t2d) else np.nan,
            "external_mean_diff_T2D_minus_ND": diff,
            "external_cohen_d": float(cohen_d(nd, t2d)) if len(nd) and len(t2d) else np.nan,
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

    p.add_argument("--external-meta-override", type=Path, default=None)

    p.add_argument("--discovery-transform", choices=["auto", "none", "log2"], default="none")
    p.add_argument("--external-transform", choices=["auto", "none", "log2"], default="auto")

    p.add_argument("--nd-max-hba1c", type=float, default=6.0)
    p.add_argument("--t2d-min-hba1c", type=float, default=6.5)
    p.add_argument("--min-overlap", type=int, default=6)

    return p.parse_args()


def main():
    args = parse_args()
    safe_mkdir(args.outdir)
    cache_dir = args.outdir / "geo_cache"
    safe_mkdir(cache_dir)

    eprint("[info] Loading discovery cohort...")
    disc_expr, disc_meta, disc_transform = load_discovery_inputs(
        expr_path=args.discovery_expr,
        meta_path=args.discovery_meta,
        sample_col=args.sample_col,
        group_col=args.group_col,
        transform_mode=args.discovery_transform,
    )

    eprint(f"[info] Discovery samples retained: {len(disc_meta)}")
    eprint(f"[info]   ND: {(disc_meta['group'] == 'ND').sum()}")
    eprint(f"[info]   T2D: {(disc_meta['group'] == 'T2D').sum()}")

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
        try:
            eprint(ext_meta.groupby(["group", "group_reason"]).size().to_string())
        except Exception:
            eprint(ext_meta["group"].value_counts(dropna=False).to_string())
    else:
        eprint(ext_meta["group"].value_counts(dropna=False).to_string())

    if "hba1c_inferred" in ext_meta.columns:
        eprint("[info] HbA1c summary in matched external samples:")
        eprint(ext_meta["hba1c_inferred"].describe().to_string())

    eprint("[info] Resolving overlap between panel genes and discovery/external matrices...")
    overlap = build_overlap_table(disc_expr, ext_expr)
    overlap.to_csv(args.outdir / "overlap_table.csv", index=False)

    X_train, X_ext, feature_genes, y_train, y_ext = build_feature_matrices(
        overlap_table=overlap,
        discovery_expr=disc_expr,
        discovery_meta=disc_meta,
        external_expr=ext_expr,
        external_meta=ext_meta,
        min_overlap=args.min_overlap,
    )

    eprint(f"[info] Overlap genes used: {feature_genes}")
    eprint(f"[info] External class counts after filtering: ND={(y_ext == 0).sum()}, T2D={(y_ext == 1).sum()}")

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Discovery cohort has only one class after filtering.")
    if len(np.unique(y_ext)) < 2:
        raise RuntimeError(
            "External cohort has only one class after metadata matching and label filtering. "
            "Inspect GSE50244_parsed_metadata.csv and thresholds."
        )

    scaler, clf = fit_locked_logistic(X_train, y_train)
    p_ext = clf.predict_proba(scaler.transform(X_ext.values))[:, 1]
    locked_metrics = external_metrics(y_ext, p_ext, threshold=0.5)

    coef_df = pd.DataFrame({
        "gene": feature_genes,
        "coefficient": clf.coef_.ravel(),
        "odds_ratio_per_SD": np.exp(clf.coef_.ravel()),
    }).sort_values("coefficient", ascending=False)
    coef_df.to_csv(args.outdir / "locked_model_coefficients.csv", index=False)

    reduced_score = compute_reduced_score(X_ext, feature_genes)
    if len(np.unique(y_ext)) < 2:
        score_auc = np.nan
        score_thr = 0.5
    else:
        score_auc = float(roc_auc_score(y_ext, reduced_score.values))
        score_thr = compute_youden_threshold(y_ext, reduced_score.values)
    reduced_metrics = external_metrics(y_ext, reduced_score.values, threshold=score_thr)

    pred_df = pd.DataFrame({
        "sample": X_ext.index,
        "group": np.where(y_ext == 1, "T2D", "ND"),
        "locked_probability": p_ext,
        "reduced_score": reduced_score.values,
    })
    pred_df.to_csv(args.outdir / "external_sample_predictions.csv", index=False)

    rep = gene_level_replication(X_ext, ext_meta, feature_genes)
    rep.to_csv(args.outdir / "external_gene_replication.csv", index=False)

    plot_roc(y_ext, p_ext, "GSE50244 external ROC (locked reduced model)", args.outdir / "roc_locked_model.png")
    plot_roc(y_ext, reduced_score.values, "GSE50244 external ROC (reduced signed score)", args.outdir / "roc_reduced_score.png")
    plot_box(
        reduced_score.values,
        np.where(y_ext == 1, "T2D", "ND"),
        "GSE50244 reduced signed score",
        "Reduced score",
        args.outdir / "boxplot_reduced_score.png",
    )
    plot_box(
        p_ext,
        np.where(y_ext == 1, "T2D", "ND"),
        "GSE50244 locked model probabilities",
        "Predicted probability of T2D",
        args.outdir / "boxplot_locked_probability.png",
    )

    summary = {
        "discovery_transform_used": disc_transform,
        "external_transform_used": ext_transform,
        "external_sample_match_mode": match_mode,
        "n_discovery_samples": int(len(disc_meta)),
        "n_discovery_ND": int((disc_meta["group"] == "ND").sum()),
        "n_discovery_T2D": int((disc_meta["group"] == "T2D").sum()),
        "n_external_samples_total_labeled": int(len(ext_meta)),
        "n_external_ND": int((ext_meta["group"] == "ND").sum()),
        "n_external_T2D": int((ext_meta["group"] == "T2D").sum()),
        "feature_overlap_genes": feature_genes,
        "n_feature_overlap": int(len(feature_genes)),
        "locked_model_metrics": locked_metrics,
        "reduced_score_auc": score_auc,
        "reduced_score_metrics_at_youden_threshold": reduced_metrics,
        "direction_concordance": {
            "n_concordant": int(np.nansum(rep["direction_concordant"].values.astype(float))) if len(rep) else 0,
            "n_total": int(len(rep)),
            "fraction_concordant": float(np.nanmean(rep["direction_concordant"].values.astype(float))) if len(rep) else None,
        },
    }
    with open(args.outdir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    report = [
        "GSE50244 external validation summary",
        "==================================",
        "",
        f"Discovery transform used: {disc_transform}",
        f"External transform used: {ext_transform}",
        f"External sample match mode: {match_mode}",
        "",
        f"Discovery samples: {len(disc_meta)}",
        f"  ND: {(disc_meta['group'] == 'ND').sum()}",
        f"  T2D: {(disc_meta['group'] == 'T2D').sum()}",
        "",
        f"External labeled samples: {len(ext_meta)}",
        f"  ND: {(ext_meta['group'] == 'ND').sum()}",
        f"  T2D: {(ext_meta['group'] == 'T2D').sum()}",
        "",
        f"Feature overlap ({len(feature_genes)} genes): {', '.join(feature_genes)}",
        "",
        "Locked reduced logistic model",
        f"  AUC: {locked_metrics['AUC']:.3f}",
        f"  Sensitivity: {locked_metrics['Sensitivity']:.3f}",
        f"  Specificity: {locked_metrics['Specificity']:.3f}",
        f"  Balanced accuracy: {locked_metrics['BalancedAccuracy']:.3f}",
        f"  MCC: {locked_metrics['MCC']:.3f}",
        "",
        "Reduced signed score",
        f"  AUC: {score_auc:.3f}",
        f"  Youden threshold: {reduced_metrics['Threshold']:.5f}",
        f"  Sensitivity: {reduced_metrics['Sensitivity']:.3f}",
        f"  Specificity: {reduced_metrics['Specificity']:.3f}",
        f"  Balanced accuracy: {reduced_metrics['BalancedAccuracy']:.3f}",
        f"  MCC: {reduced_metrics['MCC']:.3f}",
        "",
        f"Direction concordance: {int(np.nansum(rep['direction_concordant'].values.astype(float)))}/{len(rep)}",
    ]
    with open(args.outdir / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")

    print("\n".join(report))


if __name__ == "__main__":
    main()
