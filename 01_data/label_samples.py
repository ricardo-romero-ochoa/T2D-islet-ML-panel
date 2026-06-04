#!/usr/bin/env python3
"""
01_data/label_samples.py

Assigns T2D (1) / Control (0) / Exclude (-1) labels to each sample.
Uses LABEL_MAPS from config.py, with regex fallback for unmatched samples.
Outputs per-dataset label CSVs and a combined sample manifest.
"""

import os, sys, re, logging
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_RAW, DATA_PROCESSED, ALL_DATASETS, LABEL_MAPS, LOGS_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "01_labeling.log")), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

T2D_RE  = re.compile(r"\b(t2d|t2dm|type.?2|diabetic|diabetes mellitus)\b", re.I)
CTRL_RE = re.compile(r"\b(control|healthy|normal|ngt|non.?diabetic|ctrl)\b", re.I)

def infer_label(text, label_map):
    tl = text.lower()
    for kw, lbl in label_map.items():
        if kw.lower() in tl:
            return lbl
    if T2D_RE.search(text):  return 1
    if CTRL_RE.search(text): return 0
    return None

def label_dataset(gse_id):
    meta_path = os.path.join(DATA_RAW, gse_id, f"{gse_id}_metadata.csv")
    if not os.path.exists(meta_path):
        log.warning(f"  {gse_id}: metadata not found")
        return None

    meta = pd.read_csv(meta_path, index_col=0)
    lmap = LABEL_MAPS.get(gse_id, {})
    PRIORITY = ["Sample_characteristics_ch1","Sample_source_name_ch1",
                "Sample_title","Sample_description"]
    rows = []
    for sid in meta.index:
        row = meta.loc[sid]
        label, src = None, "unknown"
        for col in PRIORITY:
            if col in row.index:
                label = infer_label(str(row[col]), lmap)
                if label is not None:
                    src = col; break
        if label is None:
            for col in row.index:
                label = infer_label(str(row[col]), lmap)
                if label is not None:
                    src = f"fallback:{col}"; break
        rows.append({"sample_id": sid, "label": label, "label_source": src, "gse_id": gse_id})

    df = pd.DataFrame(rows)
    t2d  = (df["label"] == 1).sum()
    ctrl = (df["label"] == 0).sum()
    excl = (df["label"] == -1).sum()
    unres = df["label"].isna().sum()
    log.info(f"  {gse_id}: T2D={t2d}  Ctrl={ctrl}  Excl={excl}  Unresolved={unres}")
    return df

def main():
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    all_labels = []
    for gse_id in ALL_DATASETS:
        log.info(f"\nLabeling: {gse_id}")
        df = label_dataset(gse_id)
        if df is not None:
            out = os.path.join(DATA_PROCESSED, f"{gse_id}_labels.csv")
            df.to_csv(out, index=False)
            all_labels.append(df)

    if all_labels:
        manifest = pd.concat(all_labels, ignore_index=True)
        manifest.to_csv(os.path.join(DATA_PROCESSED, "sample_manifest.csv"), index=False)
        log.info(f"\nManifest: {len(manifest)} samples")
        summary = manifest[manifest["label"].isin([0,1,-1])].groupby("gse_id")["label"].value_counts().unstack(fill_value=0)
        log.info(f"\n{summary.to_string()}")

if __name__ == "__main__":
    main()
