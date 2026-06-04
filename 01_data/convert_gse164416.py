#!/usr/bin/env python3
"""
01_data/convert_gse164416.py

Converts GSE164416 HTSeq supplementary count file to the expected
expression CSV format, renaming DP-code columns to GSM accession IDs.

Prerequisites:
    1. Run download_geo.py for GSE164416 (downloads metadata)
    2. Download the supplementary count file:
       curl -o data/raw/GSE164416/GSE164416_DP_htseq_counts.txt.gz \\
         "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164416/suppl/GSE164416_DP_htseq_counts.txt.gz"
    3. Decompress:
       gunzip data/raw/GSE164416/GSE164416_DP_htseq_counts.txt.gz
    4. Run this script:
       python 01_data/convert_gse164416.py
"""

import os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_RAW, LOGS_DIR
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

GSE = "GSE164416"
RAW_DIR = os.path.join(DATA_RAW, GSE)
COUNT_FILE = os.path.join(RAW_DIR, f"{GSE}_DP_htseq_counts.txt")
META_FILE  = os.path.join(RAW_DIR, f"{GSE}_metadata.csv")
OUT_FILE   = os.path.join(RAW_DIR, f"{GSE}_expression.csv")

def main():
    if not os.path.exists(COUNT_FILE):
        log.error(f"Count file not found: {COUNT_FILE}")
        log.error("Download it first — see script docstring for curl command.")
        sys.exit(1)

    if not os.path.exists(META_FILE):
        log.error(f"Metadata not found: {META_FILE}")
        log.error("Run: python 01_data/download_geo.py --datasets GSE164416")
        sys.exit(1)

    log.info("Loading count matrix...")
    df = pd.read_csv(COUNT_FILE, sep="\t", index_col=0)

    # Drop HTSeq summary rows
    df = df[~df.index.str.startswith("__")]
    # Keep only Ensembl gene IDs
    df = df[df.index.str.startswith("ENSG")]
    log.info(f"  {df.shape[0]} genes × {df.shape[1]} samples")
    log.info(f"  Column format: {df.columns[:3].tolist()}")

    # Build DP-code → GSM accession map from metadata
    meta = pd.read_csv(META_FILE, index_col=0)
    dp_to_gsm = {}
    for gsm, row in meta.iterrows():
        title = str(row.get("Sample_title", ""))
        for part in title.split("_"):
            if part.startswith("DP"):
                dp_to_gsm[part] = gsm
                break

    df = df.rename(columns=dp_to_gsm)
    matched = sum(1 for c in df.columns if str(c).startswith("GSM"))
    log.info(f"  Matched {matched}/{len(df.columns)} columns to GSM IDs")

    df.to_csv(OUT_FILE)
    log.info(f"  Saved: {OUT_FILE}")
    log.info("Done. Now run: python 01_data/label_samples.py")

if __name__ == "__main__":
    main()
