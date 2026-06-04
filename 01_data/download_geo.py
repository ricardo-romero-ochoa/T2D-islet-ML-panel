#!/usr/bin/env python3
"""
01_data/download_geo.py

Downloads GEO series matrix files and parses expression matrices + metadata.
For RNA-seq datasets whose expression is in supplementary files (e.g. GSE164416),
place the downloaded count matrix in data/raw/<GSE>/<GSE>_expression.csv and
re-run with --parse-only.

Usage:
    python 01_data/download_geo.py                          # all datasets in config
    python 01_data/download_geo.py --datasets GSE164416    # specific dataset
    python 01_data/download_geo.py --skip-existing         # skip already downloaded
    python 01_data/download_geo.py --parse-only            # skip download, parse only

Special case — GSE164416 (HTSeq supplementary file):
    curl -o data/raw/GSE164416/GSE164416_DP_htseq_counts.txt.gz \\
      "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164416/suppl/GSE164416_DP_htseq_counts.txt.gz"
    gunzip data/raw/GSE164416/GSE164416_DP_htseq_counts.txt.gz
    python 01_data/convert_gse164416.py
"""

import os, sys, argparse, logging, urllib.request, gzip, shutil, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_RAW, ALL_DATASETS, LOGS_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, "01_download.log")), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"

def geo_path(gse): return f"{GEO_FTP}/{gse[:-3]}nnn/{gse}"

def download_file(url, dest, retries=3):
    for attempt in range(retries):
        try:
            log.info(f"  Downloading: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as out:
                shutil.copyfileobj(r, out)
            return True
        except Exception as e:
            log.warning(f"  Attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return False

def decompress(gz_path):
    out = gz_path.replace(".gz", "")
    with gzip.open(gz_path, "rb") as fi, open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(gz_path)
    return out

def parse_series_matrix(matrix_file, gse_id, out_dir):
    import pandas as pd
    from io import StringIO
    meta_lines, expr_lines, in_table = [], [], False

    with open(matrix_file, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!"):
                meta_lines.append(line)
            elif line.startswith('"ID_REF"') or line.startswith("ID_REF"):
                in_table = True; expr_lines.append(line)
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                expr_lines.append(line)

    if expr_lines:
        expr_df = pd.read_csv(StringIO("\n".join(expr_lines)), sep="\t", index_col=0)
        expr_df = expr_df.dropna(how="all").apply(pd.to_numeric, errors="coerce")
        if not expr_df.empty:
            expr_df.to_csv(os.path.join(out_dir, f"{gse_id}_expression.csv"))
            log.info(f"  Expression: {expr_df.shape[0]} probes × {expr_df.shape[1]} samples")

    meta = {}
    for line in meta_lines:
        if line.startswith("!Sample_"):
            parts = line[1:].split("\t")
            key = parts[0].strip()
            vals = [v.strip().strip('"') for v in parts[1:]]
            meta.setdefault(key, []).extend(vals)

    if meta:
        n = max(len(v) for v in meta.values())
        meta_aligned = {k: v for k, v in meta.items() if len(v) == n}
        if meta_aligned:
            meta_df = pd.DataFrame(meta_aligned)
            if "Sample_geo_accession" in meta_df.columns:
                meta_df = meta_df.set_index("Sample_geo_accession")
            meta_df.to_csv(os.path.join(out_dir, f"{gse_id}_metadata.csv"))
            log.info(f"  Metadata: {meta_df.shape}")

def download_dataset(gse_id, out_dir, skip_existing=False):
    os.makedirs(out_dir, exist_ok=True)
    base = geo_path(gse_id)
    for url, local_gz in [
        (f"{base}/matrix/{gse_id}_series_matrix.txt.gz",
         os.path.join(out_dir, f"{gse_id}_series_matrix.txt.gz")),
        (f"{base}/soft/{gse_id}_family.soft.gz",
         os.path.join(out_dir, f"{gse_id}_family.soft.gz")),
    ]:
        local_txt = local_gz.replace(".gz", "")
        if skip_existing and os.path.exists(local_txt):
            log.info(f"  Skipping (exists): {os.path.basename(local_txt)}")
            continue
        if download_file(url, local_gz):
            decompress(local_gz)
            log.info(f"  ✓ {os.path.basename(local_txt)}")
        else:
            log.error(f"  ✗ Failed: {url}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--parse-only", action="store_true")
    args = ap.parse_args()

    for gse_id in args.datasets:
        out_dir = os.path.join(DATA_RAW, gse_id)
        log.info(f"\n{'─'*50}\n{gse_id}")
        if not args.parse_only:
            download_dataset(gse_id, out_dir, args.skip_existing)
        matrix = os.path.join(out_dir, f"{gse_id}_series_matrix.txt")
        if os.path.exists(matrix):
            parse_series_matrix(matrix, gse_id, out_dir)
        else:
            log.warning(f"  No matrix file — if RNA-seq, place count CSV manually")

if __name__ == "__main__":
    main()
