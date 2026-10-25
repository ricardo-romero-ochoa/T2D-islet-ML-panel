#!/usr/bin/env python3
"""Check the current manuscript-facing results against expected values."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

EXPECTED = {"panel_n": 10, "nested_auc": 0.960, "external_auc": 0.907}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--nested-dir", type=Path, default=Path("results/nested_validation"))
    p.add_argument("--external-dir", type=Path, default=Path("results/external_validation/gse50244_fixed_threshold"))
    return p.parse_args()

def maybe_read(path):
    return pd.read_csv(path) if path.exists() else None

def main():
    args = parse_args(); out = {}
    panel = maybe_read(args.results_dir / "final_gene_panel.csv")
    if panel is not None: out["panel_n"] = int(len(panel))
    ns = args.nested_dir / "nested_5fold_summary.json"
    if ns.exists():
        obj = json.loads(ns.read_text())
        out["nested_auc"] = float(obj.get("nested_5fold_auc", obj.get("nested_loocv_auc", float("nan"))))
    ms = args.external_dir / "metrics_summary.json"
    if ms.exists():
        obj = json.loads(ms.read_text())
        out["external_auc"] = float(obj.get("reduced_score_auc", obj.get("external_auc", float("nan"))))
    rows=[]
    for k,v in EXPECTED.items():
        obs=out.get(k)
        if obs is None: status="missing"
        elif isinstance(v,int): status="PASS" if obs==v else "CHECK"
        else: status="PASS" if abs(obs-v)<=0.02 else "CHECK"
        rows.append({"metric":k,"expected":v,"observed":obs,"status":status})
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__": main()
