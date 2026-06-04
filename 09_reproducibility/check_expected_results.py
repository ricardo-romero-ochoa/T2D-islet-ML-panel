#!/usr/bin/env python3
"""Check key manuscript-facing results against expected values with loose tolerances."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

EXPECTED = {
    'deg_n': 184,
    'panel_n': 10,
    'loocv_auc': 1.000,
    'external_auc': 0.907,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=Path, default=Path('results'))
    p.add_argument('--external-dir', type=Path, default=Path('results/external_validation/gse50244_fixed_threshold'))
    return p.parse_args()

def maybe_read(path):
    return pd.read_csv(path) if path.exists() else None

def main():
    args = parse_args()
    out = {}
    deg = maybe_read(args.results_dir / 'deg_list.csv')
    if deg is not None:
        out['deg_n'] = int(len(deg))
    panel = maybe_read(args.results_dir / 'final_gene_panel.csv')
    if panel is not None:
        out['panel_n'] = int(len(panel))
    loocv = maybe_read(args.results_dir / 'loocv_performance.csv')
    if loocv is not None and not loocv.empty:
        col = 'auc' if 'auc' in loocv.columns else 'AUC'
        out['loocv_auc'] = float(loocv.iloc[0][col])
    ms = args.external_dir / 'metrics_summary.json'
    if ms.exists():
        obj = json.loads(ms.read_text())
        out['external_auc'] = float(obj.get('reduced_score_auc', obj.get('external_auc', float('nan'))))

    rows = []
    for k,v in EXPECTED.items():
        obs = out.get(k)
        if obs is None:
            status = 'missing'
        elif isinstance(v, int):
            status = 'PASS' if obs == v else 'CHECK'
        else:
            status = 'PASS' if abs(obs - v) <= 0.02 else 'CHECK'
        rows.append({'metric': k, 'expected': v, 'observed': obs, 'status': status})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
