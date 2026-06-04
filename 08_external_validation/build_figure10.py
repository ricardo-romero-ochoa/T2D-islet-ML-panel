#!/usr/bin/env python3
"""
Build manuscript Figure 10 from saved external-validation outputs.

This version removes the overall figure-number title from inside the image and
keeps only panel labels/titles (A/B/C). Panel C is rendered as an ECDF overlay
for a cleaner display of the discovery-vs-external distribution shift.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURES_DIR, FIGURE_DPI


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--discovery-scores', required=True, type=Path)
    p.add_argument('--external-scores', required=True, type=Path)
    p.add_argument('--metrics-json', type=Path, default=None)
    p.add_argument('--metrics-txt', type=Path, default=None)
    p.add_argument('--outdir', type=Path, default=Path(FIGURES_DIR))
    return p.parse_args()


def read_table_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine='python', compression='infer')


def load_scores(path: Path) -> pd.DataFrame:
    df = read_table_auto(path).copy()
    if 'reduced_score' not in df.columns:
        raise ValueError(f'Missing reduced_score in {path}')
    if 'group' not in df.columns:
        raise ValueError(f'Missing group in {path}')
    cols = ['group', 'reduced_score']
    if 'sample' in df.columns:
        cols = ['sample'] + cols
    return df[cols].copy()


def extract_auc_and_threshold(metrics_json: Path | None, metrics_txt: Path | None):
    auc_ext = None
    threshold = None
    if metrics_json and metrics_json.exists():
        obj = json.loads(metrics_json.read_text())
        if 'external_reduced_score_metrics_at_frozen_threshold' in obj:
            auc_ext = obj['external_reduced_score_metrics_at_frozen_threshold'].get('AUC')
            threshold = obj.get('frozen_threshold')
        elif 'external_reduced_score' in obj:
            auc_ext = obj['external_reduced_score'].get('AUC')
            threshold = obj.get('frozen_threshold')
        else:
            auc_ext = obj.get('reduced_score_auc') or obj.get('external_auc')
            threshold = obj.get('frozen_threshold')
    if (auc_ext is None or threshold is None) and metrics_txt and metrics_txt.exists():
        txt = metrics_txt.read_text(encoding='utf-8', errors='ignore')
        m_auc = re.search(r'External reduced score.*?AUC:\s*([0-9.]+)', txt, re.S)
        m_thr = re.search(r'Frozen threshold:\s*([-0-9.]+)', txt)
        if m_auc:
            auc_ext = float(m_auc.group(1))
        if m_thr:
            threshold = float(m_thr.group(1))
    return auc_ext, threshold


def ecdf(vals):
    vals = np.sort(np.asarray(vals, dtype=float))
    y = np.arange(1, len(vals)+1) / len(vals)
    return vals, y


def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    disc = load_scores(args.discovery_scores)
    ext = load_scores(args.external_scores)
    auc_ext, threshold = extract_auc_and_threshold(args.metrics_json, args.metrics_txt)
    if threshold is None:
        raise ValueError('Could not determine frozen threshold from metrics file.')

    y_ext = (ext['group'].astype(str) == 'T2D').astype(int).values
    score_ext = ext['reduced_score'].values.astype(float)
    if auc_ext is None:
        auc_ext = float(roc_auc_score(y_ext, score_ext))
    fpr, tpr, _ = roc_curve(y_ext, score_ext)

    fig = plt.figure(figsize=(13.8, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.18, 1.18], wspace=0.32)

    axA = fig.add_subplot(gs[0,0])
    axA.plot(fpr, tpr, lw=2.5, color='#1F4E79', label=f'8-gene reduced score (AUC = {auc_ext:.3f})')
    axA.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5, label='Random')
    axA.set_xlabel('False Positive Rate')
    axA.set_ylabel('True Positive Rate')
    axA.set_title('A. External ROC in GSE50244', loc='left', fontsize=11, fontweight='bold')
    axA.legend(loc='lower right', fontsize=8)
    style_axes(axA)

    axB = fig.add_subplot(gs[0,1])
    groups = [
        disc.loc[disc['group']=='ND', 'reduced_score'].values,
        disc.loc[disc['group']=='T2D', 'reduced_score'].values,
        ext.loc[ext['group']=='ND', 'reduced_score'].values,
        ext.loc[ext['group']=='T2D', 'reduced_score'].values,
    ]
    axB.boxplot(groups, tick_labels=['Discovery ND','Discovery T2D','External ND','External T2D'])
    axB.axhline(threshold, ls='--', lw=1.2, color='#C0392B', label=f'Frozen threshold = {threshold:.3f}')
    axB.set_ylabel('Reduced score')
    axB.set_title('B. Score distributions at the frozen threshold', loc='left', fontsize=11, fontweight='bold')
    axB.tick_params(axis='x', labelrotation=16)
    axB.legend(loc='lower right', fontsize=8, frameon=False)
    style_axes(axB)

    axC = fig.add_subplot(gs[0,2])
    x_disc, y_disc = ecdf(disc['reduced_score'].values)
    x_ext, y_ext_ecdf = ecdf(ext['reduced_score'].values)
    axC.step(x_disc, y_disc, where='post', lw=2.1, color='#2E86C1', label='Discovery cohort')
    axC.step(x_ext, y_ext_ecdf, where='post', lw=2.1, color='#27AE60', label='External cohort')
    axC.axvline(threshold, ls='--', lw=1.2, color='#C0392B', label='Frozen threshold')
    axC.set_xlabel('Reduced score')
    axC.set_ylabel('Cumulative fraction')
    axC.set_title('C. Distribution shift between discovery and external cohorts', loc='left', fontsize=11, fontweight='bold')
    axC.legend(loc='upper left', fontsize=8, frameon=False)
    style_axes(axC)

    plt.tight_layout()
    for fmt in ['png','svg']:
        plt.savefig(args.outdir / f'figure10_external_transportability.{fmt}', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved Figure 10 to: {args.outdir}')


if __name__ == '__main__':
    main()
