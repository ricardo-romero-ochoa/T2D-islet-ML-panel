#!/usr/bin/env python3
"""
Export manuscript-facing CSV inputs for Figures 7–9.

This script assumes the core ML pipeline has already been run:
  03_feature_selection/deg_analysis.py
  03_feature_selection/feature_selection.py
  05_validation/loocv_validation.py
  05_validation/leakage_verification.py
  06_visualization/plot_gene_auc.py

Outputs are written to results/figure_inputs/.
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PROCESSED, RESULTS_DIR, FIGURE_INPUTS_DIR, DISCOVERY_DATASETS
from figure_metadata import (
    lookup_gene_metadata,
    IMMUNESTRESS_GENES,
    BETACELLIDENTITYSECRETION_GENES,
    DEFAULT_EDGE_THRESHOLD,
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_discovery_expression_and_labels():
    gse = DISCOVERY_DATASETS[0]
    expr = pd.read_csv(os.path.join(DATA_PROCESSED, f"{gse}_expr_normalized.csv"), index_col=0)
    labels = pd.read_csv(os.path.join(DATA_PROCESSED, f"{gse}_labels.csv"))
    labels = labels[labels["label"].isin([0, 1])].copy()
    common = [c for c in expr.columns if c in labels["sample_id"].values]
    expr = expr[common]
    labels = labels.set_index("sample_id").loc[common].reset_index()
    return expr, labels


def build_panel_metadata(final_panel: pd.DataFrame, deg_results: pd.DataFrame | None, gene_auc: pd.DataFrame | None):
    rows = []
    auc_map = dict(zip(gene_auc["gene"], gene_auc["auc"])) if gene_auc is not None else {}
    deg_map = deg_results.set_index("gene").to_dict("index") if deg_results is not None else {}

    for _, row in final_panel.iterrows():
        gene = row["gene"]
        meta = lookup_gene_metadata(gene)
        deg_info = deg_map.get(gene, {})
        rows.append({
            "gene": gene,
            "display_label": row.get("display_label", meta["display_label"]),
            "gene_symbol": row.get("gene_symbol", meta["gene_symbol"]),
            "biotype": row.get("biotype", meta["biotype"]),
            "direction": row.get("direction", meta["direction"]),
            "votes": row.get("votes", np.nan),
            "composite_score": row.get("composite_score", np.nan),
            "auc": auc_map.get(gene, np.nan),
            "log2FC": deg_info.get("log2FC", np.nan),
            "adj_pvalue": deg_info.get("adj_pvalue", np.nan),
            "pathway_group": meta["pathway_group"],
            "pathway_context": meta["pathway_context"],
            "t2d_context": meta["t2d_context"],
        })
    return pd.DataFrame(rows)


def build_edges(expr: pd.DataFrame, panel_meta: pd.DataFrame, threshold: float = DEFAULT_EDGE_THRESHOLD):
    genes = [g for g in panel_meta["gene"] if g in expr.index]
    sub = expr.loc[genes].T.corr(method="pearson")
    edges = []
    for i, g1 in enumerate(genes):
        for g2 in genes[i+1:]:
            r = float(sub.loc[g1, g2])
            if abs(r) >= threshold:
                edges.append({
                    "source": g1,
                    "target": g2,
                    "source_label": panel_meta.set_index("gene").loc[g1, "display_label"],
                    "target_label": panel_meta.set_index("gene").loc[g2, "display_label"],
                    "correlation": r,
                    "abs_correlation": abs(r),
                    "sign": "positive" if r >= 0 else "negative",
                })
    return pd.DataFrame(edges)


def build_module_sets():
    rows = []
    for g in IMMUNESTRESS_GENES:
        rows.append({"module": "ImmuneStress", "gene": g})
    for g in BETACELLIDENTITYSECRETION_GENES:
        rows.append({"module": "BetaCellIdentitySecretion", "gene": g})
    return pd.DataFrame(rows)


def build_fig9_overlap(panel_meta: pd.DataFrame):
    immune = set(IMMUNESTRESS_GENES)
    identity = set(BETACELLIDENTITYSECRETION_GENES)
    rows = []
    for _, row in panel_meta.iterrows():
        symbol = row["gene_symbol"]
        in_immune = symbol in immune
        in_identity = symbol in identity
        if in_identity:
            convergence_axis = "Beta-cell identity/secretion"
        elif in_immune:
            convergence_axis = "Immune/stress"
        else:
            # concept-level assignment
            if row["pathway_group"] in {"GABA signaling", "Glucose transport", "Arginine metabolism", "WNT inhibition"}:
                convergence_axis = "Identity-linked / dedifferentiation-linked"
            elif row["pathway_group"] in {"Neuroimmune signaling", "Neural-islet axis", "Hedgehog signaling", "Integrin/cytoskeleton", "Non-coding regulation", "RNA processing"}:
                convergence_axis = "ML-specific extension"
            else:
                convergence_axis = "ML-specific extension"
        rows.append({
            "gene": row["gene"],
            "display_label": row["display_label"],
            "gene_symbol": symbol,
            "auc": row["auc"],
            "direction": row["direction"],
            "pathway_group": row["pathway_group"],
            "t2d_context": row["t2d_context"],
            "in_ImmuneStress": in_immune,
            "in_BetaCellIdentitySecretion": in_identity,
            "literal_module_overlap": in_immune or in_identity,
            "convergence_axis": convergence_axis,
        })
    return pd.DataFrame(rows)


def main():
    ensure_dir(FIGURE_INPUTS_DIR)

    final_panel_path = os.path.join(RESULTS_DIR, "final_gene_panel.csv")
    if not os.path.exists(final_panel_path):
        raise FileNotFoundError("Run feature_selection.py first to create results/final_gene_panel.csv")

    final_panel = pd.read_csv(final_panel_path)
    deg_path = os.path.join(RESULTS_DIR, "deg_results.csv")
    deg_results = pd.read_csv(deg_path) if os.path.exists(deg_path) else None
    auc_path = os.path.join(FIGURE_INPUTS_DIR, "fig7_single_gene_auc.csv")
    gene_auc = pd.read_csv(auc_path) if os.path.exists(auc_path) else None

    expr, labels = load_discovery_expression_and_labels()
    panel_meta = build_panel_metadata(final_panel, deg_results, gene_auc)
    panel_meta.to_csv(os.path.join(FIGURE_INPUTS_DIR, "panel_gene_metadata.csv"), index=False)

    # Figure 8 inputs
    nodes = panel_meta.copy()
    nodes["node_size"] = nodes["auc"].fillna(nodes["auc"].median() if nodes["auc"].notna().any() else 0.8)
    nodes.to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig8_nodes.csv"), index=False)

    edges = build_edges(expr, panel_meta)
    edges.to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig8_edges.csv"), index=False)

    expr_panel = expr.loc[[g for g in panel_meta["gene"] if g in expr.index]].copy()
    expr_panel.index = [panel_meta.set_index("gene").loc[g, "display_label"] for g in expr_panel.index]
    expr_panel.to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig8_panel_expression_matrix.csv"))

    # Figure 9 inputs
    module_sets = build_module_sets()
    module_sets.to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig9_module_gene_sets.csv"), index=False)

    overlap = build_fig9_overlap(panel_meta)
    overlap.to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig9_panel_module_overlap.csv"), index=False)
    overlap.to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig9_bubble_table.csv"), index=False)

    summary_rows = [
        {"set_name": "ML_panel", "n_genes": int(len(panel_meta))},
        {"set_name": "ImmuneStress", "n_genes": int(len(IMMUNESTRESS_GENES))},
        {"set_name": "BetaCellIdentitySecretion", "n_genes": int(len(BETACELLIDENTITYSECRETION_GENES))},
        {"set_name": "ML ∩ ImmuneStress", "n_genes": int(overlap["in_ImmuneStress"].sum())},
        {"set_name": "ML ∩ BetaCellIdentitySecretion", "n_genes": int(overlap["in_BetaCellIdentitySecretion"].sum())},
        {"set_name": "ML literal overlap any module", "n_genes": int(overlap["literal_module_overlap"].sum())},
    ]
    pd.DataFrame(summary_rows).to_csv(os.path.join(FIGURE_INPUTS_DIR, "fig9_overlap_summary.csv"), index=False)

    print(f"Saved manuscript-facing figure inputs to: {FIGURE_INPUTS_DIR}")


if __name__ == "__main__":
    main()
