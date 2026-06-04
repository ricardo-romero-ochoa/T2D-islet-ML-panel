#!/usr/bin/env python3
"""
Build final manuscript-facing Figures 7–9 from saved CSVs.

This version removes figure-number suptitles from inside the plots (the manuscript
caption should carry the figure number/title) while retaining panel labels A/B/C.
It also redraws Figure 9 in a cleaner, less cluttered layout.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURE_INPUTS_DIR, FIGURES_DIR, FIGURE_DPI


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv(name: str) -> pd.DataFrame:
    path = os.path.join(FIGURE_INPUTS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_flowchart(ax):
    ax.axis("off")
    boxes = [
        (0.05, 0.78, 0.24, 0.12, "GSE164416 discovery cohort\nND vs T2D subset (n=57)"),
        (0.37, 0.78, 0.24, 0.12, "DEG filtering\n184 genes"),
        (0.69, 0.78, 0.24, 0.12, "Three-method feature selection\nLASSO + SVM-RFE + RF"),
        (0.05, 0.48, 0.24, 0.12, "Consensus rule\nselected by ≥2 methods"),
        (0.37, 0.48, 0.24, 0.12, "Final 10-gene panel\ncomposite ranking"),
        (0.69, 0.48, 0.24, 0.12, "Four base classifiers +\nsoft-voting ensemble"),
        (0.37, 0.18, 0.24, 0.12, "LOOCV primary validation\nensemble AUC = 1.000"),
        (0.69, 0.18, 0.24, 0.12, "Leakage verification\nglobal QN before CV"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.02",
                                    facecolor="#F8F9F9", edgecolor="#34495E", linewidth=1.2))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=9)

    arrows = [((0.29,0.84),(0.37,0.84)), ((0.61,0.84),(0.69,0.84)),
              ((0.81,0.78),(0.17,0.60)), ((0.29,0.54),(0.37,0.54)), ((0.61,0.54),(0.69,0.54)),
              ((0.49,0.48),(0.49,0.30)), ((0.81,0.48),(0.81,0.30))]
    for (x1,y1),(x2,y2) in arrows:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#566573"))
    ax.set_title("A. Workflow", fontsize=11, loc="left", fontweight="bold")


def draw_venn_feature_selection(ax, membership: pd.DataFrame):
    lasso = set(membership.loc[membership["lasso_selected"], "gene"])
    svm = set(membership.loc[membership["svm_selected"], "gene"])
    rf = set(membership.loc[membership["rf_selected"], "gene"])
    centers = [(0.35,0.42),(0.65,0.42),(0.50,0.66)]
    colors = ["#AED6F1", "#A9DFBF", "#F9E79F"]
    labels = ["LASSO", "SVM-RFE", "Random Forest"]
    label_offsets = [(0,-0.31),(0,-0.31),(0,0.29)]
    for (cx,cy), color, lab, (dx,dy) in zip(centers, colors, labels, label_offsets):
        ax.add_patch(Circle((cx,cy), 0.24, color=color, alpha=0.45, ec=color))
        ax.text(cx+dx, cy+dy, lab, ha="center", va="center", fontsize=10, fontweight="bold")
    vals = [
        (0.22,0.42,len(lasso - svm - rf)),
        (0.78,0.42,len(svm - lasso - rf)),
        (0.50,0.80,len(rf - lasso - svm)),
        (0.50,0.44,len((lasso & svm) - rf)),
        (0.38,0.56,len((lasso & rf) - svm)),
        (0.62,0.56,len((svm & rf) - lasso)),
        (0.50,0.56,len(lasso & svm & rf)),
    ]
    for x,y,n in vals:
        ax.text(x, y, str(n), ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_title("B. Feature-selection overlap", fontsize=11, loc="left", fontweight="bold")


def draw_gene_auc(ax, gene_auc: pd.DataFrame):
    df = gene_auc.sort_values("auc", ascending=True).copy()
    colors = ["#2E86C1" if a == df["auc"].max() else "#95A5A6" for a in df["auc"]]
    bars = ax.barh(df["display_label"], df["auc"], color=colors, alpha=0.9, edgecolor="white", height=0.7)
    ax.axvline(0.5, color="gray", lw=1, ls="--", alpha=0.5)
    ax.axvline(0.9, color="#3498DB", lw=1, ls=":", alpha=0.8)
    ax.axvline(0.95, color="#E74C3C", lw=1, ls=":", alpha=0.8)
    for bar, val in zip(bars, df["auc"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("Single-gene AUC-ROC")
    ax.set_xlim(max(0.5, df["auc"].min() - 0.05), 1.02)
    ax.set_title("C. Individual gene AUC values", fontsize=11, loc="left", fontweight="bold")
    style_axes(ax)


def draw_roc(ax, roc_points: pd.DataFrame, auc_text: str, title: str, color="#1F4E79"):
    ax.plot(roc_points["fpr"], roc_points["tpr"], lw=2.4, color=color, label=auc_text)
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
    style_axes(ax)


def build_figure7():
    membership = read_csv("fig7_feature_selection_membership.csv")
    gene_auc = read_csv("fig7_single_gene_auc.csv")
    loocv_roc = read_csv("fig7_loocv_roc_points.csv")
    loocv_metrics = read_csv("fig7_loocv_metrics.csv")
    leak_metrics = read_csv("fig7_leakage_verification_metrics.csv")
    proper_roc = read_csv("fig7_proper_loocv_roc_points.csv")
    leak_roc = read_csv("fig7_leakage_global_qn_roc_points.csv")

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0], hspace=0.35, wspace=0.28)

    axA = fig.add_subplot(gs[0, :]); draw_flowchart(axA)
    axB = fig.add_subplot(gs[1, 0]); draw_venn_feature_selection(axB, membership)
    axC = fig.add_subplot(gs[1, 1]); draw_gene_auc(axC, gene_auc)
    axD = fig.add_subplot(gs[2, 0])
    draw_roc(axD, loocv_roc, f"Ensemble (AUC = {float(loocv_metrics.loc[0, 'auc']):.3f})", "D. LOOCV ROC for soft-voting ensemble")
    axE = fig.add_subplot(gs[2, 1])
    proper_auc = float(leak_metrics.loc[leak_metrics['workflow'] == 'proper_loocv', 'auc'].iloc[0])
    leak_auc = float(leak_metrics.loc[leak_metrics['workflow'] == 'global_qn_before_cv', 'auc'].iloc[0])
    axE.plot(proper_roc['fpr'], proper_roc['tpr'], lw=2.4, color="#1F4E79", label=f"Proper LOOCV (AUC = {proper_auc:.3f})")
    axE.plot(leak_roc['fpr'], leak_roc['tpr'], lw=2.4, color="#C0392B", label=f"Global quantile norm before CV (AUC = {leak_auc:.3f})")
    axE.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random")
    axE.set_xlabel("False Positive Rate"); axE.set_ylabel("True Positive Rate")
    axE.legend(loc="lower right", fontsize=8)
    axE.set_title("E. Leakage-verification sensitivity analysis", fontsize=11, loc="left", fontweight="bold")
    style_axes(axE)

    plt.tight_layout()
    for fmt in ["png", "svg"]:
        plt.savefig(os.path.join(FIGURES_DIR, f"figure7_ml_pipeline.{fmt}"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _circular_positions(labels):
    n = len(labels)
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    return {lab: (np.cos(a), np.sin(a)) for lab, a in zip(labels, ang)}


def build_figure8():
    nodes = read_csv("fig8_nodes.csv")
    edges = read_csv("fig8_edges.csv")
    labels = nodes["display_label"].tolist()
    pos = _circular_positions(labels)
    color_map = {
        "GABA signaling": "#4C78A8",
        "Glucose transport": "#72B7B2",
        "Arginine metabolism": "#F58518",
        "WNT inhibition": "#E45756",
        "Neural-islet axis": "#54A24B",
        "Neuroimmune signaling": "#EECA3B",
        "Hedgehog signaling": "#B279A2",
        "Integrin/cytoskeleton": "#FF9DA6",
        "RNA processing": "#9D755D",
        "Non-coding regulation": "#BAB0AC",
        "Unannotated": "#95A5A6",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")

    for _, e in edges.iterrows():
        x1, y1 = pos[e["source_label"]]
        x2, y2 = pos[e["target_label"]]
        ax.plot([x1, x2], [y1, y2],
                color="#1F77B4" if e["sign"] == "positive" else "#D62728",
                alpha=0.25 + 0.45 * min(1.0, abs(e["correlation"])),
                lw=0.5 + 3.0 * min(1.0, abs(e["correlation"])))

    for _, n in nodes.iterrows():
        x, y = pos[n["display_label"]]
        size = 500 + 1800 * float(n.get("node_size", 0.8))
        color = color_map.get(n["pathway_group"], "#95A5A6")
        ax.scatter([x], [y], s=size, color=color, edgecolor="white", linewidth=1.2, zorder=3, alpha=0.95)
        ax.text(x, y, n["display_label"], ha="center", va="center", fontsize=9, fontweight="bold")

    handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=color_map.get(grp, '#95A5A6'), markersize=9, label=grp)
               for grp in sorted(nodes["pathway_group"].dropna().unique())]
    ax.legend(handles=handles, bbox_to_anchor=(1.03, 0.5), loc="center left", frameon=False, fontsize=9)
    plt.tight_layout()
    for fmt in ["png", "svg"]:
        plt.savefig(os.path.join(FIGURES_DIR, f"figure8_panel_network.{fmt}"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def draw_clean_overlap(ax, overlap_df: pd.DataFrame):
    immune_n = 14
    identity_n = 14
    ml_n = int(len(overlap_df))
    shared_any = int(overlap_df["literal_module_overlap"].sum())

    centers = [(0.34,0.46),(0.66,0.46),(0.50,0.67)]
    colors = ["#F6D55C", "#3CAEA3", "#20639B"]
    labels = ["ImmuneStress", "BetaCellIdentitySecretion", "ML panel"]
    for (cx,cy), color, lab in zip(centers, colors, labels):
        ax.add_patch(Circle((cx,cy), 0.22, color=color, alpha=0.35, ec=color, lw=1.5))
        if lab == "ML panel":
            ax.text(cx, cy+0.18, lab, ha="center", va="center", fontsize=10, fontweight="bold")
        else:
            ax.text(cx, cy-0.30, lab, ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(0.24, 0.46, f"n={immune_n}", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.76, 0.46, f"n={identity_n}", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.50, 0.79, f"n={ml_n}", ha="center", va="center", fontsize=12, fontweight="bold")

    if shared_any == 0:
        ax.text(0.50, 0.55, "No literal gene overlap", ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(0.50, 0.49, "Conceptual convergence is summarized in panel B", ha="center", va="center", fontsize=8.5, color="#555555")
    else:
        ax.text(0.50, 0.55, f"Shared genes: n={shared_any}", ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_title("A. Gene-set overlap", fontsize=11, loc="left", fontweight="bold")


def build_figure9():
    overlap = read_csv("fig9_panel_module_overlap.csv")
    fig = plt.figure(figsize=(12.2, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.34)

    axA = fig.add_subplot(gs[0,0])
    draw_clean_overlap(axA, overlap)

    axB = fig.add_subplot(gs[0,1])
    axis_order = ["Identity-linked / dedifferentiation-linked", "ML-specific extension"]
    order = overlap.copy()
    order["convergence_axis"] = pd.Categorical(order["convergence_axis"], categories=axis_order, ordered=True)
    order = order.sort_values(["convergence_axis", "auc"], ascending=[True, False]).reset_index(drop=True)
    order["ypos"] = np.arange(len(order))[::-1]
    x_map = {axis_order[0]: 0, axis_order[1]: 1}
    colors = order["direction"].map({"up": "#E76F51", "down": "#4C78A8"}).fillna("#95A5A6")
    sizes = 250 + 1600 * order["auc"].fillna(order["auc"].median())

    rng = np.random.default_rng(42)
    x = order["convergence_axis"].map(x_map).astype(float).values
    x = x + rng.uniform(-0.05, 0.05, size=len(x))

    axB.scatter(x, order["ypos"], s=sizes, c=colors, alpha=0.85, edgecolor="white", linewidth=1.0)
    for xi, (_, row) in zip(x, order.iterrows()):
        axB.text(xi + 0.10, row["ypos"], f"{row['display_label']}", va="center", fontsize=9.5, fontweight="bold")
        axB.text(xi + 0.10, row["ypos"] - 0.28, f"{row['pathway_group']}", va="center", fontsize=8.0, color="#555555")

    axB.set_xlim(-0.15, 1.55)
    axB.set_ylim(-0.7, len(order)-0.3)
    axB.set_yticks([])
    axB.set_xticks([0, 1])
    axB.set_xticklabels(axis_order, rotation=12, ha="right")
    axB.grid(axis="x", linestyle=":", alpha=0.35)
    axB.set_title("B. Conceptual convergence of ML panel genes", fontsize=11, loc="left", fontweight="bold")
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.spines["left"].set_visible(False)

    dir_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor="#E76F51", markersize=8, label='Up in T2D'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor="#4C78A8", markersize=8, label='Down in T2D'),
    ]
    size_handles = [
        plt.scatter([], [], s=250 + 1600*0.80, color="#CCCCCC", edgecolor="white", label="AUC ≈ 0.80"),
        plt.scatter([], [], s=250 + 1600*0.90, color="#CCCCCC", edgecolor="white", label="AUC ≈ 0.90"),
    ]
    leg1 = axB.legend(handles=dir_handles, loc="upper left", frameon=False, fontsize=8.5, title="Direction")
    axB.add_artist(leg1)
    axB.legend(handles=size_handles, loc="lower left", frameon=False, fontsize=8.5, title="Bubble size")

    plt.tight_layout()
    for fmt in ["png", "svg"]:
        plt.savefig(os.path.join(FIGURES_DIR, f"figure9_convergence.{fmt}"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dir(FIGURES_DIR)
    build_figure7()
    build_figure8()
    build_figure9()
    print(f"Saved Figures 7–9 to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
