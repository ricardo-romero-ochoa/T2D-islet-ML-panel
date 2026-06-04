#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Primary reproducibility runner for the manuscript.
# Assumes curated processed discovery inputs are already present:
#   data/processed/GSE164416_expr_normalized.csv
#   data/processed/GSE164416_labels.csv
# Optional: external-label override for GSE50244 can be passed via --external-meta-override

FAST_MODE="--skip-tuning"
EXTERNAL_META_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-tuning) FAST_MODE=""; shift ;;
    --external-meta-override) EXTERNAL_META_OVERRIDE="--external-meta-override $2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

req=(
  data/processed/GSE164416_expr_normalized.csv
  data/processed/GSE164416_labels.csv
)
for f in "${req[@]}"; do
  [[ -f "$f" ]] || { echo "Missing required curated input: $f"; exit 1; }
done

mkdir -p results/external_validation figures logs

echo "[1/10] DEG analysis"
python 03_feature_selection/deg_analysis.py --dataset GSE164416

echo "[2/10] Feature selection"
python 03_feature_selection/feature_selection.py --dataset GSE164416

echo "[3/10] Repeated-CV model training"
python 04_models/train_models.py --dataset GSE164416 $FAST_MODE

echo "[4/10] LOOCV validation"
python 05_validation/loocv_validation.py

echo "[5/10] Leakage verification"
python 05_validation/leakage_verification.py

echo "[6/10] Gene-level AUC and manuscript figure inputs"
python 06_visualization/plot_gene_auc.py
python 07_reporting/export_figure_inputs_7_9.py
python 06_visualization/build_figures_7_9.py

echo "[7/10] External validation in GSE50244"
python 08_external_validation/validate_gse50244_external_fixed.py       --discovery-expr data/processed/GSE164416_expr_normalized.csv       --discovery-meta data/processed/GSE164416_labels.csv       --outdir results/external_validation/gse50244_initial       --sample-col sample_id       --group-col label       --discovery-transform none       ${EXTERNAL_META_OVERRIDE}

echo "[8/10] Frozen-threshold reduced-score validation"
python 08_external_validation/validate_gse50244_reduced_score_fixed_threshold.py       --discovery-expr data/processed/GSE164416_expr_normalized.csv       --discovery-meta data/processed/GSE164416_labels.csv       --overlap-table results/external_validation/gse50244_initial/overlap_table.csv       --outdir results/external_validation/gse50244_fixed_threshold       --sample-col sample_id       --group-col label       --discovery-transform none       --threshold-method train_youden       ${EXTERNAL_META_OVERRIDE}

echo "[9/10] Distribution shift and recalibration"
python 08_external_validation/analyze_reduced_score_distribution_shift.py       --discovery-scores results/external_validation/gse50244_fixed_threshold/discovery_reduced_scores.csv       --external-scores results/external_validation/gse50244_fixed_threshold/external_reduced_scores.csv       --metrics-summary results/external_validation/gse50244_fixed_threshold/metrics_summary.txt       --outdir results/external_validation/gse50244_distribution_shift

python 08_external_validation/recalibration_analysis.py       --discovery-scores results/external_validation/gse50244_fixed_threshold/discovery_reduced_scores.csv       --external-scores results/external_validation/gse50244_fixed_threshold/external_reduced_scores.csv       --outdir results/external_validation/gse50244_recalibration

echo "[10/10] Figure 10"
python 08_external_validation/build_figure10.py       --discovery-scores results/external_validation/gse50244_fixed_threshold/discovery_reduced_scores.csv       --external-scores results/external_validation/gse50244_fixed_threshold/external_reduced_scores.csv       --metrics-json results/external_validation/gse50244_fixed_threshold/metrics_summary.json       --outdir figures

echo
echo "Reproducibility pipeline complete. Key outputs:"
echo "  results/final_gene_panel.csv"
echo "  results/loocv_performance.csv"
echo "  results/figure_inputs/"
echo "  results/external_validation/gse50244_fixed_threshold/"
echo "  figures/figure7_ml_pipeline.png"
echo "  figures/figure8_panel_network.png"
echo "  figures/figure9_convergence.png"
echo "  figures/figure10_external_transportability.png"
