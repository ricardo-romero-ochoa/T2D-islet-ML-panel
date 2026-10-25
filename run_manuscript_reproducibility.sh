#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

FAST_MODE="--skip-tuning"
EXTERNAL_META_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-tuning) FAST_MODE=""; shift ;;
    --external-meta-override) EXTERNAL_META_OVERRIDE="--external-meta-override $2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

req=(data/processed/GSE164416_expr_normalized.csv data/processed/GSE164416_labels.csv)
for f in "${req[@]}"; do [[ -f "$f" ]] || { echo "Missing required curated input: $f"; exit 1; }; done
mkdir -p results/nested_validation results/external_validation figures logs

echo "[1/11] Full-cohort DEG analysis (descriptive)"
python 03_feature_selection/deg_analysis.py --dataset GSE164416

echo "[2/11] Full-cohort feature selection (descriptive panel definition)"
python 03_feature_selection/feature_selection.py --dataset GSE164416

echo "[3/11] Fully nested stratified 5-fold validation (PRIMARY internal estimate)"
python 05_validation/nested_5fold_validation.py \
  --expr data/processed/GSE164416_expr_normalized.csv \
  --labels data/processed/GSE164416_labels.csv \
  --outdir results/nested_validation

echo "[4/11] Repeated-CV model training (sensitivity)"
python 04_models/train_models.py --dataset GSE164416 $FAST_MODE

echo "[5/11] Fixed-panel LOOCV (descriptive sensitivity only)"
python 05_validation/loocv_validation.py

echo "[6/11] Additional leakage/normalization sensitivity checks"
python 05_validation/leakage_verification.py

echo "[7/11] Gene-level AUC and supplementary figure inputs"
python 06_visualization/plot_gene_auc.py
python 07_reporting/export_figure_inputs_7_9.py
python 06_visualization/build_figures_7_9.py

echo "[8/11] Main Figure 7 from nested validation"
python 06_visualization/make_figure7_nested.py \
  --figure-inputs results/figure_inputs \
  --nested-dir results/nested_validation \
  --outdir figures

echo "[9/11] External validation in GSE50244"
python 08_external_validation/validate_gse50244_external_fixed.py \
  --discovery-expr data/processed/GSE164416_expr_normalized.csv \
  --discovery-meta data/processed/GSE164416_labels.csv \
  --outdir results/external_validation/gse50244_initial \
  --sample-col sample_id --group-col label --discovery-transform none ${EXTERNAL_META_OVERRIDE}
python 08_external_validation/validate_gse50244_reduced_score_fixed_threshold.py \
  --discovery-expr data/processed/GSE164416_expr_normalized.csv \
  --discovery-meta data/processed/GSE164416_labels.csv \
  --overlap-table results/external_validation/gse50244_initial/overlap_table.csv \
  --outdir results/external_validation/gse50244_fixed_threshold \
  --sample-col sample_id --group-col label --discovery-transform none \
  --threshold-method train_youden ${EXTERNAL_META_OVERRIDE}

echo "[10/11] Distribution shift and recalibration"
python 08_external_validation/analyze_reduced_score_distribution_shift.py \
  --discovery-scores results/external_validation/gse50244_fixed_threshold/discovery_reduced_scores.csv \
  --external-scores results/external_validation/gse50244_fixed_threshold/external_reduced_scores.csv \
  --metrics-summary results/external_validation/gse50244_fixed_threshold/metrics_summary.txt \
  --outdir results/external_validation/gse50244_distribution_shift
python 08_external_validation/recalibration_analysis.py \
  --discovery-scores results/external_validation/gse50244_fixed_threshold/discovery_reduced_scores.csv \
  --external-scores results/external_validation/gse50244_fixed_threshold/external_reduced_scores.csv \
  --outdir results/external_validation/gse50244_recalibration

echo "[11/11] Reproducibility check"
python 09_reproducibility/check_expected_results.py

echo "Pipeline complete. Primary internal result: results/nested_validation/nested_5fold_summary.json"
