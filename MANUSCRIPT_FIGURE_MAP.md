# Manuscript figure map

## Figure 7 — Machine learning diagnostic pipeline
Built by:
- `03_feature_selection/feature_selection.py`
- `05_validation/loocv_validation.py`
- `05_validation/leakage_verification.py`
- `06_visualization/plot_gene_auc.py`
- `07_reporting/export_figure_inputs_7_9.py`
- `06_visualization/build_figures_7_9.py`

Main inputs:
- `results/figure_inputs/fig7_feature_selection_membership.csv`
- `results/figure_inputs/fig7_single_gene_auc.csv`
- `results/figure_inputs/fig7_loocv_roc_points.csv`
- `results/figure_inputs/fig7_leakage_verification_metrics.csv`
- `results/figure_inputs/fig7_proper_loocv_roc_points.csv`
- `results/figure_inputs/fig7_leakage_global_qn_roc_points.csv`

## Figure 8 — Panel network and pathway context
Built by:
- `07_reporting/export_figure_inputs_7_9.py`
- `06_visualization/build_figures_7_9.py`

Main inputs:
- `results/figure_inputs/fig8_nodes.csv`
- `results/figure_inputs/fig8_edges.csv`

## Figure 9 — Convergence between meta-analysis and ML frameworks
Built by:
- `07_reporting/export_figure_inputs_7_9.py`
- `06_visualization/build_figures_7_9.py`

Main inputs:
- `results/figure_inputs/fig9_module_gene_sets.csv`
- `results/figure_inputs/fig9_panel_module_overlap.csv`
- `results/figure_inputs/fig9_bubble_table.csv`

## Figure 10 — External transportability and calibration-shift analysis
Built by:
- `08_external_validation/validate_gse50244_external_fixed.py`
- `08_external_validation/validate_gse50244_reduced_score_fixed_threshold.py`
- `08_external_validation/analyze_reduced_score_distribution_shift.py`
- `08_external_validation/recalibration_analysis.py`
- `08_external_validation/build_figure10.py`
