# T2D islet ML panel — reproducible manuscript pipeline

Reproducible repository for the machine-learning and external-validation components of the manuscript on human pancreatic islet transcriptomics in T2D.

This repository is organized around the **curated processed discovery inputs** actually used in the paper:

- `data/processed/GSE164416_expr_normalized.csv`
- `data/processed/GSE164416_labels.csv`

where labels are:

- `1` = T2D
- `0` = ND
- negative values = excluded intermediate classes

They can be obtained by running the pipeline:
https://github.com/ricardo-romero-ochoa/T2D-islet-integrative

The default workflow reproduces the paper-facing machine-learning analyses from those curated inputs, rather than trying to relabel the discovery cohort from raw GEO metadata.

## What this repo reproduces

Discovery ML workflow:
- DEG analysis on the curated ND/T2D discovery subset
- three-method feature selection: **LASSO + SVM-RFE + Random Forest**
- final 10-gene panel export
- repeated 5-fold CV model training
- **LOOCV** primary validation
- leakage-verification sensitivity analysis
- manuscript-facing **Figures 7–9**

External validation workflow:
- independent testing in **GSE50244**
- overlap-restricted reduced score
- frozen discovery-threshold analysis
- distribution-shift analysis
- recalibration sensitivity analyses
- manuscript-facing **Figure 10**

## Quick start

1. Install requirements

```bash
pip install -r requirements.txt
```

2. Put the curated discovery files here

```text
data/processed/GSE164416_expr_normalized.csv
data/processed/GSE164416_labels.csv
```

3. Run the complete manuscript-facing pipeline

```bash
bash run_manuscript_reproducibility.sh
```

If you have a manually curated external label override for GSE50244:

```bash
bash run_manuscript_reproducibility.sh --external-meta-override path/to/curated_gse50244_labels.csv
```

## Figures note

Not all figures produced by this pipeline will be used in the main text, some will be used as supplmentary material, so numbering may change.

## Main outputs

### Core results
- `results/deg_results.csv`
- `results/deg_list.csv`
- `results/final_gene_panel.csv`
- `results/model_cv_performance.csv`
- `results/loocv_performance.csv`
- `results/loocv_per_sample.csv`

### Figure inputs
- `results/figure_inputs/fig7_*`
- `results/figure_inputs/fig8_*`
- `results/figure_inputs/fig9_*`

### External validation outputs
- `results/external_validation/gse50244_initial/`
- `results/external_validation/gse50244_fixed_threshold/`
- `results/external_validation/gse50244_distribution_shift/`
- `results/external_validation/gse50244_recalibration/`

### Final paper-facing figures
- `figures/figure7_ml_pipeline.png`
- `figures/figure8_panel_network.png`
- `figures/figure9_convergence.png`
- `figures/figure10_external_transportability.png`

## Figure map

- **Figure 7**: `06_visualization/build_figures_7_9.py`
- **Figure 8**: `06_visualization/build_figures_7_9.py`
- **Figure 9**: `06_visualization/build_figures_7_9.py`
- **Figure 10**: `08_external_validation/build_figure10.py`

The earlier discovery plots (volcano, repeated-CV ROC, feature-importance plots, etc.) are also generated and can be used as supplementary or diagnostic figures.

## Notes on reproducibility

- The discovery workflow intentionally uses the **curated processed labels** from the manuscript, which are included, not raw GEO heuristic relabeling.
- The external GSE50244 workflow downloads public GEO resources automatically where possible.
- The full 10-gene panel cannot be tested one-to-one externally in GSE50244 because two non-coding features are not portable across annotation frameworks; the external pipeline therefore reproduces the paper’s **8-gene overlap-restricted reduced-score** analysis.

## Optional checks

After the run, compare key results to the manuscript-facing targets:

```bash
python 09_reproducibility/check_expected_results.py
```

## Legacy raw-data path

The original raw GEO download / labeling scripts remain in `01_data/` and `02_preprocessing/`, but they are **not** the default route for reproducing the manuscript’s final machine-learning results.
