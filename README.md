[![DOI](https://zenodo.org/badge/1259762344.svg)](https://doi.org/10.5281/zenodo.20549014)

# T2D islet ML panel — reproducible manuscript pipeline

Reproducible repository for the machine-learning and GSE50244 external-validation components of the manuscript **Human Islet Transcriptomics in Type 2 Diabetes Reveals Conserved Dysfunction Axes and a Candidate Gene Panel**.

## Primary validation in the current manuscript

The primary internal estimate is **fully nested stratified 5-fold cross-validation**. Within every outer training fold the workflow repeats:

1. DEG calling;
2. LASSO, SVM-RFE and Random Forest feature selection;
3. consensus-panel construction;
4. scaling; and
5. soft-voting ensemble fitting.

The held-out outer fold is used only for prediction. This design prevents feature-selection leakage from the full discovery cohort.

Current primary internal performance:

- pooled out-of-fold AUC: **0.960**;
- stratified-bootstrap 95% CI: **0.900–0.997**;
- sensitivity: **0.923**;
- specificity: **0.833**;
- MCC: **0.756**.

The older fixed-panel LOOCV result (AUC = 1.000) is retained only as a descriptive sensitivity analysis and is not the primary performance estimate.

## Feature-selection stability

Across the five outer folds, the most stable final-panel features were:

- GABRA2: 5/5 folds;
- ENSG00000284653: 5/5;
- SLC2A2: 4/5;
- HHATL, ARG2, PRIMA1 and ENSG00000227589: 3/5 each;
- DKK3 and RNU1-70P: 2/5;
- TAFA4: 1/5.

Fold-specific panels and complete stability results are in `results/nested_validation/`.

## External validation

The independent GSE50244 validation uses an **8-gene annotation-compatible score drawn from the broader consensus feature pool**, not a literal subset of the final ranked 10-feature panel. The portable genes are:

`DKK3, PRIMA1, TAFA4, HHATL, PARVG, GABRA2, SLC2A2, ARG2`.

External discrimination is AUC **0.907** with DeLong 95% CI **0.818–0.996** (51 ND, 11 T2D). Rank discrimination transports well, whereas the frozen discovery threshold does not because of score-location shift; target-cohort recalibration is therefore required for thresholded use.

## Curated discovery inputs

The manuscript-facing workflow uses the curated processed GSE164416 files:

```text
data/processed/GSE164416_expr_normalized.csv
data/processed/GSE164416_labels.csv
```

Labels are `1 = T2D`, `0 = ND`; negative values represent excluded intermediate classes.

## Run the manuscript-facing workflow

```bash
pip install -r requirements.txt
bash run_manuscript_reproducibility.sh
```

The runner executes the full-cohort descriptive analysis, the fully nested primary validation, fixed-panel sensitivity analyses, and GSE50244 external validation.

## Key outputs

```text
results/final_gene_panel.csv
results/nested_validation/nested_5fold_metrics.csv
results/nested_validation/nested_5fold_fold_metrics.csv
results/nested_validation/nested_5fold_predictions_and_panels.csv
results/nested_validation/nested_feature_selection_stability.csv
results/nested_validation/nested_auc_bootstrap_distribution.csv
results/external_validation/gse50244_fixed_threshold/
```

## Manuscript figure mapping

See `MANUSCRIPT_FIGURE_MAP.md`. In the current manuscript:

- **Figure 7**: fully nested ML validation and feature stability;
- **Figure 8**: GSE50244 external transportability/calibration shift;
- **Supplementary Figure S1**: candidate-panel biological/network context;
- **Supplementary Figure S2**: convergence between module and ML frameworks.

The single-cell Figure 10 is maintained in the companion `scAnalysis-T2D-Islets` repository.

## Reproducibility check

```bash
python 09_reproducibility/check_expected_results.py
```

The check uses the nested AUC (0.960) as the primary internal target and the GSE50244 AUC (0.907) as the external target.

## Legacy analyses

Repeated CV, fixed-panel LOOCV, and the earlier leakage-verification scripts remain available for sensitivity and historical reproducibility. They should not be interpreted as the primary estimate of model generalization.

## Citation

Use the Zenodo concept DOI: **10.5281/zenodo.20549014**. `CITATION.cff` reflects the current GitHub revision; a new version-specific Zenodo DOI can be added after archiving this release.
