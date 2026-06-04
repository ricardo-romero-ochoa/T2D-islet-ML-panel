# Data layout for manuscript reproduction

## Primary mode used in the paper

The manuscript-facing reproducibility workflow expects these curated processed discovery files:

```text
data/processed/GSE164416_expr_normalized.csv
data/processed/GSE164416_labels.csv
```

with labels encoded as:

- `1` = T2D
- `0` = ND
- negative values = excluded intermediate classes

This is the **recommended** route for reproducing the published ML results and manuscript figures.

## Optional raw-data rebuild

The repository still contains the older raw-data path for GEO download, HTSeq conversion, metadata parsing, and QC/normalization. Use that path only if you intentionally want to regenerate the processed discovery matrix from scratch.

Those scripts are:
- `01_data/download_geo.py`
- `01_data/convert_gse164416.py`
- `01_data/label_samples.py`
- `02_preprocessing/qc_normalize.py`

## External validation data

External validation in GSE50244 is handled automatically by the scripts in:

- `08_external_validation/`

Those scripts download and parse the public GEO resources directly and write outputs to:

```text
results/external_validation/
```
