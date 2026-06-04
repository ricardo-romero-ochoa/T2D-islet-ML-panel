# config.py — Central configuration for T2D Islet ML Pipeline
# Edit dataset lists, thresholds, and hyperparameters here.
# All scripts import from this file; no hardcoded paths elsewhere.

import os

# ── Project Paths ─────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_RAW       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
FIGURES_DIR    = os.path.join(BASE_DIR, "figures")
LOGS_DIR       = os.path.join(BASE_DIR, "logs")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
FIGURE_INPUTS_DIR = os.path.join(RESULTS_DIR, "figure_inputs")

for d in [DATA_RAW, DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR, FIGURE_INPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── GEO Datasets ──────────────────────────────────────────────────────────────
DISCOVERY_DATASETS = ["GSE164416"]

VALIDATION_DATASETS = []   # No compatible external cohorts available at time of analysis.
                            # See manuscript Methods section for justification.

SUPPLEMENTARY_DATASETS = [
    "GSE163980", "GSE68224", "GSE281291", "GSE262614",
    "GSE166652", "GSE166502",
]

ALL_DATASETS = DISCOVERY_DATASETS + VALIDATION_DATASETS + SUPPLEMENTARY_DATASETS

# ── Sample Label Mapping ───────────────────────────────────────────────────────
# Keywords matched against Sample_title / Sample_characteristics fields.
# -1 = exclude (intermediate phenotype); 0 = control; 1 = T2D
LABEL_MAPS = {
    "GSE164416": {
        "_T2D":  1,  "_ND":   0,  "T2D":   1,  "ND":    0,
        "_IGT": -1,  "_IFG": -1,  "_T3cD": -1,
        "IGT":  -1,  "IFG":  -1,  "T3cD":  -1,
    },
    "GSE163980":  {"T2D": 1, "diabetic": 1, "control": 0, "normal": 0, "healthy": 0},
    "GSE68224":   {"T2D": 1, "T2DM": 1, "diabetic": 1, "control": 0, "normal": 0},
    "GSE281291":  {"T2D": 1, "diabetic": 1, "control": 0, "normal": 0},
    "GSE262614":  {"T2D": 1, "diabetic": 1, "control": 0, "normal": 0},
    "GSE166652":  {"T2D": 1, "diabetic": 1, "control": 0, "normal": 0},
    "GSE166502":  {"T2D": 1, "diabetic": 1, "control": 0, "normal": 0},
}

# ── QC Thresholds ─────────────────────────────────────────────────────────────
MIN_DETECTION_FRACTION = 0.5    # Gene must be detected in ≥50% of samples
LOG2_TRANSFORM         = True   # Apply log2(x+1) if data is not log-scaled
QUANTILE_NORMALIZE     = False  # IMPORTANT: False for RNA-seq classification.
                                 # Quantile norm equalises distributions and
                                 # destroys between-group biological signal.

# ── DEG Thresholds ────────────────────────────────────────────────────────────
DEG_FDR_CUTOFF   = 0.01    # Benjamini-Hochberg adjusted p-value
DEG_LOGFC_CUTOFF = 1.5     # |log2FC| threshold

# ── Feature Selection ─────────────────────────────────────────────────────────
LASSO_CV_FOLDS      = 10
SVM_RFE_STEP        = 0.1
SVM_RFE_CV          = 5
RF_N_ESTIMATORS     = 500
TOP_N_FEATURES      = 50
CONSENSUS_MIN_VOTES = 2         # Gene must be selected by ≥2 of 3 methods
FINAL_PANEL_SIZE    = 10

# ── Model Training ────────────────────────────────────────────────────────────
CV_FOLDS      = 5
CV_REPEATS    = 20              # Repeated CV for performance stability
RANDOM_STATE  = 42
TEST_SIZE     = 0.25

SVM_PARAM_GRID = {"C": [0.01, 0.1, 1, 10, 100], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]}
RF_PARAM_GRID  = {"n_estimators": [100, 300, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 5], "max_features": ["sqrt", "log2"]}
LR_PARAM_GRID  = {"C": [0.001, 0.01, 0.1, 1, 10], "penalty": ["l1", "l2"], "solver": ["liblinear"]}

# ── Figure Settings ───────────────────────────────────────────────────────────
FIGURE_DPI    = 300
FIGURE_FORMAT = ["png", "svg"]
PALETTE_T2D   = "#E74C3C"
PALETTE_CTRL  = "#2980B9"
FONT_FAMILY   = "Arial"
FONT_SIZE     = 12
