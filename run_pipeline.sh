#!/usr/bin/env bash
# Backward-compatible wrapper.
# The manuscript now uses curated processed discovery inputs by default.
# For a clean, paper-facing end-to-end run use:
#   bash run_manuscript_reproducibility.sh
set -euo pipefail
echo "This wrapper now delegates to the manuscript reproducibility runner."
echo "Primary mode expects curated processed inputs in data/processed/."
bash run_manuscript_reproducibility.sh "$@"
